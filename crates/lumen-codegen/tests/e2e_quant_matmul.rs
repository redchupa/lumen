//! End-to-end correctness for the fused Q8_0 × F32 matmul.
//!
//! The Lumen pipeline:
//! ```text
//!   IR: Param(q8_0 [M,K]), Param(f32 [K,N]), Dequantize, MatMul, Return
//!   → x86_64 detects the pattern as one fused kernel
//!   → emits a single loop that loads i8 + fp16 d, scales on the fly,
//!     broadcasts, and accumulates with FMA. Never materializes the full
//!     dequantized weight matrix.
//!   → JIT region, call.
//! ```
//!
//! Reference: pure Rust naive `dequant → matmul` two-pass.

#![cfg(target_arch = "x86_64")]

use lumen_codegen::backend::{Backend, CodegenOpts};
use lumen_codegen::x86_64::X86_64;
use lumen_ir::ty::{DType, Dim, Shape, TensorType};
use lumen_ir::{Function, IrModule, Op, Value};
use lumen_jit::ExecRegion;
use lumen_runtime::quant::{dequantize_q8_0, quantize_q8_0, BlockQ8_0, QK};

fn naive_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn build_quant_matmul_ir(m: u32, k: u32, n: u32) -> IrModule {
    let w_ty = TensorType {
        dtype: DType::Q8_0,
        shape: Shape(vec![Dim::Static(m), Dim::Static(k)]),
    };
    let a_ty = TensorType {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Static(k), Dim::Static(n)]),
    };
    let dq_ty = TensorType {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Static(m), Dim::Static(k)]),
    };
    let out_ty = TensorType {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Static(m), Dim::Static(n)]),
    };

    let mut f = Function::new("quant_matmul", vec![w_ty, a_ty], out_ty.clone());
    let v_w = f.param_values[0];
    let v_a = f.param_values[1];
    let v_dq = f.push(Value {
        op: Op::Dequantize { x: v_w },
        ty: dq_ty,
    });
    let v_mm = f.push(Value {
        op: Op::MatMul {
            lhs: v_dq,
            rhs: v_a,
        },
        ty: out_ty,
    });
    let placeholder = TensorType {
        dtype: DType::F32,
        shape: Shape(vec![]),
    };
    f.push(Value {
        op: Op::Return { value: v_mm },
        ty: placeholder,
    });

    IrModule { functions: vec![f] }
}

unsafe fn jit_run_quant_matmul(
    m: u32,
    k: u32,
    n: u32,
    weights: &[BlockQ8_0],
    activations: &[f32],
) -> Vec<f32> {
    let ir = build_quant_matmul_ir(m, k, n);
    let backend = X86_64::host();
    let mc = backend.lower(&ir, &CodegenOpts::default()).expect("lower");
    let region = ExecRegion::from_machine_code(&mc).expect("exec region");
    let mut out = vec![0.0f32; (m * n) as usize];
    unsafe {
        let f: unsafe extern "C" fn(*const BlockQ8_0, *const f32, *mut f32) = region.as_fn();
        f(weights.as_ptr(), activations.as_ptr(), out.as_mut_ptr());
    }
    out
}

fn make_weights(m: usize, k: usize) -> (Vec<BlockQ8_0>, Vec<f32>) {
    assert!(k % QK == 0);
    let w_fp32: Vec<f32> = (0..m * k).map(|i| ((i % 23) as f32) * 0.01 - 0.1).collect();
    let num_blocks = m * k / QK;
    let mut blocks = vec![BlockQ8_0 { d: 0, qs: [0; 32] }; num_blocks];
    quantize_q8_0(&w_fp32, &mut blocks);
    // Reference fp32 weights (dequantized once, used by the naive reference).
    let mut dequant_ref = vec![0.0f32; m * k];
    dequantize_q8_0(&blocks, &mut dequant_ref);
    (blocks, dequant_ref)
}

#[test]
fn quant_matmul_q8_1x32x8_matches_reference() {
    // Smallest case: 1 row of weights, K=32 (one Q8_0 block), N=8.
    let m = 1;
    let k = 32;
    let n = 8;

    let (blocks, w_dequant) = make_weights(m as usize, k as usize);
    let a: Vec<f32> = (0..(k * n) as usize)
        .map(|i| ((i % 7) as f32) * 0.13 + 0.05)
        .collect();

    let reference = naive_matmul(&w_dequant, &a, m as usize, k as usize, n as usize);

    let native = unsafe { jit_run_quant_matmul(m, k, n, &blocks, &a) };

    for (idx, (g, w)) in native.iter().zip(reference.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-3,
            "idx {}: native {} reference {}",
            idx,
            g,
            w
        );
    }
}

#[test]
fn quant_matmul_q8_4x64x16_matches_reference() {
    let m = 4;
    let k = 64; // 2 Q8_0 blocks per row
    let n = 16; // 2 ymm columns

    let (blocks, w_dequant) = make_weights(m as usize, k as usize);
    let a: Vec<f32> = (0..(k * n) as usize)
        .map(|i| ((i % 11) as f32) * 0.07 - 0.3)
        .collect();

    let reference = naive_matmul(&w_dequant, &a, m as usize, k as usize, n as usize);

    let native = unsafe { jit_run_quant_matmul(m, k, n, &blocks, &a) };

    for (idx, (g, w)) in native.iter().zip(reference.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-2,
            "idx {}: native {} reference {}",
            idx,
            g,
            w
        );
    }
}

#[test]
fn quant_matmul_q8_8x128x32_matches_reference() {
    let m = 8;
    let k = 128; // 4 blocks per row
    let n = 32; // 4 ymm columns

    let (blocks, w_dequant) = make_weights(m as usize, k as usize);
    let a: Vec<f32> = (0..(k * n) as usize)
        .map(|i| ((i % 19) as f32) * 0.02 + 0.01)
        .collect();

    let reference = naive_matmul(&w_dequant, &a, m as usize, k as usize, n as usize);

    let native = unsafe { jit_run_quant_matmul(m, k, n, &blocks, &a) };

    for (idx, (g, w)) in native.iter().zip(reference.iter()).enumerate() {
        assert!(
            (g - w).abs() < 5e-2,
            "idx {}: native {} reference {}",
            idx,
            g,
            w
        );
    }
}
