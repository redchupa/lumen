//! End-to-end correctness for Q8_0 dequantization on the self-hosted
//! x86_64 backend. Path:
//!
//! ```text
//! IR(Param q8_0 -> Dequantize -> Return f32)
//!   → x86_64 emit
//!   → JIT region
//!   → call
//!   → compare against the pure-Rust reference in `lumen-runtime::quant`.
//! ```

#![cfg(target_arch = "x86_64")]

use lumen_codegen::backend::{Backend, CodegenOpts};
use lumen_codegen::x86_64::X86_64;
use lumen_ir::ty::{DType, Dim, Shape, TensorType};
use lumen_ir::{Function, IrModule, Op, Value};
use lumen_jit::ExecRegion;
use lumen_runtime::quant::{dequantize_q8_0, quantize_q8_0, BlockQ8_0, QK};

fn build_dequant_q8_ir(num_elements: u32) -> IrModule {
    let in_ty = TensorType {
        dtype: DType::Q8_0,
        shape: Shape(vec![Dim::Static(num_elements)]),
    };
    let out_ty = TensorType {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Static(num_elements)]),
    };
    let mut f = Function::new("dequant_q8", vec![in_ty.clone()], out_ty.clone());
    let v_in = f.param_values[0];
    let v_out = f.push(Value {
        op: Op::Dequantize { x: v_in },
        ty: out_ty,
    });
    let placeholder = TensorType {
        dtype: DType::F32,
        shape: Shape(vec![]),
    };
    f.push(Value {
        op: Op::Return { value: v_out },
        ty: placeholder,
    });

    IrModule { functions: vec![f] }
}

unsafe fn jit_run_dequant(num_elements: u32, blocks: &[BlockQ8_0]) -> Vec<f32> {
    let ir = build_dequant_q8_ir(num_elements);
    let backend = X86_64::host();
    let mc = backend.lower(&ir, &CodegenOpts::default()).expect("lower");
    let region = ExecRegion::from_machine_code(&mc).expect("exec region");
    // SAFETY: the signature matches the emitted function.
    let f: unsafe extern "C" fn(*const BlockQ8_0, *mut f32) = unsafe { region.as_fn() };
    let mut out = vec![0.0f32; num_elements as usize];
    unsafe {
        f(blocks.as_ptr(), out.as_mut_ptr());
    }
    out
}

fn make_quantized(input: &[f32]) -> Vec<BlockQ8_0> {
    assert!(input.len() % QK == 0);
    let nb = input.len() / QK;
    let mut blocks = vec![BlockQ8_0 { d: 0, qs: [0; 32] }; nb];
    quantize_q8_0(input, &mut blocks);
    blocks
}

#[test]
fn dequant_q8_one_block_matches_reference() {
    let input: Vec<f32> = (0..32).map(|i| ((i as f32) / 32.0) - 0.5).collect();
    let blocks = make_quantized(&input);

    let mut reference = vec![0.0f32; 32];
    dequantize_q8_0(&blocks, &mut reference);

    let native = unsafe { jit_run_dequant(32, &blocks) };

    for (i, (g, w)) in native.iter().zip(reference.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-5,
            "block 0 idx {}: native {} reference {}",
            i,
            g,
            w
        );
    }
}

#[test]
fn dequant_q8_eight_blocks_matches_reference() {
    let n = 32 * 8; // 256 elements
    let input: Vec<f32> = (0..n).map(|i| ((i % 17) as f32) * 0.07 - 1.0).collect();
    let blocks = make_quantized(&input);

    let mut reference = vec![0.0f32; n];
    dequantize_q8_0(&blocks, &mut reference);

    let native = unsafe { jit_run_dequant(n as u32, &blocks) };

    for (i, (g, w)) in native.iter().zip(reference.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-5,
            "idx {}: native {} reference {}",
            i,
            g,
            w
        );
    }
}
