//! End-to-end: build an IR `RmsNorm` function by hand, JIT it through the
//! self-hosted x86_64 backend, and check the result against
//! `lumen-runtime::ops::rms_norm`.

#![cfg(target_arch = "x86_64")]

use lumen_codegen::backend::{Backend, CodegenOpts};
use lumen_codegen::x86_64::X86_64;
use lumen_ir::ty::{DType, Dim, Shape, TensorType};
use lumen_ir::{Function, IrModule, Op, Value};
use lumen_jit::ExecRegion;
use lumen_runtime::ops::rms_norm;

fn build_rms_norm_ir(h: u32, eps: f32) -> IrModule {
    let ty = TensorType {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Static(h)]),
    };
    let mut f = Function::new("rms_norm", vec![ty.clone(), ty.clone()], ty.clone());
    let v_x = f.param_values[0];
    let v_w = f.param_values[1];
    let v_y = f.push(Value {
        op: Op::RmsNorm {
            x: v_x,
            weight: v_w,
            eps,
        },
        ty: ty.clone(),
    });
    let placeholder = TensorType {
        dtype: DType::F32,
        shape: Shape(vec![]),
    };
    f.push(Value {
        op: Op::Return { value: v_y },
        ty: placeholder,
    });
    IrModule { functions: vec![f] }
}

unsafe fn jit_run(h: u32, eps: f32, x: &[f32], w: &[f32]) -> Vec<f32> {
    let ir = build_rms_norm_ir(h, eps);
    let backend = X86_64::host();
    let mc = backend.lower(&ir, &CodegenOpts::default()).expect("lower");
    let region = ExecRegion::from_machine_code(&mc).expect("exec");
    let mut out = vec![0.0f32; h as usize];
    unsafe {
        let f: unsafe extern "C" fn(*const f32, *const f32, *mut f32) = region.as_fn();
        f(x.as_ptr(), w.as_ptr(), out.as_mut_ptr());
    }
    out
}

#[test]
fn native_rms_norm_h8_matches_reference() {
    let h = 8u32;
    let eps = 1e-5;
    let x: Vec<f32> = (0..h as usize).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let w: Vec<f32> = vec![1.0; h as usize];

    let mut want = vec![0.0f32; h as usize];
    rms_norm(&x, &w, &mut want, h as usize, eps);

    let got = unsafe { jit_run(h, eps, &x, &w) };
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-5,
            "idx {}: native {} reference {}",
            i,
            g,
            w
        );
    }
}

#[test]
fn native_rms_norm_h128_with_nontrivial_weight() {
    let h = 128u32;
    let eps = 1e-5;
    let x: Vec<f32> = (0..h as usize)
        .map(|i| ((i % 13) as f32) * 0.07 - 0.5)
        .collect();
    let w: Vec<f32> = (0..h as usize)
        .map(|i| 0.5 + (i as f32) / (h as f32))
        .collect();

    let mut want = vec![0.0f32; h as usize];
    rms_norm(&x, &w, &mut want, h as usize, eps);

    let got = unsafe { jit_run(h, eps, &x, &w) };
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-4,
            "idx {}: native {} reference {}",
            i,
            g,
            w
        );
    }
}

#[test]
fn native_rms_norm_h4096_like_llama_hidden() {
    // Llama-scale hidden dim. Catches accumulation issues that small sizes hide.
    let h = 4096u32;
    let eps = 1e-5;
    let x: Vec<f32> = (0..h as usize)
        .map(|i| ((i % 97) as f32) * 0.013 - 0.6)
        .collect();
    let w: Vec<f32> = (0..h as usize)
        .map(|i| 1.0 + ((i % 31) as f32) * 0.01)
        .collect();

    let mut want = vec![0.0f32; h as usize];
    rms_norm(&x, &w, &mut want, h as usize, eps);

    let got = unsafe { jit_run(h, eps, &x, &w) };
    // Larger H → more accumulation rounding; loosen tolerance.
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-3,
            "idx {}: native {} reference {}",
            i,
            g,
            w
        );
    }
}
