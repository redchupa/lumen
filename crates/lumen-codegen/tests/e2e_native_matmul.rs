//! End-to-end correctness for the self-hosted x86_64 backend.
//!
//! Path: Lumen source → IR → x86_64 machine code (via [`lumen_codegen::x86_64`])
//!   → JIT region (via [`lumen_jit::ExecRegion`]) → call → compare against
//!   a naive Rust reference.
//!
//! Runs only on `cfg(target_arch = "x86_64")`. On other arches it compiles
//! the backend but skips execution.

#![cfg(target_arch = "x86_64")]

use lumen_codegen::backend::{Backend, CodegenOpts};
use lumen_codegen::x86_64::X86_64;
use lumen_dsl::Parser;
use lumen_ir::lower;
use lumen_jit::ExecRegion;

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

fn compile_and_run(src: &str, a: &[f32], b: &[f32], out: &mut [f32]) {
    let module = Parser::parse(src).unwrap();
    let ir = lower(&module).unwrap();

    let backend = X86_64::host();
    let mc = backend
        .lower(&ir, &CodegenOpts::default())
        .expect("x86_64 lower");

    let region = ExecRegion::from_machine_code(&mc).expect("exec region");
    // SAFETY: we just emitted this function and its signature matches.
    unsafe {
        let f: unsafe extern "C" fn(*const f32, *const f32, *mut f32) = region.as_fn();
        f(a.as_ptr(), b.as_ptr(), out.as_mut_ptr());
    }
}

#[test]
fn native_matmul_4x4x4() {
    let src = r#"
        fn matmul(
            a: tensor<f32, [4, 4]>,
            b: tensor<f32, [4, 4]>,
        ) -> tensor<f32, [4, 4]> {
            return a @ b;
        }
    "#;
    let a: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5).collect();
    let b: Vec<f32> = (0..16).map(|i| (i as f32) * 0.25 + 1.0).collect();
    let mut out = vec![0.0f32; 16];

    compile_and_run(src, &a, &b, &mut out);

    let reference = naive_matmul(&a, &b, 4, 4, 4);
    for (i, (got, want)) in out.iter().zip(reference.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "mismatch at {}: got {} expected {}",
            i,
            got,
            want
        );
    }
}

#[test]
fn native_matmul_8x8x8_matches_c_backend_inputs() {
    // Same inputs as the C backend e2e for cross-backend consistency.
    let src = r#"
        fn matmul(
            a: tensor<f32, [8, 8]>,
            b: tensor<f32, [8, 8]>,
        ) -> tensor<f32, [8, 8]> {
            return a @ b;
        }
    "#;
    let a: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05 + 1.0).collect();
    let mut out = vec![0.0f32; 64];

    compile_and_run(src, &a, &b, &mut out);

    let reference = naive_matmul(&a, &b, 8, 8, 8);
    for (got, want) in out.iter().zip(reference.iter()) {
        assert!((got - want).abs() < 1e-4, "got {} expected {}", got, want);
    }
}

#[test]
fn native_matmul_rectangular_4x5x3() {
    let src = r#"
        fn matmul(
            a: tensor<f32, [4, 5]>,
            b: tensor<f32, [5, 3]>,
        ) -> tensor<f32, [4, 3]> {
            return a @ b;
        }
    "#;
    let a: Vec<f32> = (0..20).map(|i| (i as f32) - 10.0).collect();
    let b: Vec<f32> = (0..15).map(|i| 1.0 / ((i + 1) as f32)).collect();
    let mut out = vec![0.0f32; 12];

    compile_and_run(src, &a, &b, &mut out);

    let reference = naive_matmul(&a, &b, 4, 5, 3);
    for (got, want) in out.iter().zip(reference.iter()) {
        assert!((got - want).abs() < 1e-4, "got {} expected {}", got, want);
    }
}
