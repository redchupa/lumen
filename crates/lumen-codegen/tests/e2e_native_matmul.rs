//! End-to-end correctness for the self-hosted x86_64 backend.
//!
//! Path: Lumen source → IR → x86_64 machine code (scalar OR AVX2/FMA)
//!   → JIT region → call → compare against a naive Rust reference.
//!
//! For shapes where `N % 8 == 0`, the backend picks the AVX2 path automatically.
//! We test both:
//! - `host()` — automatic synthesis (AVX2 when applicable)
//! - `scalar_only()` — forces the scalar path
//!
//! and assert they give the same result.

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

fn compile_and_run(backend: &X86_64, src: &str, a: &[f32], b: &[f32], out: &mut [f32]) {
    let module = Parser::parse(src).unwrap();
    let ir = lower(&module).unwrap();
    let mc = backend.lower(&ir, &CodegenOpts::default()).expect("lower");
    let region = ExecRegion::from_machine_code(&mc).expect("exec region");
    // SAFETY: signature matches the emitted function.
    unsafe {
        let f: unsafe extern "C" fn(*const f32, *const f32, *mut f32) = region.as_fn();
        f(a.as_ptr(), b.as_ptr(), out.as_mut_ptr());
    }
}

#[test]
fn scalar_matmul_4x4x4() {
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

    // N=4, AVX2 path requires N % 8 == 0 → scalar path used.
    compile_and_run(&X86_64::host(), src, &a, &b, &mut out);

    let reference = naive_matmul(&a, &b, 4, 4, 4);
    for (got, want) in out.iter().zip(reference.iter()) {
        assert!((got - want).abs() < 1e-4);
    }
}

#[test]
fn avx2_matmul_8x8x8_matches_scalar_and_reference() {
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
    let reference = naive_matmul(&a, &b, 8, 8, 8);

    // N=8 → AVX2 path is picked.
    let mut out_avx2 = vec![0.0f32; 64];
    compile_and_run(&X86_64::host(), src, &a, &b, &mut out_avx2);

    // Force scalar.
    let mut out_scalar = vec![0.0f32; 64];
    compile_and_run(&X86_64::scalar_only(), src, &a, &b, &mut out_scalar);

    for i in 0..64 {
        assert!(
            (out_avx2[i] - reference[i]).abs() < 1e-3,
            "AVX2 diverges at {}: got {} expected {}",
            i,
            out_avx2[i],
            reference[i]
        );
        assert!(
            (out_scalar[i] - reference[i]).abs() < 1e-4,
            "scalar diverges at {}",
            i
        );
    }
}

#[test]
fn avx2_matmul_32x32x32_matches_reference() {
    let src = r#"
        fn matmul(
            a: tensor<f32, [32, 32]>,
            b: tensor<f32, [32, 32]>,
        ) -> tensor<f32, [32, 32]> {
            return a @ b;
        }
    "#;
    let a: Vec<f32> = (0..1024).map(|i| ((i % 17) as f32) * 0.01 - 0.5).collect();
    let b: Vec<f32> = (0..1024).map(|i| ((i % 13) as f32) * 0.03 + 0.1).collect();
    let reference = naive_matmul(&a, &b, 32, 32, 32);

    let mut out_avx2 = vec![0.0f32; 1024];
    compile_and_run(&X86_64::host(), src, &a, &b, &mut out_avx2);

    // Tolerance is looser because order-of-summation differs (32 FMAs vs 32
    // scalar adds), so rounding accumulates differently.
    for (i, (got, want)) in out_avx2.iter().zip(reference.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-2,
            "AVX2 diverges at {}: got {} expected {}",
            i,
            got,
            want
        );
    }
}

#[test]
fn tile_4x8_matmul_16x16x16_matches_reference() {
    // M=16 (mult of 4), N=16 (mult of 8) → 4x8 register tile path selected.
    let src = r#"
        fn matmul(
            a: tensor<f32, [16, 16]>,
            b: tensor<f32, [16, 16]>,
        ) -> tensor<f32, [16, 16]> {
            return a @ b;
        }
    "#;
    let a: Vec<f32> = (0..256).map(|i| ((i % 11) as f32) * 0.07 - 0.3).collect();
    let b: Vec<f32> = (0..256).map(|i| ((i % 19) as f32) * 0.02 + 0.5).collect();
    let mut out = vec![0.0f32; 256];

    compile_and_run(&X86_64::host(), src, &a, &b, &mut out);

    let reference = naive_matmul(&a, &b, 16, 16, 16);
    for (i, (got, want)) in out.iter().zip(reference.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "4x8 tile diverges at {}: got {} expected {}",
            i,
            got,
            want
        );
    }
}

#[test]
fn tile_4x8_matmul_64x64x64_matches_reference() {
    let src = r#"
        fn matmul(
            a: tensor<f32, [64, 64]>,
            b: tensor<f32, [64, 64]>,
        ) -> tensor<f32, [64, 64]> {
            return a @ b;
        }
    "#;
    let a: Vec<f32> = (0..4096)
        .map(|i| ((i % 23) as f32) * 0.001 - 0.05)
        .collect();
    let b: Vec<f32> = (0..4096).map(|i| ((i % 29) as f32) * 0.002 + 0.1).collect();
    let mut out = vec![0.0f32; 4096];

    compile_and_run(&X86_64::host(), src, &a, &b, &mut out);

    let reference = naive_matmul(&a, &b, 64, 64, 64);
    for (i, (got, want)) in out.iter().zip(reference.iter()).enumerate() {
        // K=64 → 64 multiply-adds per output. Looser tolerance.
        assert!(
            (got - want).abs() < 1e-2,
            "4x8 tile 64x64 diverges at {}: got {} expected {}",
            i,
            got,
            want
        );
    }
}

#[test]
fn scalar_matmul_rectangular_4x5x3() {
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
    compile_and_run(&X86_64::host(), src, &a, &b, &mut out);

    let reference = naive_matmul(&a, &b, 4, 5, 3);
    for (got, want) in out.iter().zip(reference.iter()) {
        assert!((got - want).abs() < 1e-4);
    }
}
