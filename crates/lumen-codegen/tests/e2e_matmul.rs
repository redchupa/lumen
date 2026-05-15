//! End-to-end correctness: Lumen source → C → cc → dylib → execute → compare
//! against a Rust naive reference.
//!
//! Skips automatically if no C compiler is available on PATH. The build sets a
//! cfg flag `lumen_have_cc` later (Phase 3 CI) so we can fail rather than skip
//! in environments where a compiler is required.

use std::path::{Path, PathBuf};
use std::process::Command;

use lumen_codegen::emit_c;
use lumen_dsl::Parser;
use lumen_ir::lower;

fn find_cc() -> Option<&'static str> {
    ["cc", "gcc", "clang"]
        .into_iter()
        .find(|cand| Command::new(cand).arg("--version").output().is_ok())
}

#[cfg(target_os = "windows")]
const DYLIB_EXT: &str = "dll";
#[cfg(target_os = "macos")]
const DYLIB_EXT: &str = "dylib";
#[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
const DYLIB_EXT: &str = "so";

/// Build the C source as a shared library, returning its path.
fn build_shared(cc: &str, c_path: &Path, out_dir: &Path, name: &str) -> Option<PathBuf> {
    let lib_path = out_dir.join(format!("lib{}.{}", name, DYLIB_EXT));
    let status = Command::new(cc)
        .arg("-shared")
        .arg("-O2")
        .arg("-o")
        .arg(&lib_path)
        .arg(c_path)
        .arg("-fPIC")
        .status()
        .ok()?;
    if !status.success() {
        eprintln!("cc returned {:?}", status);
        return None;
    }
    Some(lib_path)
}

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

#[test]
fn matmul_8x8x8_matches_reference() {
    let Some(cc) = find_cc() else {
        eprintln!("skipping e2e_matmul: no C compiler on PATH");
        return;
    };

    let src = r#"
        fn matmul(
            a: tensor<f32, [8, 8]>,
            b: tensor<f32, [8, 8]>,
        ) -> tensor<f32, [8, 8]> {
            return a @ b;
        }
    "#;
    let module = Parser::parse(src).unwrap();
    let ir = lower(&module).unwrap();
    let c_src = emit_c(&ir).unwrap();

    let tmp = tempfile::tempdir().expect("tempdir");
    let c_path = tmp.path().join("kernel.c");
    std::fs::write(&c_path, &c_src).unwrap();
    let lib_path = build_shared(cc, &c_path, tmp.path(), "lumen_kernel").expect("build shared lib");

    // SAFETY: we just built this library; the symbol matches `void f(const float*, const float*, float*)`.
    let result = unsafe {
        let lib = libloading::Library::new(&lib_path).expect("load lib");
        type Fn3 = unsafe extern "C" fn(*const f32, *const f32, *mut f32);
        let func: libloading::Symbol<Fn3> = lib.get(b"lumen_matmul").expect("symbol");

        let a: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05 + 1.0).collect();
        let mut out = vec![0.0f32; 64];
        func(a.as_ptr(), b.as_ptr(), out.as_mut_ptr());

        let reference = naive_matmul(&a, &b, 8, 8, 8);
        for (i, (got, want)) in out.iter().zip(reference.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "mismatch at index {}: got {} expected {}",
                i,
                got,
                want
            );
        }
        out
    };

    assert_eq!(result.len(), 64);
}

#[test]
fn matmul_rectangular_4x5x3_matches_reference() {
    let Some(cc) = find_cc() else {
        eprintln!("skipping e2e_matmul: no C compiler on PATH");
        return;
    };

    let src = r#"
        fn matmul(
            a: tensor<f32, [4, 5]>,
            b: tensor<f32, [5, 3]>,
        ) -> tensor<f32, [4, 3]> {
            return a @ b;
        }
    "#;
    let module = Parser::parse(src).unwrap();
    let ir = lower(&module).unwrap();
    let c_src = emit_c(&ir).unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let c_path = tmp.path().join("kernel.c");
    std::fs::write(&c_path, &c_src).unwrap();
    let lib_path =
        build_shared(cc, &c_path, tmp.path(), "lumen_kernel2").expect("build shared lib");

    unsafe {
        let lib = libloading::Library::new(&lib_path).unwrap();
        type Fn3 = unsafe extern "C" fn(*const f32, *const f32, *mut f32);
        let func: libloading::Symbol<Fn3> = lib.get(b"lumen_matmul").unwrap();

        let a: Vec<f32> = (0..20).map(|i| (i as f32) - 10.0).collect(); // [-10..9]
        let b: Vec<f32> = (0..15).map(|i| 1.0 / ((i + 1) as f32)).collect();
        let mut out = vec![0.0f32; 12];
        func(a.as_ptr(), b.as_ptr(), out.as_mut_ptr());

        let reference = naive_matmul(&a, &b, 4, 5, 3);
        for (got, want) in out.iter().zip(reference.iter()) {
            assert!((got - want).abs() < 1e-4, "got {} expected {}", got, want);
        }
    }
}
