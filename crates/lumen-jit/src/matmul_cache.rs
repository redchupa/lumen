//! Shape-indexed JIT cache for fp32 matmul kernels.
//!
//! Phase 6.G.1 scope: build / cache / call kernels for the
//! `A [M, K] @ B [K, N] -> C [M, N]` shape. `B` must be stored row-major as
//! `[K, N]` (so `b[k * N + j]` is the right element). Backend picks scalar /
//! 1×8 AVX2 / 4×8 tile automatically based on M and N divisibility, same as
//! `lumen_codegen::x86_64::X86_64::host()`.
//!
//! Usage:
//! ```ignore
//! let mut cache = MatmulJitCache::new();
//! let f = cache.get_or_compile(M, K, N);
//! unsafe { f(a.as_ptr(), b.as_ptr(), out.as_mut_ptr()); }
//! ```

use std::collections::HashMap;

use lumen_codegen::backend::{Backend, CodegenOpts};
use lumen_codegen::x86_64::X86_64;
use lumen_ir::ty::{DType, Dim, Shape, TensorType};
use lumen_ir::{Function, IrModule, Op, Value};

use crate::exec::{ExecError, ExecRegion};

/// `(M, K, N)` keyed JIT cache. Each entry owns its W^X page; dropping the
/// cache frees them all.
pub struct MatmulJitCache {
    entries: HashMap<(u32, u32, u32), ExecRegion>,
}

/// Function signature emitted by `lumen_codegen::x86_64` for matmul.
pub type MatmulFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32);

impl MatmulJitCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns a cached kernel for `(M, K, N)`, compiling on first request.
    pub fn get_or_compile(&mut self, m: u32, k: u32, n: u32) -> Result<MatmulFn, JitError> {
        let key = (m, k, n);
        let region = match self.entries.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let compiled = compile_matmul(m, k, n)?;
                v.insert(compiled)
            }
        };
        // SAFETY: the region was produced by `lumen_codegen::x86_64`, which
        // emits a function with the `MatmulFn` signature. We checked the
        // shape matches what the kernel was compiled for.
        Ok(unsafe { region.as_fn::<MatmulFn>() })
    }
}

impl Default for MatmulJitCache {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(thiserror::Error, Debug)]
pub enum JitError {
    #[error("codegen failed: {0}")]
    Codegen(String),
    #[error("exec region setup failed: {0}")]
    Exec(#[from] ExecError),
}

/// Build the IR for an `A @ B` matmul of the given shapes and JIT-compile it.
fn compile_matmul(m: u32, k: u32, n: u32) -> Result<ExecRegion, JitError> {
    let a_ty = TensorType {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Static(m), Dim::Static(k)]),
    };
    let b_ty = TensorType {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Static(k), Dim::Static(n)]),
    };
    let c_ty = TensorType {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Static(m), Dim::Static(n)]),
    };

    let mut f = Function::new("matmul", vec![a_ty.clone(), b_ty.clone()], c_ty.clone());
    let a = f.param_values[0];
    let b = f.param_values[1];
    let prod = f.push(Value {
        op: Op::MatMul { lhs: a, rhs: b },
        ty: c_ty,
    });
    let placeholder = TensorType {
        dtype: DType::F32,
        shape: Shape(vec![]),
    };
    f.push(Value {
        op: Op::Return { value: prod },
        ty: placeholder,
    });

    let ir = IrModule { functions: vec![f] };

    let backend = X86_64::host();
    let mc = backend
        .lower(&ir, &CodegenOpts::default())
        .map_err(|e| JitError::Codegen(format!("{:?}", e)))?;
    let region = ExecRegion::from_machine_code(&mc)?;
    Ok(region)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
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
    fn cache_compiles_and_calls_matmul() {
        let mut cache = MatmulJitCache::new();
        let f = cache.get_or_compile(4, 4, 8).expect("compile");
        assert_eq!(cache.len(), 1);

        let a: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5).collect();
        let b: Vec<f32> = (0..32).map(|i| (i as f32) * 0.25 + 1.0).collect();
        let mut out = vec![0.0f32; 32];
        // SAFETY: function was just compiled for these exact dims.
        unsafe { f(a.as_ptr(), b.as_ptr(), out.as_mut_ptr()) };

        let want = naive(&a, &b, 4, 4, 8);
        for (g, w) in out.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-4, "{} vs {}", g, w);
        }
    }

    #[test]
    fn cache_returns_same_region_on_second_call() {
        let mut cache = MatmulJitCache::new();
        let _ = cache.get_or_compile(8, 8, 8).expect("compile 1");
        let _ = cache.get_or_compile(8, 8, 8).expect("compile 2");
        assert_eq!(
            cache.len(),
            1,
            "second compile of same shape should hit cache"
        );

        let _ = cache.get_or_compile(8, 8, 16).expect("compile other shape");
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn multiple_shapes_round_trip_correctly() {
        let mut cache = MatmulJitCache::new();
        for &(m, k, n) in &[(4u32, 4, 8), (8, 8, 8), (1, 8, 8), (16, 32, 16)] {
            let f = cache
                .get_or_compile(m, k, n)
                .unwrap_or_else(|e| panic!("shape ({},{},{}): {}", m, k, n, e));

            let a: Vec<f32> = (0..(m * k) as usize)
                .map(|i| ((i % 7) as f32) * 0.13 - 0.5)
                .collect();
            let b: Vec<f32> = (0..(k * n) as usize)
                .map(|i| ((i % 11) as f32) * 0.07 + 0.1)
                .collect();
            let mut out = vec![0.0f32; (m * n) as usize];
            unsafe { f(a.as_ptr(), b.as_ptr(), out.as_mut_ptr()) };

            let want = naive(&a, &b, m as usize, k as usize, n as usize);
            // Tolerance grows with K (more FMAs accumulate rounding).
            let tol = 1e-3 * (k as f32).max(1.0);
            for (g, w) in out.iter().zip(want.iter()) {
                assert!(
                    (g - w).abs() < tol,
                    "shape ({},{},{}): {} vs {}",
                    m,
                    k,
                    n,
                    g,
                    w
                );
            }
        }
        assert_eq!(cache.len(), 4);
    }
}
