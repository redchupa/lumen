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
    q8_entries: HashMap<(u32, u32, u32), ExecRegion>,
    q8q8_entries: HashMap<(u32, u32, u32), ExecRegion>,
}

/// Function signature emitted by `lumen_codegen::x86_64` for matmul.
pub type MatmulFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32);

/// Function signature for the Q8_0 weight × F32 activation fused matmul kernel.
/// First argument is the raw Q8_0 block buffer (`*const u8`).
pub type Q8MatmulFn = unsafe extern "C" fn(*const u8, *const f32, *mut f32);

/// Function signature for the Q8_0 weight × Q8_0 activation fused matmul kernel.
/// Both inputs are raw Q8_0 block buffers (`*const u8`).
pub type Q8Q8MatmulFn = unsafe extern "C" fn(*const u8, *const u8, *mut f32);

impl MatmulJitCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            q8_entries: HashMap::new(),
            q8q8_entries: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of Q8 fused kernels currently cached.
    pub fn q8_len(&self) -> usize {
        self.q8_entries.len()
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

    /// Returns a cached fused Q8_0×F32 kernel of shape
    /// `(weights[M, K]: q8_0) @ (activations[K, N]: f32) → (out[M, N]: f32)`.
    /// Constraints: `K % 32 == 0`. `N == 1` (decode) or `N % 8 == 0` (prefill).
    pub fn get_or_compile_q8(&mut self, m: u32, k: u32, n: u32) -> Result<Q8MatmulFn, JitError> {
        let key = (m, k, n);
        let region = match self.q8_entries.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let compiled = compile_q8_matmul(m, k, n)?;
                v.insert(compiled)
            }
        };
        // SAFETY: the region was produced by `lumen_codegen::x86_64` via the
        // Q8 fused IR pattern (Param Q8 + Param F32 + Dequantize + MatMul +
        // Return), which emits a `Q8MatmulFn`-signature function.
        Ok(unsafe { region.as_fn::<Q8MatmulFn>() })
    }

    /// Number of Q8×Q8 fused kernels currently cached.
    pub fn q8q8_len(&self) -> usize {
        self.q8q8_entries.len()
    }

    /// Returns a cached fused Q8_0×Q8_0 kernel of shape
    /// `(weights[M, K]: q8_0) @ (activations[K, 1]: q8_0) → (out[M, 1]: f32)`.
    /// Constraints: `K % 32 == 0`. Only `N == 1` (decode) is supported.
    /// Phase 7.M.
    pub fn get_or_compile_q8q8(&mut self, m: u32, k: u32) -> Result<Q8Q8MatmulFn, JitError> {
        let key = (m, k, 1);
        let region = match self.q8q8_entries.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let compiled = compile_q8q8_matmul(m, k)?;
                v.insert(compiled)
            }
        };
        // SAFETY: the region was produced by `lumen_codegen::x86_64` via the
        // Q8×Q8 fused IR pattern (Param Q8 + Param Q8 + 2× Dequantize +
        // MatMul + Return), which emits a `Q8Q8MatmulFn`-signature function.
        Ok(unsafe { region.as_fn::<Q8Q8MatmulFn>() })
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

/// Build the IR for a fused Q8_0 weight × F32 activation matmul and
/// JIT-compile it. IR pattern (recognized by `emit_function_quant_matmul_q8`):
/// ```text
///   v0 = Param(0) : tensor<q8_0, [M, K]>     # weights
///   v1 = Param(1) : tensor<f32,  [K, N]>     # activations
///   v2 = Dequantize v0 : tensor<f32, [M, K]>
///   v3 = MatMul v2, v1 : tensor<f32, [M, N]>
///   return v3
/// ```
fn compile_q8_matmul(m: u32, k: u32, n: u32) -> Result<ExecRegion, JitError> {
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
    let c_ty = TensorType {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Static(m), Dim::Static(n)]),
    };

    let mut f = Function::new(
        "quant_matmul",
        vec![w_ty.clone(), a_ty.clone()],
        c_ty.clone(),
    );
    let w = f.param_values[0];
    let a = f.param_values[1];
    let dq = f.push(Value {
        op: Op::Dequantize { x: w },
        ty: dq_ty,
    });
    let prod = f.push(Value {
        op: Op::MatMul { lhs: dq, rhs: a },
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

/// Build the IR for a Q8×Q8 fused matmul (N=1 decode) and JIT-compile it.
/// IR pattern (recognized by `emit_function_quant_matmul_q8`):
/// ```text
///   v0 = Param(0) : tensor<q8_0, [M, K]>     # weights
///   v1 = Param(1) : tensor<q8_0, [K, 1]>     # activations (pre-quantized)
///   v2 = Dequantize v0 : tensor<f32, [M, K]>
///   v3 = Dequantize v1 : tensor<f32, [K, 1]>
///   v4 = MatMul v2, v3 : tensor<f32, [M, 1]>
///   return v4
/// ```
fn compile_q8q8_matmul(m: u32, k: u32) -> Result<ExecRegion, JitError> {
    let n = 1u32;
    let w_ty = TensorType {
        dtype: DType::Q8_0,
        shape: Shape(vec![Dim::Static(m), Dim::Static(k)]),
    };
    let a_ty = TensorType {
        dtype: DType::Q8_0,
        shape: Shape(vec![Dim::Static(k), Dim::Static(n)]),
    };
    let dq_w_ty = TensorType {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Static(m), Dim::Static(k)]),
    };
    let dq_a_ty = TensorType {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Static(k), Dim::Static(n)]),
    };
    let c_ty = TensorType {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Static(m), Dim::Static(n)]),
    };

    let mut f = Function::new(
        "q8q8_matmul",
        vec![w_ty.clone(), a_ty.clone()],
        c_ty.clone(),
    );
    let w = f.param_values[0];
    let a = f.param_values[1];
    let dq_w = f.push(Value {
        op: Op::Dequantize { x: w },
        ty: dq_w_ty,
    });
    let dq_a = f.push(Value {
        op: Op::Dequantize { x: a },
        ty: dq_a_ty,
    });
    let prod = f.push(Value {
        op: Op::MatMul {
            lhs: dq_w,
            rhs: dq_a,
        },
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

    /// Phase 7.C: decode-shape (M=1, N % 32 == 0) hits the new 1xN 4-acc AVX2
    /// path. Validates the new dispatcher branch and the emitted code against
    /// the same naive baseline.
    #[test]
    fn decode_shape_1xn_4acc_path_matches_naive() {
        let mut cache = MatmulJitCache::new();
        // Mix of K values, all N divisible by 32 — covers lm_head (large N),
        // FFN-style (medium N), and projection (smaller N).
        for &(m, k, n) in &[
            (1u32, 16, 32),
            (1, 32, 64),
            (1, 64, 96),
            (1, 128, 256),
            (1, 7, 32),  // odd K
            (1, 13, 64), // odd K, larger N
        ] {
            let f = cache
                .get_or_compile(m, k, n)
                .unwrap_or_else(|e| panic!("compile ({},{},{}): {}", m, k, n, e));

            let a: Vec<f32> = (0..(m * k) as usize)
                .map(|i| ((i % 5) as f32) * 0.21 - 0.3)
                .collect();
            let b: Vec<f32> = (0..(k * n) as usize)
                .map(|i| ((i % 13) as f32) * 0.07 + 0.05)
                .collect();
            let mut out = vec![0.0f32; (m * n) as usize];
            // SAFETY: kernel compiled for these exact (M,K,N).
            unsafe { f(a.as_ptr(), b.as_ptr(), out.as_mut_ptr()) };

            let want = naive(&a, &b, m as usize, k as usize, n as usize);
            let tol = 1e-3 * (k as f32).max(1.0);
            for (g, w) in out.iter().zip(want.iter()) {
                assert!(
                    (g - w).abs() < tol,
                    "shape ({},{},{}): {} vs {} (tol {})",
                    m,
                    k,
                    n,
                    g,
                    w,
                    tol
                );
            }
        }
    }

    /// Phase 7.D: fused Q8 × F32 matmul for the decode shape (N=1). Verifies
    /// the new K-direction kernel produces the same result as
    /// dequant-then-naive-matmul on a range of M and K.
    #[test]
    fn q8_n1_kernel_matches_dequant_then_naive() {
        use lumen_runtime::quant::{dequantize_q8_0, quantize_q8_0, BlockQ8_0};

        let mut cache = MatmulJitCache::new();
        // Shapes representative of decode-time projections / lm_head:
        //   - M=8,  K=32  : smallest case, one Q8 block per row.
        //   - M=16, K=64  : two blocks per row.
        //   - M=32, K=128 : larger row count.
        //   - M=7,  K=256 : odd M, big K to surface accumulator issues.
        for &(m, k) in &[
            (8u32, 32),
            (16, 64),
            (32, 128),
            (7, 256),
            // Phase 7.G additions — exercise the multi-accumulator kernel at
            // shapes closer to Qwen2.5-0.5B's actual decode matmuls so that
            // FMA dependency-chain bugs surface in the unit test, not just
            // in e2e.
            (896, 896),  // QKV proj-ish, lm_head row stride
            (128, 4864), // FFN-ish (small d_out, large d_in)
            (4864, 896), // FFN gate/up (large d_out, small d_in)
        ] {
            let n: u32 = 1;

            // Random-ish but deterministic F32 weight matrix.
            let wts_f32: Vec<f32> = (0..(m * k) as usize)
                .map(|i| (((i % 17) as f32) - 8.0) * 0.12)
                .collect();
            // Quantize to Q8_0.
            let mut wts_q8 = vec![
                BlockQ8_0 {
                    d: 0,
                    qs: [0i8; 32]
                };
                (m * k / 32) as usize
            ];
            quantize_q8_0(&wts_f32, &mut wts_q8);
            // Reference: dequant back and run naive matmul on the dequantized values.
            // (The JIT kernel fuses these two steps; agreement here means the
            // fusion was correct, not that quantize is lossless.)
            let mut wts_dq = vec![0.0f32; (m * k) as usize];
            dequantize_q8_0(&wts_q8, &mut wts_dq);

            let acts: Vec<f32> = (0..(k * n) as usize)
                .map(|i| (((i % 11) as f32) - 5.0) * 0.07)
                .collect();
            let want = naive(&wts_dq, &acts, m as usize, k as usize, n as usize);

            let f = cache
                .get_or_compile_q8(m, k, n)
                .unwrap_or_else(|e| panic!("compile q8 ({},{},{}): {}", m, k, n, e));
            let mut out = vec![0.0f32; (m * n) as usize];
            // SAFETY: kernel was just compiled for this exact (M,K,N) Q8 shape.
            unsafe {
                f(
                    wts_q8.as_ptr() as *const u8,
                    acts.as_ptr(),
                    out.as_mut_ptr(),
                )
            };

            // Tolerance is generous: we accumulate K FMAs of fp32 values, and
            // the rms of fp32 rounding scales with sqrt(K).
            let tol = 1e-3 * (k as f32).sqrt();
            for (idx, (g, w)) in out.iter().zip(want.iter()).enumerate() {
                assert!(
                    (g - w).abs() < tol,
                    "shape ({},{},{}) row {}: {} vs {} (tol {})",
                    m,
                    k,
                    n,
                    idx,
                    g,
                    w,
                    tol
                );
            }
        }
        // Cache should hold one Q8 entry per distinct shape.
        assert_eq!(cache.q8_len(), 7);
    }

    /// Phase 7.M: Q8 weights × Q8 activations fused matmul kernel (N=1).
    /// Validates that the int-dot path agrees with dequant-both-then-naive
    /// across decode-relevant shapes.
    #[test]
    fn q8q8_n1_kernel_matches_dequant_both_then_naive() {
        use lumen_runtime::quant::{dequantize_q8_0, quantize_q8_0, BlockQ8_0};

        let mut cache = MatmulJitCache::new();
        for &(m, k) in &[
            (8u32, 32),
            (16, 64),
            (32, 128),
            (7, 256),    // odd M
            (896, 896),  // Qwen wq/wo shape
            (4864, 896), // Qwen FFN gate/up
            (128, 4864), // Qwen FFN down (small M, large K)
        ] {
            // Random-ish but deterministic source values, kept in a range
            // where Q8 quantization round-trips reasonably (max ≲ 1).
            let wts_f32: Vec<f32> = (0..(m * k) as usize)
                .map(|i| (((i % 17) as f32) - 8.0) * 0.07)
                .collect();
            let acts_f32: Vec<f32> = (0..k as usize)
                .map(|i| (((i % 13) as f32) - 6.0) * 0.04)
                .collect();

            // Quantize both, then dequantize for the reference matmul so the
            // test compares apples-to-apples (Q8-quantized inputs).
            let mut wts_q8 = vec![
                BlockQ8_0 {
                    d: 0,
                    qs: [0i8; 32]
                };
                (m * k / 32) as usize
            ];
            quantize_q8_0(&wts_f32, &mut wts_q8);
            let mut acts_q8 = vec![
                BlockQ8_0 {
                    d: 0,
                    qs: [0i8; 32]
                };
                (k / 32) as usize
            ];
            quantize_q8_0(&acts_f32, &mut acts_q8);

            let mut wts_dq = vec![0.0f32; (m * k) as usize];
            dequantize_q8_0(&wts_q8, &mut wts_dq);
            let mut acts_dq = vec![0.0f32; k as usize];
            dequantize_q8_0(&acts_q8, &mut acts_dq);

            let want = naive(&wts_dq, &acts_dq, m as usize, k as usize, 1);

            let f = cache
                .get_or_compile_q8q8(m, k)
                .unwrap_or_else(|e| panic!("compile q8q8 ({},{}): {}", m, k, e));
            let mut out = vec![0.0f32; m as usize];
            // SAFETY: kernel was just compiled for this exact (m, k, 1) Q8×Q8 shape.
            unsafe {
                f(
                    wts_q8.as_ptr() as *const u8,
                    acts_q8.as_ptr() as *const u8,
                    out.as_mut_ptr(),
                )
            };

            let tol = 1e-3 * (k as f32).sqrt();
            for (idx, (g, w)) in out.iter().zip(want.iter()).enumerate() {
                assert!(
                    (g - w).abs() < tol,
                    "shape ({},{},1) row {}: {} vs {} (tol {})",
                    m,
                    k,
                    idx,
                    g,
                    w,
                    tol
                );
            }
        }
        assert_eq!(cache.q8q8_len(), 7);
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
