//! Phase 8.D.1: validate that the Q8 weight × F32 activation JIT kernel is
//! correct at N > 1 (prefill shape). The kernel was originally built for the
//! N = 1 decode path; the matmul_cache docstring already advertises N % 8 == 0
//! as supported but no test actually exercises the prefill side end-to-end.
//!
//! This sits in lumen-model/tests because lumen-codegen's own unit tests
//! pull in criterion via dev-deps, and criterion's transitive deps fail to
//! build with `dlltool not found` on this Windows host. lumen-model's deps
//! are clean.

use lumen_jit::MatmulJitCache;
use lumen_runtime::quant::{dequantize_q8_0, quantize_q8_0, BlockQ8_0};

/// Reference: textbook A[M,K] @ B[K,N] -> C[M,N], row-major throughout.
fn naive_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for l in 0..k {
                acc += a[i * k + l] * b[l * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// Exercise the JIT path at the shapes prefill will actually request from
/// `forward_layer_prefill_jit` once it's written. Q8 weight per row (K must
/// be a multiple of 32); F32 activation [K,N] in row-major; F32 output [M,N].
fn q8_prefill_one_shape(m: usize, k: usize, n: usize) {
    assert!(k % 32 == 0, "K must be a multiple of the Q8 block size");

    // Deterministic but non-trivial weights — values mix sign and magnitude
    // so a missed sub-accumulator or wrong row stride blows up the check.
    let weights_f32: Vec<f32> = (0..m * k)
        .map(|i| ((i % 17) as f32) * 0.13 - ((i % 7) as f32) * 0.21)
        .collect();
    let mut weights_q8 = vec![BlockQ8_0 { d: 0, qs: [0; 32] }; m * k / 32];
    quantize_q8_0(&weights_f32, &mut weights_q8);

    // Round-trip dequant for the reference path — that way the comparison
    // is JIT-vs-naive on the *same* quantized weights, not JIT-vs-original.
    // Otherwise we'd be measuring Q8 quantization error, not kernel error.
    let mut weights_dq = vec![0.0f32; m * k];
    dequantize_q8_0(&weights_q8, &mut weights_dq);

    let activations: Vec<f32> = (0..k * n)
        .map(|i| ((i % 11) as f32) * 0.07 + 0.05)
        .collect();

    let mut cache = MatmulJitCache::new();
    let f = cache
        .get_or_compile_q8(m as u32, k as u32, n as u32)
        .unwrap_or_else(|e| panic!("compile q8 ({},{},{}): {}", m, k, n, e));

    let mut out_jit = vec![0.0f32; m * n];
    // SAFETY: kernel compiled for exactly (m, k, n); buffers sized accordingly.
    unsafe {
        f(
            weights_q8.as_ptr() as *const u8,
            activations.as_ptr(),
            out_jit.as_mut_ptr(),
        );
    }

    let out_ref = naive_matmul(&weights_dq, &activations, m, k, n);

    // Tolerance scales with K because each output is a K-long fp32 sum;
    // worst-case relative rounding accumulates ~K*eps.
    let tol = 1e-3 * (k as f32).max(1.0);
    for (idx, (g, w)) in out_jit.iter().zip(out_ref.iter()).enumerate() {
        assert!(
            (g - w).abs() < tol,
            "shape ({},{},{}) idx {}: jit={} ref={} diff={} tol={}",
            m,
            k,
            n,
            idx,
            g,
            w,
            (g - w).abs(),
            tol
        );
    }
}

/// Smallest viable prefill shape: M=1 row, K=one Q8 block, N=8 lanes.
/// If this fails the whole path is dead.
#[test]
fn q8_prefill_m1_k32_n8_matches_naive() {
    q8_prefill_one_shape(1, 32, 8);
}

/// Realistic Qwen-style decode-but-batched: M is small (one row per output
/// channel of the projection), K=896 (Qwen2.5-0.5B hidden), N varies to
/// cover small / medium / large prefill batches.
#[test]
fn q8_prefill_qwen_like_shapes() {
    for &(m, k, n) in &[
        // K=896 is Qwen2.5-0.5B hidden. K_blocks = 28.
        (4, 896, 8),
        (8, 896, 16),
        (8, 896, 32),
        // K=4864 is the FFN inner dim (down_matmul's K). K_blocks = 152.
        (4, 4864, 8),
        // Lopsided shapes — odd row count, larger N.
        (7, 896, 16),
        (16, 896, 64),
    ] {
        q8_prefill_one_shape(m, k, n);
    }
}

/// N=1 sanity: the prefill-shape kernel must still produce the same answer
/// as the decode-path kernel when N==1. Regression guard for the case where
/// the new code path forks decode vs prefill differently.
#[test]
fn q8_n1_decode_still_matches_naive() {
    q8_prefill_one_shape(8, 32, 1);
    q8_prefill_one_shape(8, 128, 1);
    q8_prefill_one_shape(32, 896, 1);
}
