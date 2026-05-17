//! Pure-Rust reference implementations of the per-token / per-layer ops a
//! Llama-family transformer needs in its forward pass.
//!
//! These are the *correctness ground truth* against which Phase 6.C.2+ native
//! kernels will be diff-tested. They are deliberately naive — no SIMD, no
//! tiling, no rayon — so the math is easy to read and audit.
//!
//! Naming follows ggml / llama.cpp conventions where it matters.

/// Root-mean-square layer normalization.
///
/// `y[i] = (x[i] / sqrt(mean(x²) + eps)) * weight[i]`
///
/// For a 2-D activation `x[B, H]`, this is applied independently to each row
/// (typical case: B = sequence length, H = hidden dim). `weight` is shared
/// across rows.
///
/// Llama / Qwen2 / EXAONE all use this exact formulation. Note the order:
/// `(x * inv_rms) * weight`, **not** `(x * weight) * inv_rms` — the math is
/// commutative but the integer-FMA accumulation order differs, and matching
/// llama.cpp's bit-for-bit output requires this order.
pub fn rms_norm(x: &[f32], weight: &[f32], out: &mut [f32], hidden: usize, eps: f32) {
    assert_eq!(x.len() % hidden, 0, "x.len must be a multiple of hidden");
    assert_eq!(out.len(), x.len());
    assert_eq!(weight.len(), hidden);

    for row in x.chunks_exact(hidden).zip(out.chunks_exact_mut(hidden)) {
        let (xr, yr) = row;
        let sum_sq: f32 = xr.iter().map(|&v| v * v).sum();
        let inv_rms = 1.0f32 / (sum_sq / hidden as f32 + eps).sqrt();
        for i in 0..hidden {
            yr[i] = xr[i] * inv_rms * weight[i];
        }
    }
}

/// SiLU activation, in-place: `x[i] = x[i] * sigmoid(x[i])`.
///
/// Also called "Swish". Used in Llama-family FFN gates. Numerically stable
/// formulation: `sigmoid(v) = 1 / (1 + exp(-v))`.
///
/// Phase 7.H: dispatches to an AVX2+FMA implementation when available. The
/// scalar path remains as the reference (and the fallback on non-x86_64).
pub fn silu_in_place(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: feature detection just confirmed avx2 + fma are available.
            unsafe {
                silu_in_place_avx2_fma(x);
            }
            return;
        }
    }
    silu_in_place_scalar(x);
}

#[inline]
fn silu_in_place_scalar(x: &mut [f32]) {
    for v in x.iter_mut() {
        let s = 1.0f32 / (1.0 + (-*v).exp());
        *v *= s;
    }
}

/// Element-wise multiply, in-place: `lhs[i] *= rhs[i]`. Used in Llama FFN
/// after the SiLU gate (`down(silu(gate) * up)` pattern).
pub fn mul_in_place(lhs: &mut [f32], rhs: &[f32]) {
    assert_eq!(lhs.len(), rhs.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: feature detection just confirmed avx2 is available.
            unsafe {
                mul_in_place_avx2(lhs, rhs);
            }
            return;
        }
    }
    for (l, &r) in lhs.iter_mut().zip(rhs.iter()) {
        *l *= r;
    }
}

// ============================================================================
// x86_64 AVX2 implementations of SiLU and elementwise multiply.
//
// Phase 7.H. SiLU+mul were ~14% of decode time per the Phase 7.E.0 profile;
// vectorizing them with std::arch (no extra deps) gives that share back.
// The vector `expf` here is a degree-5 polynomial in the reduced range plus
// integer exponent reconstruction — same idea as cephes/libm but inlined into
// 8-wide ymm arithmetic. Accurate to ~1e-7 relative on |x| ≤ 87, which is
// more than enough for argmax-preserving sigmoid output.
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn silu_in_place_avx2_fma(x: &mut [f32]) {
    use std::arch::x86_64::*;
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();
    let n = x.len();
    let mut i = 0;
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        let neg_x = _mm256_sub_ps(zero, xv);
        let exp_neg_x = exp_ps_avx2_fma(neg_x);
        let denom = _mm256_add_ps(one, exp_neg_x);
        let result = _mm256_div_ps(xv, denom);
        _mm256_storeu_ps(x.as_mut_ptr().add(i), result);
        i += 8;
    }
    // Scalar tail.
    while i < n {
        let v = x[i];
        x[i] = v / (1.0 + (-v).exp());
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mul_in_place_avx2(lhs: &mut [f32], rhs: &[f32]) {
    use std::arch::x86_64::*;
    let n = lhs.len();
    let mut i = 0;
    while i + 8 <= n {
        let lv = _mm256_loadu_ps(lhs.as_ptr().add(i));
        let rv = _mm256_loadu_ps(rhs.as_ptr().add(i));
        let pv = _mm256_mul_ps(lv, rv);
        _mm256_storeu_ps(lhs.as_mut_ptr().add(i), pv);
        i += 8;
    }
    while i < n {
        lhs[i] *= rhs[i];
        i += 1;
    }
}

/// Vectorized `exp(x)` for ymm, valid on `|x| ≤ 87` (clamped internally for
/// safe 2^n reconstruction; outside that range exp saturates to 0 / +inf
/// numerically). Algorithm: range-reduce `x = n*ln2 + r` with `n = round(x*log2e)`,
/// approximate `exp(r)` with a degree-5 polynomial, multiply by `2^n` built
/// from integer exponent bits.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_ps_avx2_fma(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    // Clamp so that the integer exponent fits in the [-126, 127] biased range.
    let lo = _mm256_set1_ps(-87.0);
    let hi = _mm256_set1_ps(87.0);
    let xc = _mm256_max_ps(lo, _mm256_min_ps(hi, x));

    let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);

    // n = round(x * log2(e))
    let xlog2e = _mm256_mul_ps(xc, log2e);
    // _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC == 0
    let n = _mm256_round_ps::<0>(xlog2e);
    // r = x - n*ln2  (use FMA: r = -n*ln2 + x)
    let r = _mm256_fnmadd_ps(n, ln2, xc);

    // exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120, Horner from high degree
    let c1_120 = _mm256_set1_ps(1.0 / 120.0);
    let c1_24 = _mm256_set1_ps(1.0 / 24.0);
    let c1_6 = _mm256_set1_ps(1.0 / 6.0);
    let c1_2 = _mm256_set1_ps(0.5);
    let c_one = _mm256_set1_ps(1.0);

    let mut p = _mm256_fmadd_ps(c1_120, r, c1_24);
    p = _mm256_fmadd_ps(p, r, c1_6);
    p = _mm256_fmadd_ps(p, r, c1_2);
    p = _mm256_fmadd_ps(p, r, c_one);
    p = _mm256_fmadd_ps(p, r, c_one);

    // 2^n: convert n to int, bias by 127, shift into exponent bits.
    let n_int = _mm256_cvtps_epi32(n);
    let bias = _mm256_set1_epi32(127);
    let exp_bits = _mm256_slli_epi32::<23>(_mm256_add_epi32(n_int, bias));
    let pow2n = _mm256_castsi256_ps(exp_bits);

    _mm256_mul_ps(p, pow2n)
}

/// Llama-style Rotary Position Embedding (RoPE), applied in-place to a single
/// query or key tensor laid out as `[seq, n_heads, head_dim]` row-major.
///
/// Pairing scheme: **split half** (a.k.a. "GPT-NeoX-style") — element `i` is
/// paired with element `i + head_dim/2` within each head. This matches Llama2+
/// and Qwen2's GGUF layout.
///
/// Frequencies follow the standard formula:
///   θ_i = pos * base^(-2i / head_dim)
///
/// `positions[t]` gives the absolute position for token `t` in the sequence.
pub fn rope_in_place(x: &mut [f32], positions: &[u32], n_heads: usize, head_dim: usize, base: f32) {
    let seq = positions.len();
    assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");
    assert_eq!(x.len(), seq * n_heads * head_dim);

    let half = head_dim / 2;

    // Phase 7.I: hoist `inv_freq` and the per-(pos, i) `sin_cos` out of the
    // per-head loop. The old code did `n_heads × half` powf + sin_cos calls
    // per token; now it does `half` powf + `seq × half` sin_cos calls per
    // call regardless of head count. For Qwen2 decode (n_heads=14, half=32)
    // that's a 14× reduction in transcendental calls on Q, and 2× on K.
    let inv_freqs: Vec<f32> = (0..half)
        .map(|i| base.powf(-(2.0 * i as f32) / head_dim as f32))
        .collect();
    let mut sins = Vec::with_capacity(seq * half);
    let mut coss = Vec::with_capacity(seq * half);
    for &pos_u32 in positions {
        let pos = pos_u32 as f32;
        for &ifr in &inv_freqs {
            let (s, c) = (pos * ifr).sin_cos();
            sins.push(s);
            coss.push(c);
        }
    }

    for t in 0..seq {
        for h in 0..n_heads {
            let base_idx = t * n_heads * head_dim + h * head_dim;
            for i in 0..half {
                let s = sins[t * half + i];
                let c = coss[t * half + i];
                let a = x[base_idx + i];
                let b = x[base_idx + i + half];
                x[base_idx + i] = a * c - b * s;
                x[base_idx + i + half] = a * s + b * c;
            }
        }
    }
}

/// Numerically stable softmax along the last axis of a 2-D tensor laid out
/// as `[batch, axis]` row-major. Each row sums to 1.
///
/// Standard "subtract max" trick to avoid overflow when inputs are large
/// (matters for attention weights).
pub fn softmax_rows(x: &mut [f32], axis_len: usize) {
    assert_eq!(x.len() % axis_len, 0);
    for row in x.chunks_exact_mut(axis_len) {
        let mut max = f32::NEG_INFINITY;
        for &v in row.iter() {
            if v > max {
                max = v;
            }
        }
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    // ----- rms_norm --------------------------------------------------------

    /// With weights = 1 and inputs sampled so that mean(x²) = 1, RMSNorm
    /// should pass the inputs through nearly unchanged (modulo the eps term).
    #[test]
    fn rms_norm_identity_when_mean_sq_is_one() {
        // x = [1, -1, 1, -1] → mean(x²) = 1.
        let x = vec![1.0, -1.0, 1.0, -1.0];
        let w = vec![1.0; 4];
        let mut y = vec![0.0; 4];
        rms_norm(&x, &w, &mut y, 4, 0.0);
        for (g, want) in y.iter().zip(x.iter()) {
            assert!(approx_eq(*g, *want, 1e-6));
        }
    }

    #[test]
    fn rms_norm_scales_by_weight() {
        // Constant input, constant weight → constant output.
        // x = 2 throughout, mean(x²) = 4, inv_rms = 1/2, scale = 1/2 * w.
        let x = vec![2.0; 4];
        let w = vec![3.0; 4];
        let mut y = vec![0.0; 4];
        rms_norm(&x, &w, &mut y, 4, 0.0);
        // y = 2 * (1/2) * 3 = 3.
        for v in &y {
            assert!(approx_eq(*v, 3.0, 1e-6));
        }
    }

    #[test]
    fn rms_norm_handles_two_rows_independently() {
        let x = vec![1.0, -1.0, 1.0, -1.0, 2.0, 2.0, 2.0, 2.0];
        let w = vec![1.0; 4];
        let mut y = vec![0.0; 8];
        rms_norm(&x, &w, &mut y, 4, 0.0);
        // Row 0: [1, -1, 1, -1] passes through.
        // Row 1: [2, 2, 2, 2] → mean_sq=4 → inv_rms=0.5 → output [1, 1, 1, 1].
        for (g, want) in y
            .iter()
            .zip([1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0].iter())
        {
            assert!(approx_eq(*g, *want, 1e-6), "got {} want {}", g, want);
        }
    }

    // ----- silu ------------------------------------------------------------

    #[test]
    fn silu_zero_is_zero() {
        let mut x = vec![0.0f32];
        silu_in_place(&mut x);
        assert!(approx_eq(x[0], 0.0, 1e-6));
    }

    #[test]
    fn silu_matches_x_times_sigmoid_x() {
        let mut x: Vec<f32> = vec![1.0, 2.0, -1.0, -3.0];
        let expected: Vec<f32> = x
            .iter()
            .map(|&v: &f32| v * (1.0f32 / (1.0 + (-v).exp())))
            .collect();
        silu_in_place(&mut x);
        // Tolerance: the AVX2 path uses a degree-5 polynomial approximation
        // of exp, accurate to ~1e-7 relative — for SiLU values up to ~8 this
        // is ~1e-6 absolute.
        for (g, w) in x.iter().zip(expected.iter()) {
            assert!(approx_eq(*g, *w, 1e-5));
        }
    }

    /// Phase 7.H: drive a long input through the vectorized path (>8 lanes +
    /// a scalar tail) and compare against the scalar reference. Catches
    /// expf-poly drift at extremes and the loop-tail edge case.
    #[test]
    fn silu_vectorized_matches_scalar_long() {
        // 257 = 32 ymm tiles + a 1-element tail. Mix of positive, negative,
        // and near-zero values; a few extreme magnitudes for the clamp.
        let mut got: Vec<f32> = (0..257)
            .map(|i| (((i as f32) - 128.0) * 0.4).sin() * 4.0)
            .collect();
        let mut want = got.clone();
        silu_in_place(&mut got);
        silu_in_place_scalar(&mut want);
        for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-5, "idx {}: got {}  want {}", idx, g, w);
        }
    }

    // ----- mul_in_place ----------------------------------------------------

    #[test]
    fn mul_in_place_works() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        mul_in_place(&mut a, &b);
        assert_eq!(a, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn mul_in_place_vectorized_long() {
        // 41 elements: 5 ymm tiles + 1 tail.
        let mut a: Vec<f32> = (0..41).map(|i| (i as f32) * 0.3 - 5.0).collect();
        let b: Vec<f32> = (0..41).map(|i| ((i as f32) * 0.7).cos()).collect();
        let want: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
        mul_in_place(&mut a, &b);
        for (g, w) in a.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-6, "{} vs {}", g, w);
        }
    }

    // ----- rope ------------------------------------------------------------

    /// At position 0, all rotation angles are 0 → RoPE is identity.
    #[test]
    fn rope_at_position_zero_is_identity() {
        // 1 seq, 1 head, head_dim 4.
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let original = x.clone();
        rope_in_place(&mut x, &[0], 1, 4, 10000.0);
        for (g, w) in x.iter().zip(original.iter()) {
            assert!(approx_eq(*g, *w, 1e-6));
        }
    }

    /// For position 1 and head_dim 2, only one pair, freq = base^0 = 1.
    /// θ = 1 * 1 = 1 rad. Expected:
    ///   x'[0] = a*cos(1) - b*sin(1)
    ///   x'[1] = a*sin(1) + b*cos(1)
    #[test]
    fn rope_explicit_rotation_two_dim_head() {
        let a = 1.0f32;
        let b = 2.0f32;
        let mut x = vec![a, b];
        rope_in_place(&mut x, &[1], 1, 2, 10000.0);
        let (s, c) = 1.0f32.sin_cos();
        assert!(approx_eq(x[0], a * c - b * s, 1e-5));
        assert!(approx_eq(x[1], a * s + b * c, 1e-5));
    }

    /// Rotation preserves the L2 norm of each (lower, upper) pair.
    #[test]
    fn rope_preserves_pair_norm() {
        // 1 seq, 1 head, head_dim 8 → 4 pairs.
        let head_dim = 8;
        let mut x: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.5 - 1.0).collect();
        let original = x.clone();
        rope_in_place(&mut x, &[5], 1, head_dim, 10000.0);
        let half = head_dim / 2;
        for i in 0..half {
            let a0 = original[i];
            let b0 = original[i + half];
            let a1 = x[i];
            let b1 = x[i + half];
            let n0 = a0 * a0 + b0 * b0;
            let n1 = a1 * a1 + b1 * b1;
            assert!(
                approx_eq(n0, n1, 1e-4),
                "pair {}: norm changed {} -> {}",
                i,
                n0,
                n1
            );
        }
    }

    // ----- softmax ---------------------------------------------------------

    #[test]
    fn softmax_rows_sum_to_one() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        softmax_rows(&mut x, 4);
        let s0: f32 = x[..4].iter().sum();
        let s1: f32 = x[4..].iter().sum();
        assert!(approx_eq(s0, 1.0, 1e-6));
        assert!(approx_eq(s1, 1.0, 1e-6));
        // Row of zeros is uniform.
        for v in &x[4..] {
            assert!(approx_eq(*v, 0.25, 1e-6));
        }
    }

    #[test]
    fn softmax_handles_large_inputs() {
        // Without the subtract-max trick this would overflow to NaN.
        let mut x = vec![1000.0, 1001.0, 999.0];
        softmax_rows(&mut x, 3);
        let s: f32 = x.iter().sum();
        assert!(approx_eq(s, 1.0, 1e-6));
        for v in &x {
            assert!(v.is_finite());
        }
    }
}
