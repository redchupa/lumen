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
pub fn silu_in_place(x: &mut [f32]) {
    for v in x.iter_mut() {
        let s = 1.0f32 / (1.0 + (-*v).exp());
        *v *= s;
    }
}

/// Element-wise multiply, in-place: `lhs[i] *= rhs[i]`. Used in Llama FFN
/// after the SiLU gate (`down(silu(gate) * up)` pattern).
pub fn mul_in_place(lhs: &mut [f32], rhs: &[f32]) {
    assert_eq!(lhs.len(), rhs.len());
    for (l, &r) in lhs.iter_mut().zip(rhs.iter()) {
        *l *= r;
    }
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
    let _ = seq; // shape assertion above already used it
    for (t, &pos_u32) in positions.iter().enumerate() {
        let pos = pos_u32 as f32;
        for h in 0..n_heads {
            let base_idx = t * n_heads * head_dim + h * head_dim;
            for i in 0..half {
                let inv_freq = base.powf(-(2.0 * i as f32) / head_dim as f32);
                let theta = pos * inv_freq;
                let (sin_t, cos_t) = theta.sin_cos();
                let a = x[base_idx + i];
                let b = x[base_idx + i + half];
                x[base_idx + i] = a * cos_t - b * sin_t;
                x[base_idx + i + half] = a * sin_t + b * cos_t;
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
        for (g, w) in x.iter().zip(expected.iter()) {
            assert!(approx_eq(*g, *w, 1e-6));
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
