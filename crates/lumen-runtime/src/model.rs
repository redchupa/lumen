//! Llama-family transformer block — Hybrid-B forward pass.
//!
//! "Hybrid-B" means: the heavy linear-algebra ops (matmul, dequant-matmul,
//! rms-norm) are intended to come from the JIT-emitted native kernels in
//! `lumen-codegen`, while the small per-element ops (RoPE rotation, SiLU
//! activation, softmax, residual add) use the pure-Rust reference
//! implementations in [`crate::ops`].
//!
//! Phase 6.D scope: layer building blocks + a single-layer prefill forward.
//! No KV cache (that's Phase 6.E), so attention re-derives K/V from the input
//! tokens every call. Causal mask enforced.
//!
//! All weights are fp32 here; quantized weights and the JIT kernel hookup
//! land alongside Phase 6.F when we wire a real GGUF model.

use crate::ops::{mul_in_place, rms_norm, rope_in_place, silu_in_place, softmax_rows};

/// Architecture hyperparameters for one Llama-family transformer.
#[derive(Clone, Debug)]
pub struct LayerConfig {
    pub hidden: usize,     // model dim
    pub n_heads: usize,    // query heads
    pub n_kv_heads: usize, // key/value heads (GQA: n_kv_heads ≤ n_heads)
    pub head_dim: usize,
    pub ffn_hidden: usize,
    pub rms_norm_eps: f32,
    pub rope_base: f32,
}

impl LayerConfig {
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
    /// Number of Q heads that share each KV head.
    pub fn heads_per_kv(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
}

/// All the learned weights one transformer layer needs.
///
/// Layout convention: row-major. For a weight `W` that turns an input of
/// dim `D_in` into an output of dim `D_out`, we store it as `[D_out, D_in]`
/// (one output row per matmul output element). This matches what
/// `matmul(activation [_, D_in], W^T [D_in, D_out])` consumes when we
/// call into the native backend, which expects `A @ B` with shapes
/// `[M, K] @ [K, N]`.
#[derive(Clone, Debug)]
pub struct LayerWeights {
    pub attn_norm_w: Vec<f32>, // [hidden]
    pub wq: Vec<f32>,          // [q_dim, hidden] flattened
    pub wk: Vec<f32>,          // [kv_dim, hidden]
    pub wv: Vec<f32>,          // [kv_dim, hidden]
    pub wo: Vec<f32>,          // [hidden, q_dim]
    pub ffn_norm_w: Vec<f32>,  // [hidden]
    pub w_gate: Vec<f32>,      // [ffn_hidden, hidden]
    pub w_up: Vec<f32>,        // [ffn_hidden, hidden]
    pub w_down: Vec<f32>,      // [hidden, ffn_hidden]
}

/// Naive `[M, K] @ [K, N] -> [M, N]` matmul. The Phase 6.F integration will
/// swap each call for a JIT-compiled native kernel of the right shape; the
/// signature stays identical so the executor doesn't change.
fn matmul_naive(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
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

/// `wmatmul(a, w)` computes `a @ w^T`. `w` is stored row-major as
/// `[d_out, d_in]`; this is the standard ggml/llama.cpp layout.
fn weight_matmul(a: &[f32], w: &[f32], rows: usize, d_in: usize, d_out: usize) -> Vec<f32> {
    // Equivalent to `a [rows, d_in]  @  w^T [d_in, d_out]`.
    // Compute it inline (no explicit transpose) by indexing w accordingly.
    let mut out = vec![0.0f32; rows * d_out];
    for r in 0..rows {
        for o in 0..d_out {
            let mut acc = 0.0f32;
            for k in 0..d_in {
                acc += a[r * d_in + k] * w[o * d_in + k];
            }
            out[r * d_out + o] = acc;
        }
    }
    out
}

/// Multi-head self-attention with optional GQA, causal masking, and *no* KV
/// cache (re-derives K/V for the whole sequence each call). Inputs:
/// - `q`: [seq, n_heads, head_dim]  (post-RoPE)
/// - `k`: [seq, n_kv_heads, head_dim] (post-RoPE)
/// - `v`: [seq, n_kv_heads, head_dim]
/// - `seq`: sequence length
///
/// Output: `[seq, n_heads * head_dim]` (concatenated heads, ready for `wo`).
pub fn multi_head_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    cfg: &LayerConfig,
) -> Vec<f32> {
    assert_eq!(q.len(), seq * cfg.q_dim());
    assert_eq!(k.len(), seq * cfg.kv_dim());
    assert_eq!(v.len(), seq * cfg.kv_dim());

    let h = cfg.n_heads;
    let kvh = cfg.n_kv_heads;
    let hd = cfg.head_dim;
    let group = cfg.heads_per_kv();
    let scale = 1.0f32 / (hd as f32).sqrt();

    let mut out = vec![0.0f32; seq * h * hd];

    for t in 0..seq {
        for head in 0..h {
            let kv_head = head / group;
            // q row for this token & head
            let q_off = t * h * hd + head * hd;
            // scores[i] = q · k[i]  for i ≤ t (causal)
            let mut scores = vec![f32::NEG_INFINITY; t + 1];
            for (i, score) in scores.iter_mut().enumerate() {
                let k_off = i * kvh * hd + kv_head * hd;
                let mut s = 0.0f32;
                for d in 0..hd {
                    s += q[q_off + d] * k[k_off + d];
                }
                *score = s * scale;
            }
            // Softmax over the valid causal range.
            softmax_rows(&mut scores, t + 1);
            // Weighted sum of value rows.
            let out_off = t * h * hd + head * hd;
            for (i, &w) in scores.iter().enumerate() {
                let v_off = i * kvh * hd + kv_head * hd;
                for d in 0..hd {
                    out[out_off + d] += w * v[v_off + d];
                }
            }
        }
    }
    out
}

/// One full transformer layer forward pass on `seq` tokens at once (prefill).
/// Returns the post-layer hidden state, same shape as input `[seq, hidden]`.
pub fn forward_layer(
    x: &[f32],
    seq: usize,
    layer: &LayerWeights,
    positions: &[u32],
    cfg: &LayerConfig,
) -> Vec<f32> {
    assert_eq!(x.len(), seq * cfg.hidden);
    assert_eq!(positions.len(), seq);

    // --- 1. attention RMSNorm ---
    let mut x_norm = vec![0.0f32; x.len()];
    rms_norm(
        x,
        &layer.attn_norm_w,
        &mut x_norm,
        cfg.hidden,
        cfg.rms_norm_eps,
    );

    // --- 2. Q / K / V projection ---
    let mut q = weight_matmul(&x_norm, &layer.wq, seq, cfg.hidden, cfg.q_dim());
    let mut k = weight_matmul(&x_norm, &layer.wk, seq, cfg.hidden, cfg.kv_dim());
    let v = weight_matmul(&x_norm, &layer.wv, seq, cfg.hidden, cfg.kv_dim());

    // --- 3. RoPE on Q and K (reshape view-only: layout already
    //         [seq, heads, head_dim]). ---
    rope_in_place(&mut q, positions, cfg.n_heads, cfg.head_dim, cfg.rope_base);
    rope_in_place(
        &mut k,
        positions,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.rope_base,
    );

    // --- 4. attention ---
    let attn = multi_head_attention(&q, &k, &v, seq, cfg);

    // --- 5. output projection ---
    let attn_out = weight_matmul(&attn, &layer.wo, seq, cfg.q_dim(), cfg.hidden);

    // --- 6. residual ---
    let mut resid: Vec<f32> = x.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();

    // --- 7. FFN RMSNorm ---
    let mut ffn_in = vec![0.0f32; resid.len()];
    rms_norm(
        &resid,
        &layer.ffn_norm_w,
        &mut ffn_in,
        cfg.hidden,
        cfg.rms_norm_eps,
    );

    // --- 8. gate / up projection ---
    let mut gate = weight_matmul(&ffn_in, &layer.w_gate, seq, cfg.hidden, cfg.ffn_hidden);
    let up = weight_matmul(&ffn_in, &layer.w_up, seq, cfg.hidden, cfg.ffn_hidden);

    // --- 9. SiLU(gate) * up ---
    silu_in_place(&mut gate);
    mul_in_place(&mut gate, &up);

    // --- 10. down projection ---
    let ffn_out = weight_matmul(&gate, &layer.w_down, seq, cfg.ffn_hidden, cfg.hidden);

    // --- 11. residual ---
    for (r, o) in resid.iter_mut().zip(ffn_out.iter()) {
        *r += *o;
    }
    resid
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_cfg() -> LayerConfig {
        LayerConfig {
            hidden: 8,
            n_heads: 2,
            n_kv_heads: 1, // GQA: 2 Q heads share 1 KV head
            head_dim: 4,   // n_heads * head_dim = 8 = hidden ✓
            ffn_hidden: 16,
            rms_norm_eps: 1e-5,
            rope_base: 10000.0,
        }
    }

    fn random_weights(cfg: &LayerConfig, seed: u64) -> LayerWeights {
        // Tiny xorshift PRNG, no extra dep.
        let mut s = seed | 1;
        let mut next = || -> f32 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s as i32 as f32) * 1e-10
        };
        let h = cfg.hidden;
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();
        let ff = cfg.ffn_hidden;
        let mut mk = |n: usize| -> Vec<f32> { (0..n).map(|_| next()).collect() };
        LayerWeights {
            attn_norm_w: vec![1.0; h],
            wq: mk(qd * h),
            wk: mk(kvd * h),
            wv: mk(kvd * h),
            wo: mk(h * qd),
            ffn_norm_w: vec![1.0; h],
            w_gate: mk(ff * h),
            w_up: mk(ff * h),
            w_down: mk(h * ff),
        }
    }

    #[test]
    fn single_token_forward_keeps_shape() {
        let cfg = toy_cfg();
        let layer = random_weights(&cfg, 42);
        let x = vec![0.5f32; cfg.hidden];
        let positions = vec![0u32];
        let out = forward_layer(&x, 1, &layer, &positions, &cfg);
        assert_eq!(out.len(), cfg.hidden);
        // All output values are finite.
        for v in &out {
            assert!(v.is_finite(), "non-finite output: {}", v);
        }
    }

    #[test]
    fn multi_token_prefill_shape_and_finite() {
        let cfg = toy_cfg();
        let layer = random_weights(&cfg, 7);
        let seq = 4;
        let x: Vec<f32> = (0..seq * cfg.hidden)
            .map(|i| (i as f32) * 0.01 - 0.1)
            .collect();
        let positions: Vec<u32> = (0..seq as u32).collect();
        let out = forward_layer(&x, seq, &layer, &positions, &cfg);
        assert_eq!(out.len(), seq * cfg.hidden);
        for v in &out {
            assert!(v.is_finite());
        }
    }

    /// Causal masking sanity check: feeding token 0 alone vs. feeding [0, 1]
    /// must give the *same* row-0 output (causality means token 0 can't see
    /// token 1).
    #[test]
    fn attention_is_causal() {
        let cfg = toy_cfg();
        let layer = random_weights(&cfg, 9);

        let x_one = vec![0.3f32; cfg.hidden];
        let out_one = forward_layer(&x_one, 1, &layer, &[0], &cfg);

        let mut x_two = vec![0.3f32; cfg.hidden];
        x_two.extend(vec![0.7f32; cfg.hidden]); // a different second token
        let out_two = forward_layer(&x_two, 2, &layer, &[0, 1], &cfg);

        // Row 0 of out_two should equal out_one.
        for i in 0..cfg.hidden {
            let a = out_one[i];
            let b = out_two[i];
            assert!(
                (a - b).abs() < 1e-4,
                "causality violated at index {}: 1-token={} 2-token row0={}",
                i,
                a,
                b
            );
        }
    }

    /// GQA sanity: with n_kv_heads = n_heads (no grouping) the layer still
    /// runs and `heads_per_kv` is 1.
    #[test]
    fn gqa_with_single_group_works() {
        let mut cfg = toy_cfg();
        cfg.n_kv_heads = cfg.n_heads;
        assert_eq!(cfg.heads_per_kv(), 1);
        let layer = random_weights(&cfg, 11);
        let x = vec![0.2f32; cfg.hidden * 2];
        let out = forward_layer(&x, 2, &layer, &[0, 1], &cfg);
        assert_eq!(out.len(), 2 * cfg.hidden);
        for v in &out {
            assert!(v.is_finite());
        }
    }
}
