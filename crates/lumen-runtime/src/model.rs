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

use crate::kvcache::LayerKvCache;
use crate::ops::{mul_in_place, rms_norm, rope_in_place, silu_in_place, softmax_rows};
use crate::quant::{dequantize_q8_0, quantize_q8_0, BlockQ8_0, QK};
use lumen_jit::MatmulJitCache;
use std::borrow::Cow;

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

/// Storage for one large projection weight.
///
/// The naive forward path always sees `[d_out, d_in]` row-major. The JIT
/// path's F32 kernels need `[d_in, d_out]` so [`LayerWeights::transpose_in_place`]
/// flips the F32 variant. The Q8 variant is consumed by the fused Q8×F32
/// kernel, which reads weights in their native `[d_out, d_in]` ggml layout
/// and therefore never needs transposing.
#[derive(Clone, Debug, PartialEq)]
pub enum WeightStorage {
    /// F32 weights. Layout is `[d_out, d_in]` until `transpose_in_place` runs,
    /// then `[d_in, d_out]`.
    F32(Vec<f32>),
    /// Q8_0 weights packed as native ggml `[d_out, d_in]` blocks (one block
    /// covers 32 contiguous d_in elements within a row).
    Q8(Vec<BlockQ8_0>),
}

impl WeightStorage {
    /// Total number of f32 elements this weight represents (matches what
    /// `as_f32_native` would yield).
    pub fn nelem(&self) -> usize {
        match self {
            WeightStorage::F32(v) => v.len(),
            WeightStorage::Q8(blocks) => blocks.len() * QK,
        }
    }

    /// Return the weight as F32 in `[d_out, d_in]` layout for the naive path.
    /// Borrows the F32 case zero-copy; allocates and dequantizes for Q8.
    /// Only valid before `transpose_in_place` has been called.
    pub fn as_f32_native(&self, d_out: usize, d_in: usize) -> Cow<'_, [f32]> {
        match self {
            WeightStorage::F32(v) => {
                debug_assert_eq!(v.len(), d_out * d_in);
                Cow::Borrowed(v)
            }
            WeightStorage::Q8(blocks) => {
                debug_assert_eq!(blocks.len() * QK, d_out * d_in);
                let mut out = vec![0.0f32; d_out * d_in];
                dequantize_q8_0(blocks, &mut out);
                Cow::Owned(out)
            }
        }
    }
}

/// All the learned weights one transformer layer needs.
///
/// Layout convention for the seven large projections: row-major. For a weight
/// `W` that turns an input of dim `D_in` into an output of dim `D_out`, the
/// natural ggml layout is `[D_out, D_in]`. F32 storage may be transposed
/// in-place to `[D_in, D_out]` for the JIT path; Q8 storage stays native.
#[derive(Clone, Debug)]
pub struct LayerWeights {
    pub attn_norm_w: Vec<f32>, // [hidden]
    pub wq: WeightStorage,     // [q_dim, hidden]
    pub wk: WeightStorage,     // [kv_dim, hidden]
    pub wv: WeightStorage,     // [kv_dim, hidden]
    pub wo: WeightStorage,     // [hidden, q_dim]
    pub ffn_norm_w: Vec<f32>,  // [hidden]
    pub w_gate: WeightStorage, // [ffn_hidden, hidden]
    pub w_up: WeightStorage,   // [ffn_hidden, hidden]
    pub w_down: WeightStorage, // [hidden, ffn_hidden]
    // Qwen2-style attention biases (Llama2/Llama3 omit these → `None`).
    pub b_q: Option<Vec<f32>>, // [q_dim]
    pub b_k: Option<Vec<f32>>, // [kv_dim]
    pub b_v: Option<Vec<f32>>, // [kv_dim]
}

/// Transpose a `[d_out, d_in]` row-major matrix into `[d_in, d_out]` row-major,
/// returning a fresh allocation. Used to convert ggml-convention weights into
/// a layout the Lumen JIT matmul kernels can consume directly.
pub fn transpose_2d(src: &[f32], d_out: usize, d_in: usize) -> Vec<f32> {
    debug_assert_eq!(src.len(), d_out * d_in);
    let mut out = vec![0.0f32; d_out * d_in];
    for o in 0..d_out {
        for k in 0..d_in {
            out[k * d_out + o] = src[o * d_in + k];
        }
    }
    out
}

impl LayerWeights {
    /// Transpose every F32 weight matrix from `[d_out, d_in]` (ggml convention)
    /// into `[d_in, d_out]` (the layout the JIT matmul kernels expect for
    /// their `B` operand). Q8 weights are left untouched — the fused Q8×F32
    /// kernel consumes them in their native `[d_out, d_in]` layout. After
    /// calling this, the F32 weights are *only* usable via the `_jit` forward
    /// functions.
    pub fn transpose_in_place(&mut self, cfg: &LayerConfig) {
        let h = cfg.hidden;
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();
        let ff = cfg.ffn_hidden;
        transpose_storage_in_place(&mut self.wq, qd, h);
        transpose_storage_in_place(&mut self.wk, kvd, h);
        transpose_storage_in_place(&mut self.wv, kvd, h);
        transpose_storage_in_place(&mut self.wo, h, qd);
        transpose_storage_in_place(&mut self.w_gate, ff, h);
        transpose_storage_in_place(&mut self.w_up, ff, h);
        transpose_storage_in_place(&mut self.w_down, h, ff);
    }
}

/// Transpose just the F32 case in place. Q8 is a no-op (native layout already
/// matches what the fused kernel wants).
fn transpose_storage_in_place(w: &mut WeightStorage, d_out: usize, d_in: usize) {
    if let WeightStorage::F32(v) = w {
        *v = transpose_2d(v, d_out, d_in);
    }
}

/// JIT-backed matmul: `out [rows, d_out] = a [rows, d_in] @ w_t [d_in, d_out]`.
/// `w_t` must already be in transposed (post-`transpose_in_place`) layout.
fn weight_matmul_jit(
    a: &[f32],
    w_t: &[f32],
    rows: usize,
    d_in: usize,
    d_out: usize,
    jit: &mut MatmulJitCache,
) -> Vec<f32> {
    let f = jit
        .get_or_compile(rows as u32, d_in as u32, d_out as u32)
        .expect("matmul JIT compile");
    let mut out = vec![0.0f32; rows * d_out];
    // SAFETY: cache returned a kernel compiled for exactly (rows, d_in, d_out).
    unsafe {
        f(a.as_ptr(), w_t.as_ptr(), out.as_mut_ptr());
    }
    out
}

/// JIT-backed matmul that dispatches per [`WeightStorage`] variant. Only
/// supports `rows == 1` (decode); we have no native Q8 path for rows > 1 yet.
///
/// For `WeightStorage::F32`, calls into the existing fp32 kernel cache assuming
/// the F32 buffer has been transposed to `[d_in, d_out]` by `transpose_in_place`.
///
/// For `WeightStorage::Q8`, calls into the fused Q8×F32 N=1 kernel which
/// reads weights in their native `[d_out, d_in]` ggml layout — no transpose,
/// no dequant pass, no extra fp32 buffer.
///
/// Phase 7.J: large Q8 matmuls (M ≥ MULTI_THREAD_M_THRESHOLD) are split
/// across rayon worker threads on the M dimension. Each chunk runs the same
/// JIT kernel on a row-slice of weights and output; activations are shared
/// read-only across threads.
fn weight_matmul_jit_storage(
    a: &[f32],
    w: &WeightStorage,
    d_in: usize,
    d_out: usize,
    jit: &mut MatmulJitCache,
) -> Vec<f32> {
    let mut out = vec![0.0f32; d_out];
    match w {
        WeightStorage::F32(buf) => {
            let f = jit
                .get_or_compile(1, d_in as u32, d_out as u32)
                .expect("f32 matmul JIT compile");
            // SAFETY: cache returned a kernel compiled for (1, d_in, d_out).
            unsafe {
                f(a.as_ptr(), buf.as_ptr(), out.as_mut_ptr());
            }
        }
        WeightStorage::Q8(blocks) => {
            q8_matmul_dispatch(a, blocks, d_in, d_out, jit, &mut out);
        }
    }
    out
}

/// Minimum total Q8 block work (M × K_blocks, where K_blocks = K/32) for
/// parallelizing a matmul. Below this, the ~5µs/call rayon dispatch
/// overhead outweighs the saved compute. Phase 7.J profiling on Qwen2.5-0.5B
/// showed wq/wo (M=896, K_blocks=28 = 25K work units) regressed 25-30% with
/// rayon, while gate_up (M=4864, K_blocks=28 = 136K) and down (M=896,
/// K_blocks=152 = 136K) sped up ~2×.
const MULTI_THREAD_WORK_THRESHOLD: usize = 100_000;

/// Quantize an `[f32; N]` activation buffer into Q8_0 blocks. `N` must be
/// a multiple of 32. Used by the decode forward to feed the Q8×Q8 fused
/// kernel (Phase 7.M groundwork; activated in Phase 7.N when the host CPU
/// supports vpdpbusd).
fn quantize_activation_q8(x: &[f32]) -> Vec<BlockQ8_0> {
    debug_assert!(
        x.len() % QK == 0,
        "activation length must be a multiple of 32"
    );
    let mut blocks = vec![
        BlockQ8_0 {
            d: 0,
            qs: [0i8; 32]
        };
        x.len() / QK
    ];
    quantize_q8_0(x, &mut blocks);
    blocks
}

/// True when the current CPU has either AVX-VNNI (VEX) or AVX-512 VNNI
/// (EVEX) — i.e. `vpdpbusd` will be emitted by the Q8×Q8 kernel and the
/// activation-quantization pipeline is worth the cost.
fn has_vnni() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512vnni") || std::is_x86_feature_detected!("avxvnni")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Phase 7.P: pick VNNI vs fp32 path *per matmul shape*.
///
/// Phase 7.O's measurement showed VNNI (vpdpbusd) wins on short-K matmuls
/// (gate_up/qkv/wo/lm_head, K_blocks=28: -9% to -28%) but loses on long-K
/// matmuls (FFN down, K_blocks=152: +27%). Diagnosis: VNNI's 14
/// instructions per block × long inner loop saturates the CPU OoO window,
/// while the fp32 4-acc kernel's leaner 4-instruction inner overpacks the
/// pipeline and beats VNNI when there are many blocks to process serially.
///
/// Empirical threshold from the same measurement: K_blocks ≤ 64 → VNNI
/// wins; > 64 → fp32 4-acc wins. For Qwen2.5-0.5B this routes all five
/// short-K matmuls to VNNI and the one long-K matmul (down) to fp32.
fn use_vnni_for_matmul(d_in: usize) -> bool {
    if !has_vnni() {
        return false;
    }
    (d_in / 32) <= 64
}

/// Activation-quantized variant of [`weight_matmul_jit_storage`]. Caller
/// supplies the activation pre-quantized to Q8_0 blocks (typically reused
/// across the Q/K/V projections from one RMSNorm output). For F32 weights
/// (test-only) we dequantize the activation back and fall through to the
/// fp32 path.
fn weight_matmul_jit_storage_q8act(
    a_blocks: &[BlockQ8_0],
    w: &WeightStorage,
    d_in: usize,
    d_out: usize,
    jit: &mut MatmulJitCache,
) -> Vec<f32> {
    let mut out = vec![0.0f32; d_out];
    match w {
        WeightStorage::F32(buf) => {
            let mut a_f32 = vec![0.0f32; d_in];
            dequantize_q8_0(a_blocks, &mut a_f32);
            let f = jit
                .get_or_compile(1, d_in as u32, d_out as u32)
                .expect("f32 matmul JIT compile");
            // SAFETY: kernel compiled for (1, d_in, d_out); buffers sized.
            unsafe {
                f(a_f32.as_ptr(), buf.as_ptr(), out.as_mut_ptr());
            }
        }
        WeightStorage::Q8(weight_blocks) => {
            q8q8_matmul_dispatch(a_blocks, weight_blocks, d_in, d_out, jit, &mut out);
        }
    }
    out
}

/// Dispatch a Q8×Q8 N=1 matmul, possibly across multiple pool worker threads.
/// Same M-direction chunking + work-based threshold as `q8_matmul_dispatch`.
fn q8q8_matmul_dispatch(
    a_blocks: &[BlockQ8_0],
    weights: &[BlockQ8_0],
    d_in: usize,
    d_out: usize,
    jit: &mut MatmulJitCache,
    out: &mut [f32],
) {
    let pool = crate::threadpool::global();
    let nthreads = pool.n_workers();
    let work_units = d_out * (d_in / 32);

    if work_units < MULTI_THREAD_WORK_THRESHOLD || nthreads <= 1 {
        let f = jit
            .get_or_compile_q8q8(d_out as u32, d_in as u32)
            .expect("q8q8 matmul JIT compile");
        // SAFETY: kernel compiled for exactly (d_out, d_in, 1); buffers sized.
        unsafe {
            f(
                weights.as_ptr() as *const u8,
                a_blocks.as_ptr() as *const u8,
                out.as_mut_ptr(),
            );
        }
        return;
    }

    let chunk_rows = d_out.div_ceil(nthreads);
    let n_chunks = d_out.div_ceil(chunk_rows);
    let last_rows = d_out - (n_chunks - 1) * chunk_rows;

    let f_reg = jit
        .get_or_compile_q8q8(chunk_rows as u32, d_in as u32)
        .expect("q8q8 matmul JIT compile (chunk)");
    let f_last = if last_rows != chunk_rows {
        jit.get_or_compile_q8q8(last_rows as u32, d_in as u32)
            .expect("q8q8 matmul JIT compile (tail)")
    } else {
        f_reg
    };

    let k_blocks = d_in / 32;
    let row_bytes = k_blocks * 34;

    let weights_base_addr: usize = weights.as_ptr() as usize;
    let acts_base_addr: usize = a_blocks.as_ptr() as usize;
    let out_base_addr: usize = out.as_mut_ptr() as usize;

    pool.parallel_for(n_chunks, |chunk_idx| {
        let row_start = chunk_idx * chunk_rows;
        let actual_rows = (d_out - row_start).min(chunk_rows);
        let fn_ptr = if actual_rows == chunk_rows {
            f_reg
        } else {
            f_last
        };
        // SAFETY: row_start * row_bytes is within the original weights slice;
        // each thread writes a disjoint output row range; activations are
        // shared read-only.
        let w_ptr = (weights_base_addr + row_start * row_bytes) as *const u8;
        let a_ptr = acts_base_addr as *const u8;
        let out_ptr = (out_base_addr + row_start * std::mem::size_of::<f32>()) as *mut f32;
        unsafe {
            fn_ptr(w_ptr, a_ptr, out_ptr);
        }
    });
}

/// Dispatch a Q8 N=1 matmul, possibly across multiple rayon threads.
///
/// Splits the M (output row) dimension into roughly equal chunks, one per
/// worker thread. Each thread runs an independently-compiled kernel sized
/// for its chunk's row count on a slice of `weights` and `out`. `activations`
/// is borrowed read-only by every thread (no contention).
fn q8_matmul_dispatch(
    a: &[f32],
    weights: &[crate::quant::BlockQ8_0],
    d_in: usize,
    d_out: usize,
    jit: &mut MatmulJitCache,
    out: &mut [f32],
) {
    let pool = crate::threadpool::global();
    let nthreads = pool.n_workers();
    let work_units = d_out * (d_in / 32);

    if work_units < MULTI_THREAD_WORK_THRESHOLD || nthreads <= 1 {
        // Serial fast path.
        let f = jit
            .get_or_compile_q8(d_out as u32, d_in as u32, 1)
            .expect("q8 matmul JIT compile");
        // SAFETY: kernel compiled for exactly (d_out, d_in, 1); buffers sized.
        unsafe {
            f(weights.as_ptr() as *const u8, a.as_ptr(), out.as_mut_ptr());
        }
        return;
    }

    // Parallel path — one chunk per worker thread.
    let chunk_rows = d_out.div_ceil(nthreads);
    let n_chunks = d_out.div_ceil(chunk_rows);
    let last_rows = d_out - (n_chunks - 1) * chunk_rows;

    // Pre-compile both kernels in the serial section (cache requires &mut).
    let f_reg = jit
        .get_or_compile_q8(chunk_rows as u32, d_in as u32, 1)
        .expect("q8 matmul JIT compile (chunk)");
    let f_last = if last_rows != chunk_rows {
        jit.get_or_compile_q8(last_rows as u32, d_in as u32, 1)
            .expect("q8 matmul JIT compile (tail)")
    } else {
        f_reg
    };

    // One Q8_0 block packs 32 K elements; one row holds k_blocks blocks of 34B each.
    let k_blocks = d_in / 32;
    let row_bytes = k_blocks * 34;

    // Cast pointers to usize for cross-thread transport (raw pointers aren't
    // Send/Sync). Tasks reconstruct typed pointers on the other side. SAFETY:
    // each task reads/writes a disjoint row range, computed from chunk_idx.
    let weights_base_addr: usize = weights.as_ptr() as usize;
    let acts_base_addr: usize = a.as_ptr() as usize;
    let out_base_addr: usize = out.as_mut_ptr() as usize;

    pool.parallel_for(n_chunks, |chunk_idx| {
        let row_start = chunk_idx * chunk_rows;
        let actual_rows = (d_out - row_start).min(chunk_rows);
        let fn_ptr = if actual_rows == chunk_rows {
            f_reg
        } else {
            f_last
        };
        // SAFETY: pointer arithmetic stays within the original buffers;
        // each chunk reads `actual_rows * row_bytes` bytes of weights starting
        // at `row_start * row_bytes`, reads `d_in` activations (shared), and
        // writes `actual_rows` floats to out at offset `row_start`. Output
        // chunks are disjoint per chunk_idx.
        let w_ptr = (weights_base_addr + row_start * row_bytes) as *const u8;
        let a_ptr = acts_base_addr as *const f32;
        let out_ptr = (out_base_addr + row_start * std::mem::size_of::<f32>()) as *mut f32;
        unsafe {
            fn_ptr(w_ptr, a_ptr, out_ptr);
        }
    });
}

/// Add `bias` broadcast across each row of `x` (shape `[rows, dim]`).
fn add_bias_broadcast(x: &mut [f32], bias: &[f32], dim: usize) {
    debug_assert_eq!(x.len() % dim, 0);
    debug_assert_eq!(bias.len(), dim);
    for row in x.chunks_exact_mut(dim) {
        for (a, b) in row.iter_mut().zip(bias.iter()) {
            *a += b;
        }
    }
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

    // --- 2. Q / K / V projection (+ optional Qwen2 biases) ---
    let wq = layer.wq.as_f32_native(cfg.q_dim(), cfg.hidden);
    let wk = layer.wk.as_f32_native(cfg.kv_dim(), cfg.hidden);
    let wv = layer.wv.as_f32_native(cfg.kv_dim(), cfg.hidden);
    let mut q = weight_matmul(&x_norm, &wq, seq, cfg.hidden, cfg.q_dim());
    let mut k = weight_matmul(&x_norm, &wk, seq, cfg.hidden, cfg.kv_dim());
    let mut v = weight_matmul(&x_norm, &wv, seq, cfg.hidden, cfg.kv_dim());
    if let Some(b) = &layer.b_q {
        add_bias_broadcast(&mut q, b, cfg.q_dim());
    }
    if let Some(b) = &layer.b_k {
        add_bias_broadcast(&mut k, b, cfg.kv_dim());
    }
    if let Some(b) = &layer.b_v {
        add_bias_broadcast(&mut v, b, cfg.kv_dim());
    }

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
    let wo = layer.wo.as_f32_native(cfg.hidden, cfg.q_dim());
    let attn_out = weight_matmul(&attn, &wo, seq, cfg.q_dim(), cfg.hidden);

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
    let w_gate = layer.w_gate.as_f32_native(cfg.ffn_hidden, cfg.hidden);
    let w_up = layer.w_up.as_f32_native(cfg.ffn_hidden, cfg.hidden);
    let mut gate = weight_matmul(&ffn_in, &w_gate, seq, cfg.hidden, cfg.ffn_hidden);
    let up = weight_matmul(&ffn_in, &w_up, seq, cfg.hidden, cfg.ffn_hidden);

    // --- 9. SiLU(gate) * up ---
    silu_in_place(&mut gate);
    mul_in_place(&mut gate, &up);

    // --- 10. down projection ---
    let w_down = layer.w_down.as_f32_native(cfg.hidden, cfg.ffn_hidden);
    let ffn_out = weight_matmul(&gate, &w_down, seq, cfg.ffn_hidden, cfg.hidden);

    // --- 11. residual ---
    for (r, o) in resid.iter_mut().zip(ffn_out.iter()) {
        *r += *o;
    }
    resid
}

/// Cache-backed attention for a single query token at position `cur_pos`.
///
/// Inputs:
/// - `q_t`: rotated query for this token, shape `[n_heads, head_dim]`.
/// - `cache`: layer cache, already containing the new K/V row appended.
///   So `cache.len() == cur_pos + 1` (caller is responsible for the append).
///
/// Returns: attention output for this token, shape `[n_heads * head_dim]`.
fn attention_decode(q_t: &[f32], cache: &LayerKvCache, cfg: &LayerConfig) -> Vec<f32> {
    let h = cfg.n_heads;
    let kvh = cfg.n_kv_heads;
    let hd = cfg.head_dim;
    let group = cfg.heads_per_kv();
    let scale = 1.0f32 / (hd as f32).sqrt();
    let t_plus_1 = cache.len();
    debug_assert!(t_plus_1 >= 1);

    let k_filled = cache.k_filled();
    let v_filled = cache.v_filled();

    let mut out = vec![0.0f32; h * hd];

    for head in 0..h {
        let kv_head = head / group;
        let q_off = head * hd;

        let mut scores = vec![0.0f32; t_plus_1];
        for (i, score) in scores.iter_mut().enumerate() {
            let k_off = i * kvh * hd + kv_head * hd;
            let mut s = 0.0f32;
            for d in 0..hd {
                s += q_t[q_off + d] * k_filled[k_off + d];
            }
            *score = s * scale;
        }
        softmax_rows(&mut scores, t_plus_1);

        let out_off = head * hd;
        for (i, &w) in scores.iter().enumerate() {
            let v_off = i * kvh * hd + kv_head * hd;
            for d in 0..hd {
                out[out_off + d] += w * v_filled[v_off + d];
            }
        }
    }
    out
}

/// Single-token decode forward through one transformer layer, mutating the
/// per-layer KV cache. The input `x_t` is `[hidden]`; the output replaces
/// it (also `[hidden]`).
///
/// Steps mirror `forward_layer` but with `seq == 1` and Q/K/V rows
/// appended to the cache instead of recomputed for every past token.
pub fn forward_layer_decode(
    x_t: &[f32],
    position: u32,
    layer: &LayerWeights,
    cache: &mut LayerKvCache,
    cfg: &LayerConfig,
) -> Vec<f32> {
    assert_eq!(x_t.len(), cfg.hidden);
    assert_eq!(cache.kv_dim(), cfg.kv_dim());

    // 1. attention RMSNorm
    let mut x_norm = vec![0.0f32; cfg.hidden];
    rms_norm(
        x_t,
        &layer.attn_norm_w,
        &mut x_norm,
        cfg.hidden,
        cfg.rms_norm_eps,
    );

    // 2. Q / K / V projection (single row each), plus optional Qwen2 biases.
    let wq = layer.wq.as_f32_native(cfg.q_dim(), cfg.hidden);
    let wk = layer.wk.as_f32_native(cfg.kv_dim(), cfg.hidden);
    let wv = layer.wv.as_f32_native(cfg.kv_dim(), cfg.hidden);
    let mut q = weight_matmul(&x_norm, &wq, 1, cfg.hidden, cfg.q_dim());
    let mut k = weight_matmul(&x_norm, &wk, 1, cfg.hidden, cfg.kv_dim());
    let mut v = weight_matmul(&x_norm, &wv, 1, cfg.hidden, cfg.kv_dim());
    if let Some(b) = &layer.b_q {
        add_bias_broadcast(&mut q, b, cfg.q_dim());
    }
    if let Some(b) = &layer.b_k {
        add_bias_broadcast(&mut k, b, cfg.kv_dim());
    }
    if let Some(b) = &layer.b_v {
        add_bias_broadcast(&mut v, b, cfg.kv_dim());
    }

    // 3. RoPE on q and k at the current position
    rope_in_place(
        &mut q,
        &[position],
        cfg.n_heads,
        cfg.head_dim,
        cfg.rope_base,
    );
    rope_in_place(
        &mut k,
        &[position],
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.rope_base,
    );

    // 4. Append K/V to cache.
    cache.append(&k, &v);

    // 5. Cache-backed attention
    let attn = attention_decode(&q, cache, cfg);

    // 6. output projection
    let wo = layer.wo.as_f32_native(cfg.hidden, cfg.q_dim());
    let attn_out = weight_matmul(&attn, &wo, 1, cfg.q_dim(), cfg.hidden);

    // 7. residual
    let mut resid: Vec<f32> = x_t
        .iter()
        .zip(attn_out.iter())
        .map(|(a, b)| a + b)
        .collect();

    // 8. FFN RMSNorm
    let mut ffn_in = vec![0.0f32; cfg.hidden];
    rms_norm(
        &resid,
        &layer.ffn_norm_w,
        &mut ffn_in,
        cfg.hidden,
        cfg.rms_norm_eps,
    );

    // 9. gate/up
    let w_gate = layer.w_gate.as_f32_native(cfg.ffn_hidden, cfg.hidden);
    let w_up = layer.w_up.as_f32_native(cfg.ffn_hidden, cfg.hidden);
    let mut gate = weight_matmul(&ffn_in, &w_gate, 1, cfg.hidden, cfg.ffn_hidden);
    let up = weight_matmul(&ffn_in, &w_up, 1, cfg.hidden, cfg.ffn_hidden);

    // 10. SiLU(gate) * up
    silu_in_place(&mut gate);
    mul_in_place(&mut gate, &up);

    // 11. down
    let w_down = layer.w_down.as_f32_native(cfg.hidden, cfg.ffn_hidden);
    let ffn_out = weight_matmul(&gate, &w_down, 1, cfg.ffn_hidden, cfg.hidden);

    // 12. residual
    for (r, o) in resid.iter_mut().zip(ffn_out.iter()) {
        *r += *o;
    }
    resid
}

use crate::kvcache::KvCache;

/// JIT variant of [`forward_layer_decode`]: same forward semantics, same
/// inputs/outputs, but the seven weight matmuls in the layer go through
/// the JIT cache. Caller must have already called `transpose_in_place` on
/// this layer.
#[allow(clippy::too_many_arguments)]
pub fn forward_layer_decode_jit(
    x_t: &[f32],
    position: u32,
    layer: &LayerWeights,
    cache: &mut LayerKvCache,
    cfg: &LayerConfig,
    jit: &mut MatmulJitCache,
) -> Vec<f32> {
    assert_eq!(x_t.len(), cfg.hidden);
    assert_eq!(cache.kv_dim(), cfg.kv_dim());

    // 1. attention RMSNorm
    let mut x_norm = vec![0.0f32; cfg.hidden];
    rms_norm(
        x_t,
        &layer.attn_norm_w,
        &mut x_norm,
        cfg.hidden,
        cfg.rms_norm_eps,
    );

    // 2. Q/K/V projection through JIT. On CPUs with VNNI (AVX-VNNI or
    //    AVX-512 VNNI), quantize x_norm once and run the three projections
    //    through the Q8×Q8 fused kernel (vpdpbusd). Otherwise fall back to
    //    the Q8×F32 4-acc kernel — Phase 7.M measured the AVX2-only int
    //    chain as net-neutral, so we only switch when vpdpbusd is actually
    //    available.
    // Phase 7.P: per-shape VNNI vs fp32 dispatch. K = cfg.hidden here, so the
    // same choice covers q/k/v.
    let (mut q, mut k, mut v) = if use_vnni_for_matmul(cfg.hidden) {
        let x_norm_q8 = quantize_activation_q8(&x_norm);
        let q =
            weight_matmul_jit_storage_q8act(&x_norm_q8, &layer.wq, cfg.hidden, cfg.q_dim(), jit);
        let k =
            weight_matmul_jit_storage_q8act(&x_norm_q8, &layer.wk, cfg.hidden, cfg.kv_dim(), jit);
        let v =
            weight_matmul_jit_storage_q8act(&x_norm_q8, &layer.wv, cfg.hidden, cfg.kv_dim(), jit);
        (q, k, v)
    } else {
        let q = weight_matmul_jit_storage(&x_norm, &layer.wq, cfg.hidden, cfg.q_dim(), jit);
        let k = weight_matmul_jit_storage(&x_norm, &layer.wk, cfg.hidden, cfg.kv_dim(), jit);
        let v = weight_matmul_jit_storage(&x_norm, &layer.wv, cfg.hidden, cfg.kv_dim(), jit);
        (q, k, v)
    };
    if let Some(b) = &layer.b_q {
        add_bias_broadcast(&mut q, b, cfg.q_dim());
    }
    if let Some(b) = &layer.b_k {
        add_bias_broadcast(&mut k, b, cfg.kv_dim());
    }
    if let Some(b) = &layer.b_v {
        add_bias_broadcast(&mut v, b, cfg.kv_dim());
    }

    // 3. RoPE
    rope_in_place(
        &mut q,
        &[position],
        cfg.n_heads,
        cfg.head_dim,
        cfg.rope_base,
    );
    rope_in_place(
        &mut k,
        &[position],
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.rope_base,
    );

    // 4. Append to cache, attention.
    cache.append(&k, &v);
    let attn = attention_decode(&q, cache, cfg);

    // 5. output projection. K = cfg.q_dim here.
    let attn_out = if use_vnni_for_matmul(cfg.q_dim()) {
        let attn_q8 = quantize_activation_q8(&attn);
        weight_matmul_jit_storage_q8act(&attn_q8, &layer.wo, cfg.q_dim(), cfg.hidden, jit)
    } else {
        weight_matmul_jit_storage(&attn, &layer.wo, cfg.q_dim(), cfg.hidden, jit)
    };

    // 6. residual
    let mut resid: Vec<f32> = x_t
        .iter()
        .zip(attn_out.iter())
        .map(|(a, b)| a + b)
        .collect();

    // 7. FFN RMSNorm
    let mut ffn_in = vec![0.0f32; cfg.hidden];
    rms_norm(
        &resid,
        &layer.ffn_norm_w,
        &mut ffn_in,
        cfg.hidden,
        cfg.rms_norm_eps,
    );

    // 8. gate/up. K = cfg.hidden.
    let (mut gate, up) = if use_vnni_for_matmul(cfg.hidden) {
        let ffn_in_q8 = quantize_activation_q8(&ffn_in);
        let g = weight_matmul_jit_storage_q8act(
            &ffn_in_q8,
            &layer.w_gate,
            cfg.hidden,
            cfg.ffn_hidden,
            jit,
        );
        let u = weight_matmul_jit_storage_q8act(
            &ffn_in_q8,
            &layer.w_up,
            cfg.hidden,
            cfg.ffn_hidden,
            jit,
        );
        (g, u)
    } else {
        let g = weight_matmul_jit_storage(&ffn_in, &layer.w_gate, cfg.hidden, cfg.ffn_hidden, jit);
        let u = weight_matmul_jit_storage(&ffn_in, &layer.w_up, cfg.hidden, cfg.ffn_hidden, jit);
        (g, u)
    };

    // 9. SiLU(gate) * up
    silu_in_place(&mut gate);
    mul_in_place(&mut gate, &up);

    // 10. down. K = cfg.ffn_hidden (long; usually routes to fp32 4-acc).
    let ffn_out = if use_vnni_for_matmul(cfg.ffn_hidden) {
        let gate_q8 = quantize_activation_q8(&gate);
        weight_matmul_jit_storage_q8act(&gate_q8, &layer.w_down, cfg.ffn_hidden, cfg.hidden, jit)
    } else {
        weight_matmul_jit_storage(&gate, &layer.w_down, cfg.ffn_hidden, cfg.hidden, jit)
    };

    // 11. residual
    for (r, o) in resid.iter_mut().zip(ffn_out.iter()) {
        *r += *o;
    }
    resid
}

/// Per-step time accumulator for profiling the JIT decode path.
///
/// Phase 7.E.0 scope: figure out where decode time actually goes before we
/// commit to flash attention vs multi-thread vs anything else. Sum every
/// per-step elapsed time across many decode tokens, then print a sorted
/// breakdown.
#[derive(Default, Debug, Clone)]
pub struct StepTimer {
    pub embed: std::time::Duration,
    pub layer_attn_rms: std::time::Duration,
    pub layer_qkv_matmul: std::time::Duration,
    pub layer_qkv_bias: std::time::Duration,
    pub layer_rope: std::time::Duration,
    pub layer_kv_append: std::time::Duration,
    pub layer_attention: std::time::Duration,
    pub layer_wo_matmul: std::time::Duration,
    pub layer_attn_residual: std::time::Duration,
    pub layer_ffn_rms: std::time::Duration,
    pub layer_gate_up_matmul: std::time::Duration,
    pub layer_silu_mul: std::time::Duration,
    pub layer_down_matmul: std::time::Duration,
    pub layer_ffn_residual: std::time::Duration,
    pub final_rms: std::time::Duration,
    pub lm_head_matmul: std::time::Duration,
}

impl StepTimer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sum of every recorded bucket — should approximate one full forward
    /// (minus the trivial argmax that lives in the generate loop).
    pub fn total(&self) -> std::time::Duration {
        self.embed
            + self.layer_attn_rms
            + self.layer_qkv_matmul
            + self.layer_qkv_bias
            + self.layer_rope
            + self.layer_kv_append
            + self.layer_attention
            + self.layer_wo_matmul
            + self.layer_attn_residual
            + self.layer_ffn_rms
            + self.layer_gate_up_matmul
            + self.layer_silu_mul
            + self.layer_down_matmul
            + self.layer_ffn_residual
            + self.final_rms
            + self.lm_head_matmul
    }

    /// Pretty-print breakdown, sorted by share of total descending.
    pub fn report(&self, label: &str) -> String {
        let total = self.total();
        let total_ns = total.as_nanos().max(1);
        let mut rows: Vec<(&str, std::time::Duration)> = vec![
            ("embed", self.embed),
            ("layer/attn_rms", self.layer_attn_rms),
            ("layer/qkv_matmul", self.layer_qkv_matmul),
            ("layer/qkv_bias", self.layer_qkv_bias),
            ("layer/rope", self.layer_rope),
            ("layer/kv_append", self.layer_kv_append),
            ("layer/attention", self.layer_attention),
            ("layer/wo_matmul", self.layer_wo_matmul),
            ("layer/attn_residual", self.layer_attn_residual),
            ("layer/ffn_rms", self.layer_ffn_rms),
            ("layer/gate_up_matmul", self.layer_gate_up_matmul),
            ("layer/silu_mul", self.layer_silu_mul),
            ("layer/down_matmul", self.layer_down_matmul),
            ("layer/ffn_residual", self.layer_ffn_residual),
            ("final_rms", self.final_rms),
            ("lm_head_matmul", self.lm_head_matmul),
        ];
        rows.sort_by_key(|row| std::cmp::Reverse(row.1));
        let mut s = format!("{} (total {:?}):\n", label, total);
        for (name, d) in rows {
            let pct = (d.as_nanos() as f64 / total_ns as f64) * 100.0;
            s.push_str(&format!("  {:>22}  {:>10?}  {:>5.1}%\n", name, d, pct));
        }
        s
    }
}

/// Profiled variant of [`forward_layer_decode_jit`]: same semantics, but every
/// step's elapsed time is added to the shared `timer`. Used by Phase 7.E.0
/// profiling. Adds ~20 `Instant::now()` calls per layer per token — small
/// enough not to distort the picture meaningfully.
#[allow(clippy::too_many_arguments)]
pub fn forward_layer_decode_jit_timed(
    x_t: &[f32],
    position: u32,
    layer: &LayerWeights,
    cache: &mut LayerKvCache,
    cfg: &LayerConfig,
    jit: &mut MatmulJitCache,
    timer: &mut StepTimer,
) -> Vec<f32> {
    use std::time::Instant;

    // 1. attention RMSNorm
    let t = Instant::now();
    let mut x_norm = vec![0.0f32; cfg.hidden];
    rms_norm(
        x_t,
        &layer.attn_norm_w,
        &mut x_norm,
        cfg.hidden,
        cfg.rms_norm_eps,
    );
    timer.layer_attn_rms += t.elapsed();

    // 2. Q/K/V projection — Phase 7.P per-shape dispatch (K = cfg.hidden).
    let t = Instant::now();
    let (mut q, mut k, mut v) = if use_vnni_for_matmul(cfg.hidden) {
        let x_norm_q8 = quantize_activation_q8(&x_norm);
        let q =
            weight_matmul_jit_storage_q8act(&x_norm_q8, &layer.wq, cfg.hidden, cfg.q_dim(), jit);
        let k =
            weight_matmul_jit_storage_q8act(&x_norm_q8, &layer.wk, cfg.hidden, cfg.kv_dim(), jit);
        let v =
            weight_matmul_jit_storage_q8act(&x_norm_q8, &layer.wv, cfg.hidden, cfg.kv_dim(), jit);
        (q, k, v)
    } else {
        let q = weight_matmul_jit_storage(&x_norm, &layer.wq, cfg.hidden, cfg.q_dim(), jit);
        let k = weight_matmul_jit_storage(&x_norm, &layer.wk, cfg.hidden, cfg.kv_dim(), jit);
        let v = weight_matmul_jit_storage(&x_norm, &layer.wv, cfg.hidden, cfg.kv_dim(), jit);
        (q, k, v)
    };
    timer.layer_qkv_matmul += t.elapsed();

    let t = Instant::now();
    if let Some(b) = &layer.b_q {
        add_bias_broadcast(&mut q, b, cfg.q_dim());
    }
    if let Some(b) = &layer.b_k {
        add_bias_broadcast(&mut k, b, cfg.kv_dim());
    }
    if let Some(b) = &layer.b_v {
        add_bias_broadcast(&mut v, b, cfg.kv_dim());
    }
    timer.layer_qkv_bias += t.elapsed();

    // 3. RoPE
    let t = Instant::now();
    rope_in_place(
        &mut q,
        &[position],
        cfg.n_heads,
        cfg.head_dim,
        cfg.rope_base,
    );
    rope_in_place(
        &mut k,
        &[position],
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.rope_base,
    );
    timer.layer_rope += t.elapsed();

    // 4. Append + attention.
    let t = Instant::now();
    cache.append(&k, &v);
    timer.layer_kv_append += t.elapsed();

    let t = Instant::now();
    let attn = attention_decode(&q, cache, cfg);
    timer.layer_attention += t.elapsed();

    // 5. output projection (K = cfg.q_dim).
    let t = Instant::now();
    let attn_out = if use_vnni_for_matmul(cfg.q_dim()) {
        let attn_q8 = quantize_activation_q8(&attn);
        weight_matmul_jit_storage_q8act(&attn_q8, &layer.wo, cfg.q_dim(), cfg.hidden, jit)
    } else {
        weight_matmul_jit_storage(&attn, &layer.wo, cfg.q_dim(), cfg.hidden, jit)
    };
    timer.layer_wo_matmul += t.elapsed();

    // 6. residual.
    let t = Instant::now();
    let mut resid: Vec<f32> = x_t
        .iter()
        .zip(attn_out.iter())
        .map(|(a, b)| a + b)
        .collect();
    timer.layer_attn_residual += t.elapsed();

    // 7. FFN RMSNorm.
    let t = Instant::now();
    let mut ffn_in = vec![0.0f32; cfg.hidden];
    rms_norm(
        &resid,
        &layer.ffn_norm_w,
        &mut ffn_in,
        cfg.hidden,
        cfg.rms_norm_eps,
    );
    timer.layer_ffn_rms += t.elapsed();

    // 8. gate/up (K = cfg.hidden).
    let t = Instant::now();
    let (mut gate, up) = if use_vnni_for_matmul(cfg.hidden) {
        let ffn_in_q8 = quantize_activation_q8(&ffn_in);
        let g = weight_matmul_jit_storage_q8act(
            &ffn_in_q8,
            &layer.w_gate,
            cfg.hidden,
            cfg.ffn_hidden,
            jit,
        );
        let u = weight_matmul_jit_storage_q8act(
            &ffn_in_q8,
            &layer.w_up,
            cfg.hidden,
            cfg.ffn_hidden,
            jit,
        );
        (g, u)
    } else {
        let g = weight_matmul_jit_storage(&ffn_in, &layer.w_gate, cfg.hidden, cfg.ffn_hidden, jit);
        let u = weight_matmul_jit_storage(&ffn_in, &layer.w_up, cfg.hidden, cfg.ffn_hidden, jit);
        (g, u)
    };
    timer.layer_gate_up_matmul += t.elapsed();

    // 9. SiLU(gate) * up.
    let t = Instant::now();
    silu_in_place(&mut gate);
    mul_in_place(&mut gate, &up);
    timer.layer_silu_mul += t.elapsed();

    // 10. down (K = cfg.ffn_hidden; long → routes to fp32 4-acc).
    let t = Instant::now();
    let ffn_out = if use_vnni_for_matmul(cfg.ffn_hidden) {
        let gate_q8 = quantize_activation_q8(&gate);
        weight_matmul_jit_storage_q8act(&gate_q8, &layer.w_down, cfg.ffn_hidden, cfg.hidden, jit)
    } else {
        weight_matmul_jit_storage(&gate, &layer.w_down, cfg.ffn_hidden, cfg.hidden, jit)
    };
    timer.layer_down_matmul += t.elapsed();

    // 11. residual.
    let t = Instant::now();
    for (r, o) in resid.iter_mut().zip(ffn_out.iter()) {
        *r += *o;
    }
    timer.layer_ffn_residual += t.elapsed();

    resid
}

/// Top-level model hyperparameters: a `LayerConfig` plus vocab and depth.
#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub layer: LayerConfig,
    pub vocab_size: usize,
    pub n_layers: usize,
    pub max_seq: usize,
    pub eos_token_id: Option<u32>,
}

impl ModelConfig {
    pub fn hidden(&self) -> usize {
        self.layer.hidden
    }
}

/// A full transformer ready to generate.
///
/// Weight layout, all row-major:
/// - `token_embeddings`: `[vocab_size, hidden]`
/// - `lm_head_w`: `[vocab_size, hidden]` (weight-tied with embeddings is a
///   common option; we keep them separate here so toy tests can vary them
///   independently)
/// - `final_norm_w`: `[hidden]`
/// - `layers[i]`: standard `LayerWeights`
#[derive(Clone, Debug)]
pub struct Model {
    pub config: ModelConfig,
    pub token_embeddings: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub final_norm_w: Vec<f32>,
    pub lm_head_w: WeightStorage,
}

impl Model {
    /// Look up one token's embedding row.
    fn embed_token(&self, token: u32) -> Vec<f32> {
        let h = self.config.hidden();
        let off = token as usize * h;
        self.token_embeddings[off..off + h].to_vec()
    }

    /// Run one decode step through every layer, then `final_norm` + `lm_head`.
    /// Returns logits over the vocabulary.
    pub fn forward_decode(&self, token: u32, position: u32, cache: &mut KvCache) -> Vec<f32> {
        assert_eq!(cache.n_layers(), self.config.n_layers);
        let mut x = self.embed_token(token);
        for (i, layer) in self.layers.iter().enumerate() {
            x = forward_layer_decode(&x, position, layer, cache.layer_mut(i), &self.config.layer);
        }
        // Final RMSNorm
        let mut x_norm = vec![0.0f32; self.config.hidden()];
        rms_norm(
            &x,
            &self.final_norm_w,
            &mut x_norm,
            self.config.hidden(),
            self.config.layer.rms_norm_eps,
        );
        // lm_head projection: [1, hidden] @ [vocab, hidden]^T → [vocab]
        let lm_head_native = self
            .lm_head_w
            .as_f32_native(self.config.vocab_size, self.config.hidden());
        weight_matmul(
            &x_norm,
            &lm_head_native,
            1,
            self.config.hidden(),
            self.config.vocab_size,
        )
    }

    /// Transpose every F32 weight matrix in the model so the JIT matmul
    /// kernels can consume them. Q8 weights are left native (the fused
    /// Q8×F32 kernel already reads `[d_out, d_in]`). After this, the F32
    /// weights are *only* usable via the `_jit` variants; calling
    /// `forward_decode` / `generate_greedy` on a transposed F32 model
    /// produces garbage.
    pub fn transpose_for_jit(&mut self) {
        for layer in &mut self.layers {
            layer.transpose_in_place(&self.config.layer);
        }
        transpose_storage_in_place(
            &mut self.lm_head_w,
            self.config.vocab_size,
            self.config.hidden(),
        );
    }

    /// JIT variant of [`Self::forward_decode`]. Caller must have called
    /// [`Self::transpose_for_jit`] first.
    pub fn forward_decode_jit(
        &self,
        token: u32,
        position: u32,
        cache: &mut KvCache,
        jit: &mut MatmulJitCache,
    ) -> Vec<f32> {
        assert_eq!(cache.n_layers(), self.config.n_layers);
        let mut x = self.embed_token(token);
        for (i, layer) in self.layers.iter().enumerate() {
            x = forward_layer_decode_jit(
                &x,
                position,
                layer,
                cache.layer_mut(i),
                &self.config.layer,
                jit,
            );
        }
        // Final RMSNorm
        let mut x_norm = vec![0.0f32; self.config.hidden()];
        rms_norm(
            &x,
            &self.final_norm_w,
            &mut x_norm,
            self.config.hidden(),
            self.config.layer.rms_norm_eps,
        );
        // lm_head projection through JIT. K = hidden (short → usually VNNI).
        if use_vnni_for_matmul(self.config.hidden()) {
            let x_norm_q8 = quantize_activation_q8(&x_norm);
            weight_matmul_jit_storage_q8act(
                &x_norm_q8,
                &self.lm_head_w,
                self.config.hidden(),
                self.config.vocab_size,
                jit,
            )
        } else {
            weight_matmul_jit_storage(
                &x_norm,
                &self.lm_head_w,
                self.config.hidden(),
                self.config.vocab_size,
                jit,
            )
        }
    }

    /// Profiled variant of [`Self::forward_decode_jit`] — adds each step's
    /// elapsed time to `timer`. Phase 7.E.0 only; not on the production path.
    pub fn forward_decode_jit_timed(
        &self,
        token: u32,
        position: u32,
        cache: &mut KvCache,
        jit: &mut MatmulJitCache,
        timer: &mut StepTimer,
    ) -> Vec<f32> {
        use std::time::Instant;
        assert_eq!(cache.n_layers(), self.config.n_layers);

        let t = Instant::now();
        let mut x = self.embed_token(token);
        timer.embed += t.elapsed();

        for (i, layer) in self.layers.iter().enumerate() {
            x = forward_layer_decode_jit_timed(
                &x,
                position,
                layer,
                cache.layer_mut(i),
                &self.config.layer,
                jit,
                timer,
            );
        }

        let t = Instant::now();
        let mut x_norm = vec![0.0f32; self.config.hidden()];
        rms_norm(
            &x,
            &self.final_norm_w,
            &mut x_norm,
            self.config.hidden(),
            self.config.layer.rms_norm_eps,
        );
        timer.final_rms += t.elapsed();

        let t = Instant::now();
        let logits = if use_vnni_for_matmul(self.config.hidden()) {
            let x_norm_q8 = quantize_activation_q8(&x_norm);
            weight_matmul_jit_storage_q8act(
                &x_norm_q8,
                &self.lm_head_w,
                self.config.hidden(),
                self.config.vocab_size,
                jit,
            )
        } else {
            weight_matmul_jit_storage(
                &x_norm,
                &self.lm_head_w,
                self.config.hidden(),
                self.config.vocab_size,
                jit,
            )
        };
        timer.lm_head_matmul += t.elapsed();
        logits
    }

    /// Profiled variant of [`Self::generate_greedy_jit`]: returns both the
    /// generated tokens and the per-step time accumulator.
    pub fn generate_greedy_jit_timed(
        &self,
        prompt: &[u32],
        max_new: usize,
    ) -> (Vec<u32>, StepTimer) {
        assert!(!prompt.is_empty());
        assert!(prompt.len() + max_new <= self.config.max_seq);

        let mut cache = KvCache::new(
            self.config.n_layers,
            prompt.len() + max_new,
            self.config.layer.kv_dim(),
        );
        let mut jit = MatmulJitCache::new();
        let mut timer = StepTimer::new();

        let mut last_logits = Vec::new();
        for (pos, &tok) in prompt.iter().enumerate() {
            last_logits =
                self.forward_decode_jit_timed(tok, pos as u32, &mut cache, &mut jit, &mut timer);
        }

        let mut out = Vec::with_capacity(max_new);
        for i in 0..max_new {
            let next = argmax(&last_logits) as u32;
            out.push(next);
            if Some(next) == self.config.eos_token_id {
                break;
            }
            let next_pos = (prompt.len() + i) as u32;
            last_logits =
                self.forward_decode_jit_timed(next, next_pos, &mut cache, &mut jit, &mut timer);
        }
        (out, timer)
    }

    /// JIT variant of [`Self::generate_greedy`]. Caller must have called
    /// [`Self::transpose_for_jit`] first.
    pub fn generate_greedy_jit(&self, prompt: &[u32], max_new: usize) -> Vec<u32> {
        assert!(!prompt.is_empty());
        assert!(prompt.len() + max_new <= self.config.max_seq);

        let mut cache = KvCache::new(
            self.config.n_layers,
            prompt.len() + max_new,
            self.config.layer.kv_dim(),
        );
        let mut jit = MatmulJitCache::new();

        let mut last_logits = Vec::new();
        for (pos, &tok) in prompt.iter().enumerate() {
            last_logits = self.forward_decode_jit(tok, pos as u32, &mut cache, &mut jit);
        }

        let mut out = Vec::with_capacity(max_new);
        for i in 0..max_new {
            let next = argmax(&last_logits) as u32;
            out.push(next);
            if Some(next) == self.config.eos_token_id {
                break;
            }
            let next_pos = (prompt.len() + i) as u32;
            last_logits = self.forward_decode_jit(next, next_pos, &mut cache, &mut jit);
        }
        out
    }

    /// Generate up to `max_new` tokens. Greedy sampling (argmax of logits).
    /// Stops early on EOS if `eos_token_id` is set.
    ///
    /// Returns *only the newly generated* tokens (not the prompt).
    pub fn generate_greedy(&self, prompt: &[u32], max_new: usize) -> Vec<u32> {
        assert!(!prompt.is_empty(), "prompt must contain at least one token");
        assert!(
            prompt.len() + max_new <= self.config.max_seq,
            "prompt+max_new ({}) exceeds max_seq ({})",
            prompt.len() + max_new,
            self.config.max_seq,
        );

        // Size the cache to exactly what this run needs, not the model's
        // declared `max_seq` (Qwen2 ships max_seq = 32K which would allocate
        // hundreds of MB for nothing).
        let needed = prompt.len() + max_new;
        let mut cache = KvCache::new(self.config.n_layers, needed, self.config.layer.kv_dim());

        // Prefill: feed each prompt token through the decode path so the cache
        // is populated. The last call's logits become our first predictor.
        let mut last_logits = Vec::new();
        for (pos, &tok) in prompt.iter().enumerate() {
            last_logits = self.forward_decode(tok, pos as u32, &mut cache);
        }

        let mut out = Vec::with_capacity(max_new);
        for i in 0..max_new {
            let next = argmax(&last_logits) as u32;
            out.push(next);
            if Some(next) == self.config.eos_token_id {
                break;
            }
            let next_pos = (prompt.len() + i) as u32;
            last_logits = self.forward_decode(next, next_pos, &mut cache);
        }
        out
    }
}

fn argmax(v: &[f32]) -> usize {
    debug_assert!(!v.is_empty());
    let mut best_i = 0usize;
    let mut best_v = v[0];
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x > best_v {
            best_v = x;
            best_i = i;
        }
    }
    best_i
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
        let mks = |buf: Vec<f32>| WeightStorage::F32(buf);
        LayerWeights {
            attn_norm_w: vec![1.0; h],
            wq: mks(mk(qd * h)),
            wk: mks(mk(kvd * h)),
            wv: mks(mk(kvd * h)),
            wo: mks(mk(h * qd)),
            ffn_norm_w: vec![1.0; h],
            w_gate: mks(mk(ff * h)),
            w_up: mks(mk(ff * h)),
            w_down: mks(mk(h * ff)),
            b_q: None,
            b_k: None,
            b_v: None,
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

    /// The decode path must match the prefill path for the *same* token
    /// sequence: prefill on [t0, t1, t2] should give the same last-row output
    /// as prefill on [t0, t1] followed by decode on t2.
    ///
    /// This is the property that makes autoregressive generation correct.
    #[test]
    fn decode_matches_prefill() {
        let cfg = toy_cfg();
        let layer = random_weights(&cfg, 13);
        let h = cfg.hidden;

        let x0 = vec![0.3f32; h];
        let x1 = vec![0.7f32; h];
        let x2 = vec![-0.4f32; h];

        // Path A: prefill on all three tokens.
        let mut all = Vec::with_capacity(3 * h);
        all.extend_from_slice(&x0);
        all.extend_from_slice(&x1);
        all.extend_from_slice(&x2);
        let prefill_out = forward_layer(&all, 3, &layer, &[0, 1, 2], &cfg);
        let last_row_prefill = &prefill_out[2 * h..3 * h];

        // Path B: prefill on first two tokens (just to populate the cache),
        // then decode the third.
        let mut cache = LayerKvCache::new(8, cfg.kv_dim());

        // We need a decode-path warm-up: decode each of t0 and t1 to fill the
        // cache. The decode path is logically equivalent to prefill-of-1 for
        // each position when run sequentially.
        let _ = forward_layer_decode(&x0, 0, &layer, &mut cache, &cfg);
        let _ = forward_layer_decode(&x1, 1, &layer, &mut cache, &cfg);
        let decode_out = forward_layer_decode(&x2, 2, &layer, &mut cache, &cfg);

        for i in 0..h {
            let a = last_row_prefill[i];
            let b = decode_out[i];
            assert!(
                (a - b).abs() < 1e-4,
                "decode/prefill diverge at idx {}: prefill={} decode={}",
                i,
                a,
                b
            );
        }
        // Cache holds 3 K/V rows.
        assert_eq!(cache.len(), 3);
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

    // ------------- Model / generate tests -----------------------------------

    fn toy_model(seed: u64, vocab: usize, n_layers: usize) -> Model {
        let layer_cfg = toy_cfg();
        let cfg = ModelConfig {
            layer: layer_cfg.clone(),
            vocab_size: vocab,
            n_layers,
            max_seq: 32,
            eos_token_id: None,
        };
        let mut s = seed | 1;
        let mut next = || -> f32 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s as i32 as f32) * 1e-10
        };
        let h = cfg.hidden();
        Model {
            token_embeddings: (0..vocab * h).map(|_| next()).collect(),
            layers: (0..n_layers)
                .map(|i| random_weights(&layer_cfg, seed + i as u64))
                .collect(),
            final_norm_w: vec![1.0; h],
            lm_head_w: WeightStorage::F32((0..vocab * h).map(|_| next()).collect()),
            config: cfg,
        }
    }

    #[test]
    fn generate_produces_expected_length() {
        let model = toy_model(31, 64, 2);
        let out = model.generate_greedy(&[5, 10, 15], 7);
        assert_eq!(out.len(), 7);
        for &t in &out {
            assert!((t as usize) < model.config.vocab_size);
        }
    }

    #[test]
    fn generate_is_deterministic() {
        let model = toy_model(73, 64, 2);
        let a = model.generate_greedy(&[1, 2, 3], 8);
        let b = model.generate_greedy(&[1, 2, 3], 8);
        assert_eq!(a, b);
    }

    /// Generated sequence must vary when the prompt varies (sanity check that
    /// the model is actually reading its input rather than ignoring it).
    #[test]
    fn generate_responds_to_prompt() {
        let model = toy_model(101, 64, 2);
        let a = model.generate_greedy(&[5], 5);
        let b = model.generate_greedy(&[42], 5);
        assert_ne!(a, b);
    }

    #[test]
    fn eos_stops_generation_early() {
        let mut model = toy_model(7, 64, 2);
        // Probe which token would come first for prompt = [0]; force it as EOS.
        let probe = model.generate_greedy(&[0], 1);
        model.config.eos_token_id = Some(probe[0]);
        let out = model.generate_greedy(&[0], 20);
        // Should stop at the first generated token (the one that equals EOS).
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], probe[0]);
    }
}
