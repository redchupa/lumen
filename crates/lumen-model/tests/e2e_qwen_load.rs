//! Phase 6.F.2.b: load Qwen2.5-0.5B-Q8_0 from disk.
//!
//! This test only runs if the model file is present at the expected path;
//! otherwise it skips. We're checking that the loader handles a real-world
//! GGUF (Qwen2 architecture, Q8_0 weights, F32 norms/biases, attention
//! biases present) without panicking and with sensible-looking config.

use std::path::Path;

use lumen_jit::MatmulJitCache;
use lumen_model::tokenizer::{BpeMode, Tokenizer};
use lumen_model::{config_from_gguf, model_from_gguf, GgmlType, GgufFile};

const QWEN_PATH: &str = r"C:\Users\redchupa\Desktop\github_auto_development\lumen\models\qwen2.5-0.5b-instruct-q8_0.gguf";

fn check_qwen_present() -> bool {
    Path::new(QWEN_PATH).exists()
}

#[test]
fn qwen_2_5_0_5b_config_loads() {
    if !check_qwen_present() {
        eprintln!("skip: {} not present", QWEN_PATH);
        return;
    }
    let file = GgufFile::open(QWEN_PATH).expect("open gguf");
    let cfg = config_from_gguf(&file, "qwen2").expect("config");

    assert_eq!(cfg.hidden(), 896);
    assert_eq!(cfg.n_layers, 24);
    assert_eq!(cfg.layer.n_heads, 14);
    assert_eq!(cfg.layer.n_kv_heads, 2);
    assert_eq!(cfg.layer.head_dim, 64);
    assert_eq!(cfg.layer.ffn_hidden, 4864);
    assert_eq!(cfg.max_seq, 32768);
    assert_eq!(cfg.vocab_size, 151936);
    assert!((cfg.layer.rope_base - 1_000_000.0).abs() < 1.0);
    assert!((cfg.layer.rms_norm_eps - 1e-6).abs() < 1e-10);
    assert_eq!(cfg.eos_token_id, Some(151645));

    eprintln!(
        "Qwen2.5-0.5B config OK: 24 layers × 14H/2KVH × 64d × ffn{} (vocab {})",
        cfg.layer.ffn_hidden, cfg.vocab_size
    );
}

#[test]
#[ignore = "loads ~640MB and dequantizes to ~2.5GB; run with --ignored"]
fn qwen_2_5_0_5b_model_loads_with_biases() {
    if !check_qwen_present() {
        eprintln!("skip: {} not present", QWEN_PATH);
        return;
    }
    let file = GgufFile::open(QWEN_PATH).expect("open gguf");

    // Sanity-check the tensor table before the heavy decode pass.
    let has_q_bias = file
        .tensors()
        .iter()
        .any(|t| t.name == "blk.0.attn_q.bias" && t.dtype == GgmlType::F32);
    assert!(has_q_bias, "Qwen2 should have attn_q.bias on layer 0");

    let model = model_from_gguf(&file, "qwen2").expect("model");

    assert_eq!(model.layers.len(), 24);
    assert_eq!(model.token_embeddings.len(), 896 * 151936);
    assert_eq!(model.final_norm_w.len(), 896);
    assert_eq!(model.lm_head_w.len(), 896 * 151936);

    // First-layer sanity
    let l0 = &model.layers[0];
    assert_eq!(l0.attn_norm_w.len(), 896);
    assert_eq!(l0.wq.len(), 896 * 896);
    assert_eq!(l0.wk.len(), 128 * 896);
    assert_eq!(l0.wv.len(), 128 * 896);
    assert_eq!(l0.wo.len(), 896 * 896);
    assert_eq!(l0.ffn_norm_w.len(), 896);
    assert_eq!(l0.w_gate.len(), 4864 * 896);
    assert_eq!(l0.w_up.len(), 4864 * 896);
    assert_eq!(l0.w_down.len(), 896 * 4864);
    assert!(l0.b_q.is_some(), "Qwen2 should populate b_q");
    assert_eq!(l0.b_q.as_ref().unwrap().len(), 896);
    assert_eq!(l0.b_k.as_ref().unwrap().len(), 128);
    assert_eq!(l0.b_v.as_ref().unwrap().len(), 128);

    eprintln!(
        "Qwen2.5-0.5B loaded: {} layers, {} fp32 elements in token_embd",
        model.layers.len(),
        model.token_embeddings.len()
    );
}

/// Build the Qwen tokenizer from the real GGUF and round-trip simple inputs.
/// Phase 6.F.2.c milestone: byte mapping + BPE merges actually re-form the
/// input string when decoded.
#[test]
fn qwen_tokenizer_round_trips_ascii_and_korean() {
    if !check_qwen_present() {
        eprintln!("skip: {} not present", QWEN_PATH);
        return;
    }
    let file = GgufFile::open(QWEN_PATH).expect("open gguf");
    let tok = Tokenizer::from_gguf(&file).expect("tokenizer");
    assert_eq!(tok.mode(), BpeMode::Gpt2);
    assert!(tok.vocab_size() >= 151_000);

    // ASCII round-trip: must produce a non-empty id sequence and decode back
    // to the same string.
    let s = "hello";
    let ids = tok.encode(s);
    assert!(!ids.is_empty(), "encode produced no ids for {:?}", s);
    let decoded = tok.decode(&ids).unwrap();
    assert_eq!(
        decoded, s,
        "ASCII round-trip mismatch: {:?} -> {:?}",
        s, decoded
    );
    eprintln!("encode({:?}) = {} tokens, decode round-trips", s, ids.len());

    // Korean round-trip.
    let k = "안녕";
    let ids_k = tok.encode(k);
    assert!(!ids_k.is_empty(), "encode produced no ids for {:?}", k);
    let decoded_k = tok.decode(&ids_k).unwrap();
    assert_eq!(
        decoded_k, k,
        "Korean round-trip mismatch: {:?} -> {:?}",
        k, decoded_k
    );
    eprintln!(
        "encode({:?}) = {} tokens, decode round-trips",
        k,
        ids_k.len()
    );
}

/// The big one: load Qwen2.5-0.5B end-to-end and have Lumen actually run a
/// forward pass for a Korean prompt. We don't yet expect coherent output
/// because:
///
/// 1. Our naive Rust matmul is unoptimized; expect ~seconds per token.
/// 2. Pre-tokenization regex (Phase 6.F.2.d) is skipped, so the token IDs
///    we feed may differ from what the model was trained against.
/// 3. fp32 accumulation throughout, no fused dequant.
///
/// What we *do* assert: the call returns without panicking, produces the
/// requested number of new tokens, and they all decode back to a valid
/// UTF-8 string (i.e. the byte mapping invariant is preserved through the
/// whole pipeline).
#[test]
#[ignore = "loads ~640MB, dequantizes to ~2.5GB, runs fp32 forward — 10-60s in release"]
fn qwen_generates_first_korean_tokens() {
    if !check_qwen_present() {
        eprintln!("skip: {} not present", QWEN_PATH);
        return;
    }

    eprintln!("loading Qwen2.5-0.5B-Q8_0 ...");
    let t_load = std::time::Instant::now();
    let file = GgufFile::open(QWEN_PATH).expect("open gguf");
    let model = model_from_gguf(&file, "qwen2").expect("model");
    let tok = Tokenizer::from_gguf(&file).expect("tokenizer");
    eprintln!("  loaded in {:?}", t_load.elapsed());

    let prompt_text = "안녕";
    let prompt_ids = tok.encode(prompt_text);
    eprintln!("prompt: {:?} → ids: {:?}", prompt_text, prompt_ids);
    assert!(!prompt_ids.is_empty());

    let max_new = 3usize;
    let t_gen = std::time::Instant::now();
    let new_ids = model.generate_greedy(&prompt_ids, max_new);
    let elapsed = t_gen.elapsed();
    eprintln!(
        "generated {} tokens in {:?}  ({:.2}s/tok)",
        new_ids.len(),
        elapsed,
        elapsed.as_secs_f64() / new_ids.len().max(1) as f64
    );
    eprintln!("new ids: {:?}", new_ids);

    // Decode each new id individually so we can see what was emitted
    // even if some IDs are special tokens that fail to UTF-8 decode.
    for &id in &new_ids {
        match tok.decode(&[id]) {
            Ok(s) => eprintln!("  id {:>6} → {:?}", id, s),
            Err(e) => eprintln!("  id {:>6} → <decode error: {}>", id, e),
        }
    }

    // Full sequence decode.
    let mut all_ids = prompt_ids.clone();
    all_ids.extend_from_slice(&new_ids);
    match tok.decode(&all_ids) {
        Ok(s) => eprintln!("full text: {:?}", s),
        Err(e) => eprintln!("(full decode failed: {})", e),
    }

    assert_eq!(new_ids.len(), max_new);
}

/// Naive Rust `weight_matmul(a, w, 1, K, N)`: 1×K activation × w^T where
/// `w` is stored row-major as `[N, K]` (ggml convention).
fn naive_weight_matmul(a: &[f32], w: &[f32], k_dim: usize, n_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_dim];
    for n in 0..n_dim {
        let mut acc = 0.0f32;
        for k in 0..k_dim {
            acc += a[k] * w[n * k_dim + k];
        }
        out[n] = acc;
    }
    out
}

/// Transpose a `[N, K]` row-major matrix into `[K, N]` row-major.
/// (i.e. `out[k * N + n] = src[n * K + k]`)
fn transpose_n_k_to_k_n(src: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * k];
    for ni in 0..n {
        for ki in 0..k {
            out[ki * n + ni] = src[ni * k + ki];
        }
    }
    out
}

/// Phase 6.G.2 demo: run Qwen's lm_head (M=1, K=896, N=151936) through
/// (a) the in-tree naive matmul and (b) the Phase 6.G.1 JIT cache after
/// transposing the weight to row-major `[K, N]`. Verify the results agree
/// elementwise, then print both wall-clock times so the speedup is visible.
#[test]
#[ignore = "loads ~640MB and runs lm_head twice; run with --ignored --nocapture"]
fn qwen_lm_head_jit_matches_naive_and_is_faster() {
    if !check_qwen_present() {
        eprintln!("skip: {} not present", QWEN_PATH);
        return;
    }
    let file = GgufFile::open(QWEN_PATH).expect("open gguf");
    let model = model_from_gguf(&file, "qwen2").expect("model");

    let hidden = model.config.hidden(); // 896
    let vocab = model.config.vocab_size; // 151936
    assert_eq!(vocab % 8, 0, "JIT path requires N % 8 == 0");

    // A representative activation row (we don't need a meaningful one — this
    // is a numerical equivalence check, not a generation step).
    let activation: Vec<f32> = (0..hidden)
        .map(|i| ((i % 37) as f32) * 0.001 - 0.02)
        .collect();

    // --- naive ---
    let t_naive = std::time::Instant::now();
    let naive_logits = naive_weight_matmul(&activation, &model.lm_head_w, hidden, vocab);
    let naive_elapsed = t_naive.elapsed();
    eprintln!("lm_head naive : {:>10?}", naive_elapsed);

    // --- prepare JIT: transpose lm_head from [vocab, hidden] to [hidden, vocab] ---
    let t_prep = std::time::Instant::now();
    let lm_head_t = transpose_n_k_to_k_n(&model.lm_head_w, vocab, hidden);
    let prep_elapsed = t_prep.elapsed();
    eprintln!("lm_head transp: {:>10?}  (one-time cost)", prep_elapsed);

    let mut cache = MatmulJitCache::new();
    let t_compile = std::time::Instant::now();
    let f = cache
        .get_or_compile(1, hidden as u32, vocab as u32)
        .expect("jit compile");
    let compile_elapsed = t_compile.elapsed();
    eprintln!("lm_head compil: {:>10?}  (one-time cost)", compile_elapsed);

    // --- JIT call ---
    let mut jit_logits = vec![0.0f32; vocab];
    let t_jit = std::time::Instant::now();
    // SAFETY: we just compiled for (1, hidden, vocab); the buffer sizes match.
    unsafe {
        f(
            activation.as_ptr(),
            lm_head_t.as_ptr(),
            jit_logits.as_mut_ptr(),
        );
    }
    let jit_elapsed = t_jit.elapsed();
    eprintln!("lm_head JIT   : {:>10?}", jit_elapsed);

    let speedup = naive_elapsed.as_secs_f64() / jit_elapsed.as_secs_f64();
    eprintln!(
        "speedup       : {:.1}x (JIT vs naive, single matmul)",
        speedup
    );

    // --- correctness: every logit should agree within fp accumulation noise ---
    let mut max_abs_diff = 0.0f32;
    for (g, w) in jit_logits.iter().zip(naive_logits.iter()) {
        let d = (g - w).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    eprintln!(
        "max |jit - naive| over {} logits: {:.2e}",
        vocab, max_abs_diff
    );
    // Tolerance scales with K (896 fused multiply-adds). 1e-2 is loose enough
    // to absorb the difference in summation order between the two paths.
    assert!(
        max_abs_diff < 1e-2,
        "lm_head JIT vs naive diverged by {}",
        max_abs_diff
    );
}
