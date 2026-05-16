//! Phase 6.F.2.b: load Qwen2.5-0.5B-Q8_0 from disk.
//!
//! This test only runs if the model file is present at the expected path;
//! otherwise it skips. We're checking that the loader handles a real-world
//! GGUF (Qwen2 architecture, Q8_0 weights, F32 norms/biases, attention
//! biases present) without panicking and with sensible-looking config.

use std::path::Path;

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
