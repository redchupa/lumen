//! Phase 6.F.2.b: load Qwen2.5-0.5B-Q8_0 from disk.
//!
//! This test only runs if the model file is present at the expected path;
//! otherwise it skips. We're checking that the loader handles a real-world
//! GGUF (Qwen2 architecture, Q8_0 weights, F32 norms/biases, attention
//! biases present) without panicking and with sensible-looking config.

use std::path::Path;

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
