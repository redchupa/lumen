//! Phase 6.F.1 — the first true end-to-end generation pipeline.
//!
//! Build a small random F32 transformer in memory, serialize it to a GGUF
//! buffer using the ggml tensor-naming convention, load it back through
//! `lumen_model::model_from_gguf`, and verify that `generate_greedy` produces
//! exactly the same token sequence as the original in-memory model.
//!
//! This proves the file-on-disk → model-in-memory → inference round-trip
//! works for a Lumen-only stack. Quantized weights and a real Qwen model
//! follow in Phase 6.F.2.

use lumen_model::{config_from_gguf, gguf::KvType, model_from_gguf, GgmlType, GgufFile};
use lumen_runtime::{LayerConfig, LayerWeights, Model, ModelConfig};

// ----- minimal GGUF writer (test helper) ------------------------------------

fn w_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn w_kv_u32(buf: &mut Vec<u8>, key: &str, v: u32) {
    w_string(buf, key);
    buf.extend_from_slice(&(KvType::U32 as u32).to_le_bytes());
    buf.extend_from_slice(&v.to_le_bytes());
}

fn w_kv_f32(buf: &mut Vec<u8>, key: &str, v: f32) {
    w_string(buf, key);
    buf.extend_from_slice(&(KvType::F32 as u32).to_le_bytes());
    buf.extend_from_slice(&v.to_le_bytes());
}

fn w_kv_string(buf: &mut Vec<u8>, key: &str, v: &str) {
    w_string(buf, key);
    buf.extend_from_slice(&(KvType::String as u32).to_le_bytes());
    w_string(buf, v);
}

/// Append a tensor info record. Returns the offset (within the data section
/// once it begins) at which this tensor's bytes start.
fn w_tensor_info(buf: &mut Vec<u8>, name: &str, dims: &[u64], offset: u64) {
    w_string(buf, name);
    buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for d in dims {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    buf.extend_from_slice(&(GgmlType::F32 as u32).to_le_bytes());
    buf.extend_from_slice(&offset.to_le_bytes());
}

/// Serialize a small F32 model into a GGUF buffer.
fn write_toy_gguf(arch: &str, model: &Model) -> Vec<u8> {
    let cfg = &model.config;
    let lcfg = &cfg.layer;
    let h = lcfg.hidden;
    let qd = lcfg.q_dim();
    let kvd = lcfg.kv_dim();
    let ff = lcfg.ffn_hidden;

    // Build the list of tensors we'll write, in a stable order with their
    // shapes and source bytes.
    let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
    tensors.push((
        "token_embd.weight".into(),
        vec![h as u64, cfg.vocab_size as u64],
        model.token_embeddings.clone(),
    ));
    tensors.push((
        "output_norm.weight".into(),
        vec![h as u64],
        model.final_norm_w.clone(),
    ));
    tensors.push((
        "output.weight".into(),
        vec![h as u64, cfg.vocab_size as u64],
        model.lm_head_w.clone(),
    ));
    for (i, layer) in model.layers.iter().enumerate() {
        let p = format!("blk.{}", i);
        tensors.push((
            format!("{}.attn_norm.weight", p),
            vec![h as u64],
            layer.attn_norm_w.clone(),
        ));
        tensors.push((
            format!("{}.attn_q.weight", p),
            vec![h as u64, qd as u64],
            layer.wq.clone(),
        ));
        tensors.push((
            format!("{}.attn_k.weight", p),
            vec![h as u64, kvd as u64],
            layer.wk.clone(),
        ));
        tensors.push((
            format!("{}.attn_v.weight", p),
            vec![h as u64, kvd as u64],
            layer.wv.clone(),
        ));
        tensors.push((
            format!("{}.attn_output.weight", p),
            vec![qd as u64, h as u64],
            layer.wo.clone(),
        ));
        tensors.push((
            format!("{}.ffn_norm.weight", p),
            vec![h as u64],
            layer.ffn_norm_w.clone(),
        ));
        tensors.push((
            format!("{}.ffn_gate.weight", p),
            vec![h as u64, ff as u64],
            layer.w_gate.clone(),
        ));
        tensors.push((
            format!("{}.ffn_up.weight", p),
            vec![h as u64, ff as u64],
            layer.w_up.clone(),
        ));
        tensors.push((
            format!("{}.ffn_down.weight", p),
            vec![ff as u64, h as u64],
            layer.w_down.clone(),
        ));
    }

    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes()); // version
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes()); // n_tensors
                                                                  // n_kv counted after we list keys: alignment + general.architecture + arch.* (7 fields)
    let n_kv: u64 = 1 + 1 + 7;
    buf.extend_from_slice(&n_kv.to_le_bytes());

    // KV pairs
    w_kv_u32(&mut buf, "general.alignment", 32);
    w_kv_string(&mut buf, "general.architecture", arch);
    w_kv_u32(&mut buf, &format!("{}.embedding_length", arch), h as u32);
    w_kv_u32(
        &mut buf,
        &format!("{}.block_count", arch),
        cfg.n_layers as u32,
    );
    w_kv_u32(
        &mut buf,
        &format!("{}.attention.head_count", arch),
        lcfg.n_heads as u32,
    );
    w_kv_u32(
        &mut buf,
        &format!("{}.attention.head_count_kv", arch),
        lcfg.n_kv_heads as u32,
    );
    w_kv_u32(
        &mut buf,
        &format!("{}.feed_forward_length", arch),
        ff as u32,
    );
    w_kv_u32(
        &mut buf,
        &format!("{}.context_length", arch),
        cfg.max_seq as u32,
    );
    w_kv_f32(
        &mut buf,
        &format!("{}.attention.layer_norm_rms_epsilon", arch),
        lcfg.rms_norm_eps,
    );

    // Tensor info table — first pass to know offsets we need to assign. We
    // assign offsets sequentially in the data section.
    let mut offsets: Vec<u64> = Vec::new();
    let mut running: u64 = 0;
    for (_, _, payload) in &tensors {
        offsets.push(running);
        running += (payload.len() * 4) as u64;
    }
    for ((name, dims, _), off) in tensors.iter().zip(offsets.iter()) {
        w_tensor_info(&mut buf, name, dims, *off);
    }

    // Pad to alignment 32.
    let pad = (32 - (buf.len() % 32)) % 32;
    buf.resize(buf.len() + pad, 0);

    // Data section — each tensor's bytes in the same order.
    for (_, _, payload) in &tensors {
        for &v in payload {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

// ----- random Model builder (no PRNG dep) ----------------------------------

fn small_random_model(seed: u64) -> Model {
    let mut s = seed | 1;
    let mut next = || -> f32 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s as i32 as f32) * 1e-10
    };

    let lcfg = LayerConfig {
        hidden: 8,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 4,
        ffn_hidden: 16,
        rms_norm_eps: 1e-5,
        rope_base: 10000.0,
    };
    let cfg = ModelConfig {
        layer: lcfg.clone(),
        vocab_size: 32,
        n_layers: 2,
        max_seq: 32,
        eos_token_id: None,
    };
    let h = lcfg.hidden;
    let qd = lcfg.q_dim();
    let kvd = lcfg.kv_dim();
    let ff = lcfg.ffn_hidden;
    let mk_vec =
        |gen: &mut dyn FnMut() -> f32, n: usize| -> Vec<f32> { (0..n).map(|_| gen()).collect() };
    let mut layers = Vec::new();
    for _ in 0..cfg.n_layers {
        layers.push(LayerWeights {
            attn_norm_w: vec![1.0; h],
            wq: mk_vec(&mut next, qd * h),
            wk: mk_vec(&mut next, kvd * h),
            wv: mk_vec(&mut next, kvd * h),
            wo: mk_vec(&mut next, h * qd),
            ffn_norm_w: vec![1.0; h],
            w_gate: mk_vec(&mut next, ff * h),
            w_up: mk_vec(&mut next, ff * h),
            w_down: mk_vec(&mut next, h * ff),
            b_q: None,
            b_k: None,
            b_v: None,
        });
    }
    Model {
        token_embeddings: mk_vec(&mut next, cfg.vocab_size * h),
        layers,
        final_norm_w: vec![1.0; h],
        lm_head_w: mk_vec(&mut next, cfg.vocab_size * h),
        config: cfg,
    }
}

#[test]
fn random_model_round_trips_via_gguf_and_generates_same_tokens() {
    let arch = "lumen.toy";
    let original = small_random_model(2026);

    // Serialize -> parse -> rebuild.
    let bytes = write_toy_gguf(arch, &original);
    let file = GgufFile::from_bytes(bytes).expect("parse gguf");

    let parsed_cfg = config_from_gguf(&file, arch).expect("config");
    assert_eq!(parsed_cfg.hidden(), original.config.hidden());
    assert_eq!(parsed_cfg.n_layers, original.config.n_layers);
    assert_eq!(parsed_cfg.vocab_size, original.config.vocab_size);

    let loaded = model_from_gguf(&file, arch).expect("model");

    // Token embeddings round-trip bit-exact for f32.
    assert_eq!(loaded.token_embeddings, original.token_embeddings);
    assert_eq!(loaded.layers.len(), original.layers.len());
    assert_eq!(loaded.layers[0].wq, original.layers[0].wq);

    // Generate from both — must agree.
    let prompt = [3u32, 7, 12];
    let a = original.generate_greedy(&prompt, 5);
    let b = loaded.generate_greedy(&prompt, 5);
    assert_eq!(a, b, "GGUF round-trip changed the generated sequence");
}
