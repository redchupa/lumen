//! Build a `lumen_runtime::Model` from a parsed GGUF file.
//!
//! Tensor naming follows the ggml / llama.cpp convention:
//!
//! ```text
//!   token_embd.weight              [vocab, hidden]
//!   output_norm.weight             [hidden]
//!   output.weight                  [vocab, hidden]  (lm_head; may be tied)
//!   blk.{i}.attn_norm.weight       [hidden]
//!   blk.{i}.attn_q.weight          [q_dim,  hidden]
//!   blk.{i}.attn_k.weight          [kv_dim, hidden]
//!   blk.{i}.attn_v.weight          [kv_dim, hidden]
//!   blk.{i}.attn_output.weight     [hidden, q_dim]
//!   blk.{i}.ffn_norm.weight        [hidden]
//!   blk.{i}.ffn_gate.weight        [ffn_hidden, hidden]
//!   blk.{i}.ffn_up.weight          [ffn_hidden, hidden]
//!   blk.{i}.ffn_down.weight        [hidden, ffn_hidden]
//! ```
//!
//! Architecture metadata keys use a `{arch}.` prefix; this loader takes the
//! arch string as a parameter (look it up yourself via
//! `file.metadata().get("general.architecture")`).
//!
//! As of Phase 6.F.2.a this loader handles F32, F16, Q8_0, and Q4_0 by
//! dequantizing through the references in `lumen_runtime::quant`.
//! All Model weights are stored as fp32 at runtime.

use lumen_runtime::quant::{
    dequantize_q4_0, dequantize_q8_0, f16_bits_to_f32, BlockQ4_0, BlockQ8_0, QK,
};
use lumen_runtime::{LayerConfig, LayerWeights, Model, ModelConfig};

use crate::gguf::{GgmlType, GgufError, GgufFile, KvValue};

/// Decode a GGUF-stored tensor's raw bytes into an `f32` vector.
///
/// Handles:
/// - F32: zero-copy reinterpretation.
/// - F16: per-element fp16 → fp32 conversion.
/// - Q8_0: block dequant (34 bytes per 32 elements).
/// - Q4_0: block dequant (18 bytes per 32 elements, nibble-packed).
///
/// Any other dtype surfaces as `UnsupportedTensorType`.
fn tensor_to_f32(file: &GgufFile, name: &str) -> Result<Vec<f32>, GgufError> {
    let info = file
        .tensor(name)
        .ok_or_else(|| GgufError::NoSuchTensor(name.to_string()))?;
    let bytes = file.tensor_data(name)?;

    match info.dtype {
        GgmlType::F32 => {
            debug_assert_eq!(bytes.len() % 4, 0);
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            Ok(out)
        }
        GgmlType::F16 => {
            debug_assert_eq!(bytes.len() % 2, 0);
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                out.push(f16_bits_to_f32(bits));
            }
            Ok(out)
        }
        GgmlType::Q8_0 => {
            debug_assert_eq!(bytes.len() % 34, 0);
            let n_blocks = bytes.len() / 34;
            let nelem = info.element_count() as usize;
            debug_assert_eq!(nelem, n_blocks * QK);
            // SAFETY: `BlockQ8_0` is `repr(C, packed)` and exactly 34 bytes;
            // the GGUF payload is a contiguous run of those same blocks.
            let blocks: &[BlockQ8_0] =
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const BlockQ8_0, n_blocks) };
            let mut out = vec![0.0f32; nelem];
            dequantize_q8_0(blocks, &mut out);
            Ok(out)
        }
        GgmlType::Q4_0 => {
            debug_assert_eq!(bytes.len() % 18, 0);
            let n_blocks = bytes.len() / 18;
            let nelem = info.element_count() as usize;
            debug_assert_eq!(nelem, n_blocks * QK);
            // SAFETY: `BlockQ4_0` is `repr(C, packed)` and exactly 18 bytes.
            let blocks: &[BlockQ4_0] =
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const BlockQ4_0, n_blocks) };
            let mut out = vec![0.0f32; nelem];
            dequantize_q4_0(blocks, &mut out);
            Ok(out)
        }
        other => Err(GgufError::UnsupportedTensorType(other as u32)),
    }
}

fn get_u32(file: &GgufFile, key: &str) -> Result<u32, GgufError> {
    match file.metadata().get(key) {
        Some(KvValue::U32(v)) => Ok(*v),
        Some(KvValue::U64(v)) => Ok(*v as u32),
        Some(KvValue::I32(v)) => Ok(*v as u32),
        Some(_) => Err(GgufError::WrongKvKind {
            key: key.to_string(),
            expected: crate::gguf::KvType::U32,
        }),
        None => Err(GgufError::NoSuchKey(key.to_string())),
    }
}

fn get_f32(file: &GgufFile, key: &str) -> Result<f32, GgufError> {
    match file.metadata().get(key) {
        Some(KvValue::F32(v)) => Ok(*v),
        Some(KvValue::F64(v)) => Ok(*v as f32),
        Some(_) => Err(GgufError::WrongKvKind {
            key: key.to_string(),
            expected: crate::gguf::KvType::F32,
        }),
        None => Err(GgufError::NoSuchKey(key.to_string())),
    }
}

fn get_u32_optional(file: &GgufFile, key: &str) -> Option<u32> {
    get_u32(file, key).ok()
}

/// Read architecture hyperparameters from GGUF metadata. `arch` is the prefix
/// used by this model (e.g. "llama", "qwen2", "lumen.test").
pub fn config_from_gguf(file: &GgufFile, arch: &str) -> Result<ModelConfig, GgufError> {
    let hidden = get_u32(file, &format!("{}.embedding_length", arch))? as usize;
    let n_layers = get_u32(file, &format!("{}.block_count", arch))? as usize;
    let n_heads = get_u32(file, &format!("{}.attention.head_count", arch))? as usize;
    let n_kv_heads = get_u32(file, &format!("{}.attention.head_count_kv", arch))
        .unwrap_or(n_heads as u32) as usize;
    let ffn_hidden = get_u32(file, &format!("{}.feed_forward_length", arch))? as usize;
    let max_seq = get_u32(file, &format!("{}.context_length", arch))? as usize;
    let rms_norm_eps = get_f32(file, &format!("{}.attention.layer_norm_rms_epsilon", arch))?;
    let rope_base = get_f32(file, &format!("{}.rope.freq_base", arch)).unwrap_or(10000.0);

    let head_dim = hidden / n_heads;
    assert_eq!(
        hidden,
        n_heads * head_dim,
        "hidden must equal n_heads * head_dim"
    );

    let vocab_size = {
        // Prefer the tensor table — token_embd carries the authoritative count.
        let info = file
            .tensor("token_embd.weight")
            .ok_or_else(|| GgufError::NoSuchTensor("token_embd.weight".into()))?;
        info.dims[1] as usize // [vocab, hidden] stored as [d0, d1] = [hidden, vocab] in ggml. Wait — see note below.
    };
    // ggml stores `token_embd.weight` with dims [hidden, vocab]; the row index
    // is the token id when laid out row-major as [vocab, hidden] in fp32.
    // We treat the second dim as vocab_size to match. If your GGUF disagrees,
    // swap the index. (Some converters do reverse this.)

    Ok(ModelConfig {
        layer: LayerConfig {
            hidden,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_hidden,
            rms_norm_eps,
            rope_base,
        },
        vocab_size,
        n_layers,
        max_seq,
        eos_token_id: get_u32_optional(file, "tokenizer.ggml.eos_token_id"),
    })
}

/// Construct a `Model` from a parsed GGUF file using the standard ggml tensor
/// names. All weights must currently be `GgmlType::F32`.
pub fn model_from_gguf(file: &GgufFile, arch: &str) -> Result<Model, GgufError> {
    let cfg = config_from_gguf(file, arch)?;
    let n_layers = cfg.n_layers;

    let token_embeddings = tensor_to_f32(file, "token_embd.weight")?;
    let final_norm_w = tensor_to_f32(file, "output_norm.weight")?;
    let lm_head_w = match tensor_to_f32(file, "output.weight") {
        Ok(v) => v,
        Err(GgufError::NoSuchTensor(_)) => token_embeddings.clone(), // weight tying
        Err(e) => return Err(e),
    };

    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let p = format!("blk.{}", i);
        layers.push(LayerWeights {
            attn_norm_w: tensor_to_f32(file, &format!("{}.attn_norm.weight", p))?,
            wq: tensor_to_f32(file, &format!("{}.attn_q.weight", p))?,
            wk: tensor_to_f32(file, &format!("{}.attn_k.weight", p))?,
            wv: tensor_to_f32(file, &format!("{}.attn_v.weight", p))?,
            wo: tensor_to_f32(file, &format!("{}.attn_output.weight", p))?,
            ffn_norm_w: tensor_to_f32(file, &format!("{}.ffn_norm.weight", p))?,
            w_gate: tensor_to_f32(file, &format!("{}.ffn_gate.weight", p))?,
            w_up: tensor_to_f32(file, &format!("{}.ffn_up.weight", p))?,
            w_down: tensor_to_f32(file, &format!("{}.ffn_down.weight", p))?,
        });
    }

    Ok(Model {
        config: cfg,
        token_embeddings,
        layers,
        final_norm_w,
        lm_head_w,
    })
}
