//! End-to-end: write a GGUF file containing a Q8_0 tensor, read it back,
//! and verify that the dequantized values match what Lumen's reference
//! dequant produces directly from `BlockQ8_0` blocks.
//!
//! This is the bridge that connects on-disk GGUF format to the native
//! Lumen runtime quantization kernels.

use std::io::Write;

use lumen_model::gguf::{GgmlType, GgufFile, KvType};
use lumen_runtime::quant::{dequantize_q8_0, quantize_q8_0, BlockQ8_0, QK};

fn write_string(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(bytes);
}

fn write_gguf_with_q8_tensor(name: &str, dims: &[u64], blocks: &[BlockQ8_0]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes()); // version
    buf.extend_from_slice(&1u64.to_le_bytes()); // n_tensors = 1
    buf.extend_from_slice(&1u64.to_le_bytes()); // n_kv = 1

    // Alignment KV.
    write_string(&mut buf, "general.alignment");
    buf.extend_from_slice(&(KvType::U32 as u32).to_le_bytes());
    buf.extend_from_slice(&32u32.to_le_bytes());

    // Tensor info.
    write_string(&mut buf, name);
    buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for d in dims {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    buf.extend_from_slice(&(GgmlType::Q8_0 as u32).to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // tensor offset = 0

    // Align to 32 before data.
    let pad = (32 - (buf.len() % 32)) % 32;
    buf.resize(buf.len() + pad, 0);

    // Raw tensor bytes. Each BlockQ8_0 is exactly 34 bytes; we reinterpret.
    // SAFETY: BlockQ8_0 is `repr(C, packed)` and exactly 34 bytes.
    let raw: &[u8] =
        unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, blocks.len() * 34) };
    buf.write_all(raw).unwrap();
    buf
}

#[test]
fn round_trip_gguf_q8_tensor() {
    // 128 elements = 4 Q8_0 blocks. Shape [4, 32].
    let n_elems = 128usize;
    let src: Vec<f32> = (0..n_elems)
        .map(|i| ((i % 17) as f32) * 0.05 - 0.5)
        .collect();
    let mut blocks = vec![BlockQ8_0 { d: 0, qs: [0; 32] }; n_elems / QK];
    quantize_q8_0(&src, &mut blocks);

    let bytes = write_gguf_with_q8_tensor("blk.0.weight", &[4, 32], &blocks);

    // Parse back.
    let file = GgufFile::from_bytes(bytes).expect("parse gguf");
    let info = file.tensor("blk.0.weight").expect("tensor info");
    assert_eq!(info.dtype, GgmlType::Q8_0);
    assert_eq!(info.dims, vec![4, 32]);
    assert_eq!(info.byte_size(), (n_elems as u64 / QK as u64) * 34);

    // Get raw tensor bytes and reinterpret as Q8_0 blocks.
    let data = file.tensor_data("blk.0.weight").expect("tensor data");
    assert_eq!(data.len(), info.byte_size() as usize);
    // SAFETY: BlockQ8_0 is repr(C, packed); the GGUF tensor bytes are exactly
    // n_blocks * 34 by construction.
    let parsed_blocks: &[BlockQ8_0] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const BlockQ8_0, data.len() / 34) };

    // Dequantize both sides via the reference and compare.
    let mut got = vec![0.0f32; n_elems];
    dequantize_q8_0(parsed_blocks, &mut got);

    let mut want = vec![0.0f32; n_elems];
    dequantize_q8_0(&blocks, &mut want);

    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-6,
            "idx {}: parsed {} != original {}",
            i,
            g,
            w
        );
    }
}
