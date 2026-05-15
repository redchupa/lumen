//! End-to-end: write a toy GGUF buffer that carries a small BPE vocab and
//! merge list as metadata, parse it back through `GgufFile`, then build a
//! `Tokenizer` from it and check that encode/decode round-trip.

use lumen_model::gguf::{GgufFile, KvType};
use lumen_model::tokenizer::Tokenizer;

fn write_string(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(bytes);
}

fn write_kv_string(buf: &mut Vec<u8>, key: &str, value: &str) {
    write_string(buf, key);
    buf.extend_from_slice(&(KvType::String as u32).to_le_bytes());
    write_string(buf, value);
}

fn write_kv_string_array(buf: &mut Vec<u8>, key: &str, values: &[&str]) {
    write_string(buf, key);
    buf.extend_from_slice(&(KvType::Array as u32).to_le_bytes());
    buf.extend_from_slice(&(KvType::String as u32).to_le_bytes()); // element type
    buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for v in values {
        write_string(buf, v);
    }
}

fn make_toy_gguf_with_vocab(vocab: &[&str], merges: &[&str]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes()); // version
    buf.extend_from_slice(&0u64.to_le_bytes()); // n_tensors = 0
    buf.extend_from_slice(&3u64.to_le_bytes()); // n_kv = 3

    write_kv_string(&mut buf, "tokenizer.ggml.model", "lumen.test");
    write_kv_string_array(&mut buf, "tokenizer.ggml.tokens", vocab);
    write_kv_string_array(&mut buf, "tokenizer.ggml.merges", merges);

    // Pad to alignment 32 (default), no tensor data.
    let pad = (32 - (buf.len() % 32)) % 32;
    buf.resize(buf.len() + pad, 0);
    buf
}

#[test]
fn loads_vocab_and_merges_round_trip() {
    // 256 single-byte tokens at IDs 0..255, then 3 multi-byte merges.
    let mut vocab: Vec<String> = (0u16..256).map(|b| (b as u8 as char).to_string()).collect();
    vocab.push("ab".to_string());
    vocab.push("bc".to_string());
    vocab.push("abc".to_string());
    let vocab_strs: Vec<&str> = vocab.iter().map(|s| s.as_str()).collect();

    let merges = vec![
        "a b",  // → "ab" (rank 0)
        "b c",  // → "bc" (rank 1)
        "ab c", // → "abc" (rank 2)
    ];

    let gguf_bytes = make_toy_gguf_with_vocab(&vocab_strs, &merges);
    let file = GgufFile::from_bytes(gguf_bytes).expect("parse");
    let tok = Tokenizer::from_gguf(&file).expect("tokenizer");

    assert_eq!(tok.vocab_size(), 256 + 3);

    // "abc" should compose to id 258 (the "abc" entry).
    let ids = tok.encode("abc");
    assert_eq!(ids, vec![258]);
    assert_eq!(tok.decode(&ids).unwrap(), "abc");

    // "ab" alone composes to id 256.
    let ids = tok.encode("ab");
    assert_eq!(ids, vec![256]);

    // Bytes that don't match any merge fall through.
    let ids = tok.encode("z");
    assert_eq!(ids, vec![b'z' as u32]);
}

#[test]
fn tokenizer_works_when_merges_are_absent() {
    // Some unigram-style tokenizers ship a vocab with no merge list.
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
    buf.extend_from_slice(&1u64.to_le_bytes()); // n_kv = 1

    let mut vocab_strs: Vec<String> = (0u16..256).map(|b| (b as u8 as char).to_string()).collect();
    vocab_strs.push("hello".to_string());
    let vocab_refs: Vec<&str> = vocab_strs.iter().map(|s| s.as_str()).collect();
    write_kv_string_array(&mut buf, "tokenizer.ggml.tokens", &vocab_refs);

    let pad = (32 - (buf.len() % 32)) % 32;
    buf.resize(buf.len() + pad, 0);

    let file = GgufFile::from_bytes(buf).expect("parse");
    let tok = Tokenizer::from_gguf(&file).expect("tokenizer");
    assert_eq!(tok.vocab_size(), 257);

    // No merges → encoding is purely byte-level. "hello" → 5 byte ids.
    let ids = tok.encode("hello");
    assert_eq!(ids.len(), 5);
    assert_eq!(tok.decode(&ids).unwrap(), "hello");
}

#[test]
fn missing_tokens_key_is_an_error() {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
    buf.extend_from_slice(&0u64.to_le_bytes()); // n_kv = 0
    let pad = (32 - (buf.len() % 32)) % 32;
    buf.resize(buf.len() + pad, 0);

    let file = GgufFile::from_bytes(buf).expect("parse");
    let err = Tokenizer::from_gguf(&file).unwrap_err();
    let msg = format!("{}", err);
    assert!(msg.contains("tokenizer.ggml.tokens"), "got: {}", msg);
}
