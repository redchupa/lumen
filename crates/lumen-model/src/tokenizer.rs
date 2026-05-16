//! Byte-level BPE tokenizer compatible with Qwen2 / Llama-family vocabs.
//!
//! ## Algorithm (encode)
//!
//! 1. Treat the input string as a sequence of single-byte tokens (UTF-8).
//! 2. Repeatedly look at every adjacent pair `(left, right)` in the current
//!    token sequence. If the pair has a merge rank in the table, mark the one
//!    with the **lowest** rank (earliest learned). Replace the pair with the
//!    merged token. Repeat until no mergeable pair exists.
//! 3. Map the final tokens to their vocab IDs.
//!
//! This is the same procedure GPT-2 / tiktoken / Qwen use. Phase 6.A keeps the
//! interface deliberately small — single string in, `Vec<u32>` out — and
//! defers BOS/EOS handling to the caller.
//!
//! ## Decode
//!
//! Each token id maps back to its byte sequence. We concatenate the bytes and
//! decode as UTF-8. Tokens whose bytes form an invalid UTF-8 prefix are
//! returned with a replacement char during display; callers wanting raw bytes
//! should use [`Tokenizer::decode_bytes`].
//!
//! ## Limitations
//!
//! - No GPT-2 pre-tokenization regex split yet (so "hello world" merges as one
//!   stream rather than per-word). This matches what the Lumen e2e LLM test
//!   needs; we'll add the regex split in Phase 6.B when we hook up a real
//!   Qwen vocab and need to match its outputs.
//! - No special tokens (`<|endoftext|>`, `<s>`, ...) yet — also coming with
//!   the real vocab loader.

use std::collections::HashMap;

use crate::gguf::{GgufError, GgufFile};

#[derive(thiserror::Error, Debug)]
pub enum TokError {
    #[error("token id {0} is out of range (vocab size {1})")]
    InvalidTokenId(u32, usize),
    #[error("decoded bytes are not valid UTF-8: {0}")]
    BadUtf8(String),
}

/// One token in the vocabulary, represented by its raw bytes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TokenBytes(pub Vec<u8>);

impl TokenBytes {
    pub fn from_text(s: &str) -> Self {
        Self(s.as_bytes().to_vec())
    }
}

/// The 256-element table mapping each raw byte to the Unicode character
/// GPT-2 / Qwen2 / many other byte-level BPE tokenizers use in their vocab.
///
/// Construction matches the canonical algorithm in `bytes_to_unicode()` from
/// huggingface/transformers (`tokenizers/src/decoders/byte_level.rs` mirrors
/// it). Printable ASCII (33..=126) and the printable subset of Latin-1
/// supplement map to themselves; the remaining 68 control / whitespace
/// codepoints get mapped onto the U+0100..U+0143 range so every byte ends
/// up as a single, distinct, printable character.
pub fn gpt2_byte_to_unicode() -> [char; 256] {
    let mut out = ['\0'; 256];
    let direct: Vec<u8> = (33u8..=126u8)
        .chain(161u8..=172u8)
        .chain(174u8..=255u8)
        .collect();
    for &b in &direct {
        out[b as usize] = b as char;
    }
    let mut next_codepoint: u32 = 0x100;
    for b in 0u8..=255u8 {
        if !direct.contains(&b) {
            out[b as usize] = char::from_u32(next_codepoint).unwrap();
            next_codepoint += 1;
        }
    }
    out
}

/// Inverse of [`gpt2_byte_to_unicode`].
pub fn gpt2_unicode_to_byte() -> HashMap<char, u8> {
    let fwd = gpt2_byte_to_unicode();
    let mut rev = HashMap::with_capacity(256);
    for (i, &c) in fwd.iter().enumerate() {
        rev.insert(c, i as u8);
    }
    rev
}

/// Pre-tokenization / decoding mode applied around the BPE algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BpeMode {
    /// Raw bytes — input UTF-8 bytes are matched against vocab bytes directly.
    /// Useful for toy tokenizers; this is what Phase 6.A built.
    Raw,
    /// GPT-2 byte-level: each raw byte is mapped to a printable Unicode char
    /// via [`gpt2_byte_to_unicode`], the resulting string's UTF-8 bytes are
    /// fed to BPE, and decoding inverts the mapping back to raw bytes.
    ///
    /// Qwen2 / GPT-2 / Llama-2 (GGUF "gpt2") / many other modern tokenizers
    /// use this scheme.
    Gpt2,
}

#[derive(Debug)]
pub struct Tokenizer {
    /// Token id → bytes.
    id_to_bytes: Vec<TokenBytes>,
    /// Bytes → token id (built from `id_to_bytes`).
    bytes_to_id: HashMap<TokenBytes, u32>,
    /// `(left_bytes, right_bytes) → rank`. Lower ranks merge first.
    merge_rank: HashMap<(TokenBytes, TokenBytes), u32>,
    /// Active text-preprocessing mode (Raw or Gpt2).
    mode: BpeMode,
    /// Set when `mode == Gpt2`; nul-initialized otherwise.
    byte_to_unicode: [char; 256],
    unicode_to_byte: HashMap<char, u8>,
}

impl Tokenizer {
    /// Build a tokenizer.
    ///
    /// - `vocab` is the ordered list of token byte strings. Position `i` becomes
    ///   token id `i`.
    /// - `merges` is the ordered list of `(left, right)` pairs in the order
    ///   they were learned. The first one has rank 0 (highest priority).
    ///
    /// Every byte 0..=255 must appear as a single-byte token in the vocab.
    pub fn new(vocab: Vec<TokenBytes>, merges: Vec<(TokenBytes, TokenBytes)>) -> Self {
        Self::with_mode(vocab, merges, BpeMode::Raw)
    }

    pub fn with_mode(
        vocab: Vec<TokenBytes>,
        merges: Vec<(TokenBytes, TokenBytes)>,
        mode: BpeMode,
    ) -> Self {
        let mut bytes_to_id = HashMap::with_capacity(vocab.len());
        for (id, b) in vocab.iter().enumerate() {
            bytes_to_id.insert(b.clone(), id as u32);
        }
        let mut merge_rank = HashMap::with_capacity(merges.len());
        for (rank, (l, r)) in merges.into_iter().enumerate() {
            merge_rank.insert((l, r), rank as u32);
        }
        Self {
            id_to_bytes: vocab,
            bytes_to_id,
            merge_rank,
            mode,
            byte_to_unicode: gpt2_byte_to_unicode(),
            unicode_to_byte: gpt2_unicode_to_byte(),
        }
    }

    pub fn mode(&self) -> BpeMode {
        self.mode
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_bytes.len()
    }

    /// Encode `text` into a sequence of token ids.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        // Initial tokens. Each token's bytes must already exist in the vocab
        // as a single entry — BPE only ever *merges* existing tokens.
        //
        //   Raw  : one TokenBytes per input byte (vocab is assumed to contain
        //          every single-byte token at ids 0..=255).
        //   Gpt2 : one TokenBytes per mapped char (vocab contains each of
        //          the 256 byte-mapped chars as a single-token entry, where
        //          the bytes inside that entry are the UTF-8 form of the
        //          mapped char — 1 byte for printable ASCII, 2 bytes for the
        //          remapped control / latin-1 codepoints).
        let mut tokens: Vec<TokenBytes> = match self.mode {
            BpeMode::Raw => text
                .as_bytes()
                .iter()
                .map(|&b| TokenBytes(vec![b]))
                .collect(),
            BpeMode::Gpt2 => text
                .as_bytes()
                .iter()
                .map(|&b| {
                    let c = self.byte_to_unicode[b as usize];
                    TokenBytes(c.to_string().into_bytes())
                })
                .collect(),
        };

        // Iteratively merge the best (lowest-rank) adjacent pair until no
        // mergeable pair remains.
        loop {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..tokens.len().saturating_sub(1) {
                let pair = (tokens[i].clone(), tokens[i + 1].clone());
                if let Some(&rank) = self.merge_rank.get(&pair) {
                    if best.map_or(true, |(_, r)| rank < r) {
                        best = Some((i, rank));
                    }
                }
            }
            let Some((i, _)) = best else { break };
            // Merge tokens[i] and tokens[i+1] in place.
            let mut merged = tokens[i].0.clone();
            merged.extend_from_slice(&tokens[i + 1].0);
            tokens[i] = TokenBytes(merged);
            tokens.remove(i + 1);
        }

        tokens
            .into_iter()
            .map(|b| {
                *self.bytes_to_id.get(&b).expect(
                    "encode produced a token not in the vocab — vocab must contain all single bytes plus all merge results",
                )
            })
            .collect()
    }

    /// Decode token ids back to a UTF-8 string.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokError> {
        let token_bytes = self.decode_bytes(ids)?;
        match self.mode {
            BpeMode::Raw => {
                String::from_utf8(token_bytes).map_err(|e| TokError::BadUtf8(format!("{}", e)))
            }
            BpeMode::Gpt2 => {
                // token_bytes is the UTF-8 form of a byte-mapped string;
                // walk its chars and invert each one back to its source byte.
                let mapped = std::str::from_utf8(&token_bytes)
                    .map_err(|e| TokError::BadUtf8(format!("{}", e)))?;
                let mut raw = Vec::with_capacity(mapped.len());
                for ch in mapped.chars() {
                    let b = self.unicode_to_byte.get(&ch).ok_or_else(|| {
                        TokError::BadUtf8(format!("unmapped char {:?} in decode", ch))
                    })?;
                    raw.push(*b);
                }
                String::from_utf8(raw).map_err(|e| TokError::BadUtf8(format!("{}", e)))
            }
        }
    }

    /// Decode token ids to raw bytes (no UTF-8 validation).
    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, TokError> {
        let mut out = Vec::new();
        for &id in ids {
            let tb = self
                .id_to_bytes
                .get(id as usize)
                .ok_or(TokError::InvalidTokenId(id, self.id_to_bytes.len()))?;
            out.extend_from_slice(&tb.0);
        }
        Ok(out)
    }

    /// Borrow the bytes of one token (useful for logging / debugging).
    pub fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        self.id_to_bytes.get(id as usize).map(|t| t.0.as_slice())
    }

    /// Build a tokenizer from a parsed GGUF file.
    ///
    /// Reads `tokenizer.ggml.tokens` (required) and `tokenizer.ggml.merges`
    /// (optional — empty if absent, e.g. for unigram tokenizers). Each merge
    /// string is split on the first space into `(left, right)`. Token bytes
    /// are taken verbatim from the GGUF strings, so callers that need to
    /// reverse Qwen/GPT-2's byte-to-unicode pre-mapping should do that step
    /// upstream (Phase 6.C concern, not 6.B).
    pub fn from_gguf(file: &GgufFile) -> Result<Self, GgufError> {
        let tokens = file.metadata_string_array("tokenizer.ggml.tokens")?;
        let vocab: Vec<TokenBytes> = tokens
            .into_iter()
            .map(|s| TokenBytes(s.into_bytes()))
            .collect();

        // Merges are optional. If the key is missing we treat it as no merges.
        let merge_strs = match file.metadata_string_array("tokenizer.ggml.merges") {
            Ok(v) => v,
            Err(GgufError::NoSuchKey(_)) => Vec::new(),
            Err(e) => return Err(e),
        };

        let merges = merge_strs
            .into_iter()
            .filter_map(|s| {
                // Each merge is "<left> <right>". The split is on the *first*
                // space; the rest belongs to the right side (Qwen merges may
                // contain spaces inside the right token's byte mapping).
                let bytes = s.into_bytes();
                let sep = bytes.iter().position(|&b| b == b' ')?;
                let left = TokenBytes(bytes[..sep].to_vec());
                let right = TokenBytes(bytes[sep + 1..].to_vec());
                Some((left, right))
            })
            .collect();

        // Auto-detect mode from `tokenizer.ggml.model`. "gpt2" / "qwen2" etc.
        // all use the byte-level mapping; anything else falls back to Raw.
        let mode = match file.metadata().get("tokenizer.ggml.model") {
            Some(crate::gguf::KvValue::String(s)) if s == "gpt2" => BpeMode::Gpt2,
            _ => BpeMode::Raw,
        };

        Ok(Self::with_mode(vocab, merges, mode))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_byte_vocab(extra: &[&[u8]]) -> Vec<TokenBytes> {
        let mut v: Vec<TokenBytes> = (0u16..256).map(|b| TokenBytes(vec![b as u8])).collect();
        for &e in extra {
            v.push(TokenBytes(e.to_vec()));
        }
        v
    }

    #[test]
    fn empty_input_returns_empty() {
        let tok = Tokenizer::new(build_byte_vocab(&[]), vec![]);
        assert!(tok.encode("").is_empty());
        assert_eq!(tok.decode(&[]).unwrap(), "");
    }

    #[test]
    fn no_merges_passes_through_as_bytes() {
        let tok = Tokenizer::new(build_byte_vocab(&[]), vec![]);
        let ids = tok.encode("abc");
        assert_eq!(ids, vec![b'a' as u32, b'b' as u32, b'c' as u32]);
        assert_eq!(tok.decode(&ids).unwrap(), "abc");
    }

    #[test]
    fn single_merge_collapses_pair() {
        // Vocab: 0..=255 bytes + token "ab" at id 256.
        // Merge: (a, b) → "ab" (rank 0).
        let vocab = build_byte_vocab(&[b"ab"]);
        let merges = vec![(TokenBytes(vec![b'a']), TokenBytes(vec![b'b']))];
        let tok = Tokenizer::new(vocab, merges);
        let ids = tok.encode("ab");
        assert_eq!(ids, vec![256]);
        assert_eq!(tok.decode(&ids).unwrap(), "ab");
    }

    #[test]
    fn rank_chooses_earliest_learned_merge() {
        // Vocab adds "ab" (256), "bc" (257).
        // Merges: rank 0 = (a, b), rank 1 = (b, c).
        // Encoding "abc": rank-0 merge wins → ["ab", "c"] → [256, 99].
        let vocab = build_byte_vocab(&[b"ab", b"bc"]);
        let merges = vec![
            (TokenBytes(vec![b'a']), TokenBytes(vec![b'b'])),
            (TokenBytes(vec![b'b']), TokenBytes(vec![b'c'])),
        ];
        let tok = Tokenizer::new(vocab, merges);
        let ids = tok.encode("abc");
        assert_eq!(ids, vec![256, b'c' as u32]);
        assert_eq!(tok.decode(&ids).unwrap(), "abc");
    }

    #[test]
    fn chained_merges_compose() {
        // "ab", "abc". Merges: (a,b)=0, (ab,c)=1. Encoding "abc" → [257].
        let vocab = build_byte_vocab(&[b"ab", b"abc"]);
        let merges = vec![
            (TokenBytes(vec![b'a']), TokenBytes(vec![b'b'])),
            (TokenBytes(b"ab".to_vec()), TokenBytes(vec![b'c'])),
        ];
        let tok = Tokenizer::new(vocab, merges);
        let ids = tok.encode("abc");
        assert_eq!(ids, vec![257]);
        assert_eq!(tok.decode(&ids).unwrap(), "abc");
    }

    #[test]
    fn korean_round_trip_no_merges() {
        // "안녕" is 6 bytes in UTF-8 (3 each). With no merges we get 6 token ids,
        // and decode reconstructs the original string.
        let tok = Tokenizer::new(build_byte_vocab(&[]), vec![]);
        let s = "안녕";
        let ids = tok.encode(s);
        assert_eq!(ids.len(), s.len());
        assert_eq!(tok.decode(&ids).unwrap(), s);
    }

    #[test]
    fn korean_with_one_merge_uses_byte_pair() {
        // Real Qwen-style: 한글 글자 자체가 multi-byte merge가 되어 vocab에 들어감.
        // 여기서는 "안"의 3바이트를 하나의 토큰으로 학습한 시나리오.
        let an_bytes = "안".as_bytes().to_vec();
        let an_prefix = TokenBytes(an_bytes[..2].to_vec()); // first 2 bytes
        let an_last = TokenBytes(vec![an_bytes[2]]);
        let an_full = TokenBytes(an_bytes.clone());

        let pre1 = TokenBytes(vec![an_bytes[0]]);
        let pre2 = TokenBytes(vec![an_bytes[1]]);
        let vocab = build_byte_vocab(&[&an_bytes[..2], &an_bytes]);
        let merges = vec![
            (pre1, pre2),         // bytes[0] + bytes[1] = prefix (id 256)
            (an_prefix, an_last), // prefix + bytes[2] = full "안" (id 257)
        ];
        let tok = Tokenizer::new(vocab, merges);
        let ids = tok.encode("안");
        assert_eq!(ids, vec![257]);
        assert_eq!(tok.decode(&ids).unwrap(), "안");
        // The full-token bytes match the original UTF-8.
        assert_eq!(tok.token_bytes(257).unwrap(), an_full.0.as_slice());
    }

    #[test]
    fn decode_rejects_unknown_id() {
        let tok = Tokenizer::new(build_byte_vocab(&[]), vec![]);
        let err = tok.decode(&[9999]).unwrap_err();
        assert!(matches!(err, TokError::InvalidTokenId(9999, _)));
    }

    // ---- GPT-2 byte mapping tests -----------------------------------------

    #[test]
    fn gpt2_byte_to_unicode_has_known_anchors() {
        let m = gpt2_byte_to_unicode();
        // Space (0x20) is in the "needs remap" range, classic Ġ at U+0120.
        assert_eq!(m[0x20], '\u{120}');
        // Printable ASCII '!' (0x21) maps to itself.
        assert_eq!(m[0x21], '!');
        // Newline (0x0A) → U+010A (the 11th remapped codepoint).
        assert_eq!(m[0x0A], '\u{10A}');
        // Latin-1 'ÿ' (0xFF) maps to itself.
        assert_eq!(m[0xFF], '\u{FF}');
    }

    #[test]
    fn gpt2_mode_round_trips_ascii_with_correct_token_bytes() {
        // GPT-2 vocab layout: each of the 256 byte-mapped chars is a single
        // token at the start. For printable ASCII this is the same byte;
        // for the remapped range it's the multi-byte UTF-8 of a U+01XX char.
        let byte_unicode = gpt2_byte_to_unicode();
        let mut vocab: Vec<TokenBytes> = byte_unicode
            .iter()
            .map(|&c| TokenBytes(c.to_string().into_bytes()))
            .collect();
        // Add merged "hi" — both 'h' and 'i' are printable ASCII so their
        // mapped form equals themselves.
        vocab.push(TokenBytes("hi".as_bytes().to_vec()));
        let merges = vec![(TokenBytes(vec![b'h']), TokenBytes(vec![b'i']))];

        let tok = Tokenizer::with_mode(vocab, merges, BpeMode::Gpt2);
        let ids = tok.encode("hi");
        assert_eq!(ids, vec![256]);
        assert_eq!(tok.decode(&ids).unwrap(), "hi");
    }

    #[test]
    fn gpt2_mode_handles_space_as_g_dot() {
        // Build a GPT-2-style vocab where every byte-mapped char is one token.
        let byte_unicode = gpt2_byte_to_unicode();
        let vocab: Vec<TokenBytes> = byte_unicode
            .iter()
            .map(|&c| TokenBytes(c.to_string().into_bytes()))
            .collect();
        let tok = Tokenizer::with_mode(vocab, vec![], BpeMode::Gpt2);

        // " " (0x20) maps to 'Ġ' (U+0120), the 33rd remapped codepoint, whose
        // id is 32 + the count of bytes already mapped before it in the
        // remap pass. We don't hard-code the id — just check the round-trip.
        let ids = tok.encode(" ");
        assert_eq!(ids.len(), 1);
        assert_eq!(tok.decode(&ids).unwrap(), " ");
    }
}
