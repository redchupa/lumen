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

pub struct Tokenizer {
    /// Token id → bytes.
    id_to_bytes: Vec<TokenBytes>,
    /// Bytes → token id (built from `id_to_bytes`).
    bytes_to_id: HashMap<TokenBytes, u32>,
    /// `(left_bytes, right_bytes) → rank`. Lower ranks merge first.
    merge_rank: HashMap<(TokenBytes, TokenBytes), u32>,
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
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_bytes.len()
    }

    /// Encode `text` into a sequence of token ids.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        // Start with every byte as its own token.
        let mut tokens: Vec<TokenBytes> = text
            .as_bytes()
            .iter()
            .map(|&b| TokenBytes(vec![b]))
            .collect();

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
        let bytes = self.decode_bytes(ids)?;
        String::from_utf8(bytes).map_err(|e| TokError::BadUtf8(format!("{}", e)))
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
}
