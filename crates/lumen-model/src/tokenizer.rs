//! BPE tokenizer compatible with Llama/Qwen/EXAONE vocabs. Phase 6 work.

#[derive(thiserror::Error, Debug)]
pub enum TokError {
    #[error("not implemented yet")]
    NotImplemented,
}

pub struct Tokenizer;

impl Tokenizer {
    pub fn encode(&self, _text: &str) -> Result<Vec<u32>, TokError> {
        Err(TokError::NotImplemented)
    }

    pub fn decode(&self, _ids: &[u32]) -> Result<String, TokError> {
        Err(TokError::NotImplemented)
    }
}
