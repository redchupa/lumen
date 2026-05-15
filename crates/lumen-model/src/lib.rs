//! Model loading and tokenization.
//!
//! Phase 5+6 work. Initial scope: GGUF reader (Q4_0, Q4_K, Q8_0) and BPE
//! tokenizer compatible with Llama/Qwen/EXAONE vocabularies.

#![allow(dead_code)] // Phase 0 scaffolding.

pub mod gguf;
pub mod tokenizer;
pub mod arch;

pub use gguf::GgufFile;
pub use tokenizer::Tokenizer;
