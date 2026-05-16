//! Model loading and tokenization.
//!
//! Phase 5+6 work. Initial scope: GGUF reader (Q4_0, Q4_K, Q8_0) and BPE
//! tokenizer compatible with Llama/Qwen/EXAONE vocabularies.

#![allow(dead_code)] // Phase 0 scaffolding.

pub mod arch;
pub mod gguf;
pub mod loader;
pub mod tokenizer;

pub use gguf::{GgmlType, GgufError, GgufFile, KvType, KvValue, TensorInfo};
pub use loader::{config_from_gguf, model_from_gguf};
pub use tokenizer::{TokError, TokenBytes, Tokenizer};
