//! Lumen runtime: tensors, memory pools, kernel dispatch, KV cache.

#![allow(dead_code)] // Phase 0 scaffolding.

pub mod dispatch;
pub mod kvcache;
pub mod pool;
pub mod tensor;

pub use tensor::Tensor;
