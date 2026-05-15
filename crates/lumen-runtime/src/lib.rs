//! Lumen runtime: tensors, memory pools, kernel dispatch, KV cache.

#![allow(dead_code)] // Phase 0 scaffolding.

pub mod tensor;
pub mod pool;
pub mod kvcache;
pub mod dispatch;

pub use tensor::Tensor;
