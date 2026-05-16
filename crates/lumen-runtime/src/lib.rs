//! Lumen runtime: tensors, memory pools, kernel dispatch, KV cache.

#![allow(dead_code)] // Phase 0 scaffolding.

pub mod dispatch;
pub mod kvcache;
pub mod ops;
pub mod pool;
pub mod quant;
pub mod tensor;

pub use ops::{mul_in_place, rms_norm, rope_in_place, silu_in_place, softmax_rows};

pub use quant::{
    dequantize_q4_0, dequantize_q8_0, f16_bits_to_f32, f32_to_f16_bits, quantize_q8_0, BlockQ4_0,
    BlockQ8_0, QK,
};
pub use tensor::Tensor;
