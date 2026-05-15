//! Runtime tensor. Reference-counted buffer + shape/stride metadata.

use std::sync::Arc;

use lumen_ir::ty::DType;

pub struct TensorBuffer {
    pub bytes: Vec<u8>,
}

#[derive(Clone)]
pub struct Tensor {
    pub buffer: Arc<TensorBuffer>,
    pub dtype: DType,
    pub shape: Vec<u32>,
    pub stride: Vec<u32>, // in elements, not bytes
    pub offset: u32,
}

impl Tensor {
    pub fn numel(&self) -> u64 {
        self.shape.iter().map(|&d| d as u64).product()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}
