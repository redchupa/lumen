//! IR types. Tensor shape lives in the type system.

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F16,
    F32,
    I32,
    /// 4-bit, block-quantized (32 values per block + fp16 scale). GGUF compatible.
    Q4_0,
    /// 8-bit, block-quantized (32 values per block + fp16 scale).
    Q8_0,
}

impl DType {
    pub const fn bytes_per_element_x1000(self) -> u32 {
        // ×1000 to keep integer math for fractional sizes (q4_0 = 0.5).
        match self {
            DType::F16 => 2_000,
            DType::F32 => 4_000,
            DType::I32 => 4_000,
            DType::Q4_0 => 500 + (2_000 / 32), // 4 bits per value + fp16 scale per 32
            DType::Q8_0 => 1_000 + (2_000 / 32),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shape(pub Vec<Dim>);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Dim {
    Static(u32),
    /// JIT-resolved.
    Dynamic(u32), // id, so two `Dynamic(0)` refer to the same dim
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorType {
    pub dtype: DType,
    pub shape: Shape,
}

impl TensorType {
    pub fn is_fully_static(&self) -> bool {
        self.shape.0.iter().all(|d| matches!(d, Dim::Static(_)))
    }
}
