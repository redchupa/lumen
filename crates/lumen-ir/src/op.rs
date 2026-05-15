//! IR opcodes.
//!
//! Kept small on purpose. New ops require an RFC (`docs/rfc/`).

use crate::module::ValueId;

#[derive(Clone, Debug)]
pub enum Op {
    // -- Function I/O --
    /// `param i` — i-th function parameter. Result type taken from the
    /// owning function's signature.
    Param {
        index: u32,
    },

    // -- Memory / constants --
    LoadWeight {
        name: String,
    },
    LoadInput {
        index: u32,
    },
    Constant {
        bits: u128,
    }, // dtype lives on the result Value

    // -- Linear algebra --
    MatMul {
        lhs: ValueId,
        rhs: ValueId,
    },
    Add {
        lhs: ValueId,
        rhs: ValueId,
    },
    Mul {
        lhs: ValueId,
        rhs: ValueId,
    },

    // -- Activations --
    Silu {
        x: ValueId,
    },
    Gelu {
        x: ValueId,
    },
    Softmax {
        x: ValueId,
        axis: i32,
    },

    // -- Normalization --
    RmsNorm {
        x: ValueId,
        weight: ValueId,
        eps: f32,
    },

    // -- Attention helpers --
    Rope {
        x: ValueId,
        positions: ValueId,
        base: f32,
    },
    /// View/reshape without copying.
    Reshape {
        x: ValueId,
        new_shape: Vec<u32>,
    },
    Transpose {
        x: ValueId,
        perm: Vec<u32>,
    },

    // -- Quantization --
    Dequantize {
        x: ValueId,
    },
    Quantize {
        x: ValueId,
        target: crate::ty::DType,
    },

    // -- Control --
    Return {
        value: ValueId,
    },
}
