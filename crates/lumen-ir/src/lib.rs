//! Lumen IR — SSA form. Tensor shapes are part of types so the codegen can
//! statically specialize loops, tiling, and quantization unpack/repack.
//!
//! High-level shape: `AST → IrModule → passes → lowered IrModule → codegen`.

#![allow(dead_code)] // Phase 0 scaffolding.

pub mod module;
pub mod op;
pub mod ty;
pub mod pass;
pub mod lower;

pub use module::{IrModule, Function, Block, Value, ValueId};
pub use op::Op;
pub use ty::{TensorType, DType, Shape, Dim};
