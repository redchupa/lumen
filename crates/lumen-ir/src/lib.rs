//! Lumen IR — SSA form. Tensor shapes are part of types so the codegen can
//! statically specialize loops, tiling, and quantization unpack/repack.
//!
//! High-level shape: `AST → IrModule → passes → lowered IrModule → codegen`.

#![allow(dead_code)] // Phase 0 scaffolding.

pub mod lower;
pub mod module;
pub mod op;
pub mod pass;
pub mod ty;

pub use module::{Block, Function, IrModule, Value, ValueId};
pub use op::Op;
pub use ty::{DType, Dim, Shape, TensorType};
