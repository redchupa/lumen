//! Lumen IR — SSA form. Tensor shapes are part of types so the codegen can
//! statically specialize loops, tiling, and quantization unpack/repack.
//!
//! High-level shape: `AST → IrModule → passes → lowered IrModule → codegen`.

pub mod lower;
pub mod module;
pub mod op;
pub mod pass;
pub mod printer;
pub mod ty;
pub mod verifier;

pub use lower::{lower, LowerError};
pub use module::{Block, Function, IrModule, Value, ValueId};
pub use op::Op;
pub use printer::{print_function, print_module};
pub use ty::{DType, Dim, Shape, TensorType};
pub use verifier::{verify_module, VerifyError};
