//! Code generation backends.
//!
//! Each backend implements the [`Backend`] trait. The CLI picks one at runtime
//! based on detected hardware (`Capabilities`).

#![allow(dead_code)] // Phase 0 scaffolding.

pub mod arm64;
pub mod backend;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod emit;
pub mod x86_64;

pub use backend::{Backend, Capabilities, CodegenOpts, MachineCode};
