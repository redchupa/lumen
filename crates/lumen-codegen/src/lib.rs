//! Code generation backends.
//!
//! Each backend implements the [`Backend`] trait. The CLI picks one at runtime
//! based on detected hardware (`Capabilities`).

#![allow(dead_code)] // Phase 0 scaffolding.

pub mod backend;
pub mod x86_64;
pub mod arm64;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod emit;

pub use backend::{Backend, Capabilities, CodegenOpts, MachineCode};
