//! JIT engine: turn `MachineCode` into a callable function pointer, and cache
//! by (IR-hash, input-shape).
//!
//! Phase 4 work. This crate's `unsafe` use is intentionally narrow:
//!   - `exec.rs` allocates W^X executable pages.
//!   - Nothing else uses `unsafe`.

#![allow(dead_code)] // Phase 0 scaffolding.

pub mod cache;
pub mod exec;

pub use cache::CodeCache;
pub use exec::{ExecError, ExecRegion};
