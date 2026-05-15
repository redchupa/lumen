//! ARM64 (AArch64) backend. NEON in Phase 2, SVE if a target machine appears.

use crate::backend::{Backend, Capabilities, CodegenError, CodegenOpts, MachineCode};
use lumen_ir::IrModule;

pub struct Arm64;

impl Backend for Arm64 {
    fn name(&self) -> &'static str {
        "arm64"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    fn lower(&self, _ir: &IrModule, _opts: &CodegenOpts) -> Result<MachineCode, CodegenError> {
        Err(CodegenError::NotImplemented)
    }
}
