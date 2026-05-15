//! x86_64 backend. AVX2 first, AVX-512 in Phase 3.

use crate::backend::{Backend, Capabilities, CodegenError, CodegenOpts, MachineCode};
use lumen_ir::IrModule;

pub struct X86_64;

impl Backend for X86_64 {
    fn name(&self) -> &'static str {
        "x86_64"
    }

    fn capabilities(&self) -> Capabilities {
        // TODO Phase 2: probe CPUID
        Capabilities::default()
    }

    fn lower(&self, _ir: &IrModule, _opts: &CodegenOpts) -> Result<MachineCode, CodegenError> {
        Err(CodegenError::NotImplemented)
    }
}
