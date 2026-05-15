//! CUDA backend. v1.0 ships cuBLAS fallback; native PTX emission lands later.

use crate::backend::{Backend, Capabilities, CodegenError, CodegenOpts, MachineCode};
use lumen_ir::IrModule;

pub struct Cuda;

impl Backend for Cuda {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    fn lower(&self, _ir: &IrModule, _opts: &CodegenOpts) -> Result<MachineCode, CodegenError> {
        Err(CodegenError::NotImplemented)
    }
}
