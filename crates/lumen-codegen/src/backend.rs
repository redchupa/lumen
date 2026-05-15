//! Common backend trait.

use lumen_ir::IrModule;

#[derive(Clone, Debug, Default)]
pub struct Capabilities {
    pub avx2: bool,
    pub avx512f: bool,
    pub neon: bool,
    pub cuda: bool,
    /// Logical cores.
    pub cores: u32,
    /// L1 data cache per core, bytes.
    pub l1d_bytes: u32,
}

#[derive(Clone, Debug, Default)]
pub struct CodegenOpts {
    pub opt_level: u8, // 0..=3
    pub dump_asm: bool,
}

/// A buffer of native machine code (or PTX, for CUDA backend).
pub struct MachineCode {
    pub bytes: Vec<u8>,
    pub entry_offset: usize,
}

pub trait Backend {
    fn name(&self) -> &'static str;
    fn capabilities(&self) -> Capabilities;
    fn lower(&self, ir: &IrModule, opts: &CodegenOpts) -> Result<MachineCode, CodegenError>;
}

#[derive(thiserror::Error, Debug)]
pub enum CodegenError {
    #[error("unsupported op: {0}")]
    UnsupportedOp(String),
    #[error("shape constraint violated: {0}")]
    ShapeError(String),
    #[error("backend not implemented yet")]
    NotImplemented,
}
