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
    /// If set, the Q8×Q8 fused matmul kernel emits `vpdpbusd` in either VEX
    /// (AVX-VNNI) or EVEX (AVX-512 VNNI) form instead of the
    /// `vpsignb+vpmaddubsw+vpmaddwd` chain. Phase 7.N.
    pub vnni: Option<crate::x86_64::VnniForm>,
    /// If true, the Q8×F32 fused matmul kernel emits ZMM-wide (16 fp32 lane)
    /// inner loops instead of YMM-wide (8 fp32 lane). Requires AVX-512F
    /// (and AVX-512BW for the vpmovsxbd zmm load). Phase 7.S.
    pub use_avx512: bool,
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
