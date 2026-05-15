//! Model architectures: graph builders that emit IR for a given config.
//!
//! Supported in v1.0:
//!   - Qwen2 / Qwen2.5 (GQA, RoPE, SiLU, RMSNorm)
//!   - Llama 3 family (GQA, RoPE, SiLU, RMSNorm)
//!   - EXAONE 3.5
//!   - A.X 3.1
//!
//! Each module here exports `fn build_graph(cfg: &ArchConfig) -> IrModule`.

use lumen_ir::IrModule;

pub struct ArchConfig {
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub hidden_dim: u32,
    pub ffn_dim: u32,
    pub vocab_size: u32,
    pub rope_base: f32,
}

pub fn build_qwen2(_cfg: &ArchConfig) -> IrModule {
    IrModule::default()
}
