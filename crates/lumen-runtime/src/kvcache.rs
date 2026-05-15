//! KV cache. Phase 6 work. paged-attention style allocator planned.

pub struct KvCache {
    pub layers: u32,
    pub max_tokens: u32,
    pub head_dim: u32,
    pub kv_heads: u32,
}

impl KvCache {
    pub fn new(layers: u32, max_tokens: u32, kv_heads: u32, head_dim: u32) -> Self {
        Self { layers, max_tokens, head_dim, kv_heads }
    }
}
