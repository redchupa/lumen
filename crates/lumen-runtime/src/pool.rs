//! Size-class buffer pool so we don't allocate per token.

#[derive(Default)]
pub struct BufferPool {
    // TODO Phase 6: implement freelists per size class.
}

impl BufferPool {
    pub fn new() -> Self {
        Self::default()
    }
}
