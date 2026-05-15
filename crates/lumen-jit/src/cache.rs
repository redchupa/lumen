//! Specialized-kernel cache keyed by (IR fingerprint, runtime shape).

use std::collections::HashMap;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub ir_hash: u64,
    pub shape: Vec<u32>,
    pub backend: &'static str,
}

#[derive(Default)]
pub struct CodeCache {
    entries: HashMap<CacheKey, usize>, // value = index into module list (TBD)
}

impl CodeCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &CacheKey) -> Option<usize> {
        self.entries.get(key).copied()
    }

    pub fn insert(&mut self, key: CacheKey, idx: usize) {
        self.entries.insert(key, idx);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
