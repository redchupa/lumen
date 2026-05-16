//! Per-layer key/value cache for autoregressive decoding.
//!
//! Layout: `K[i, d] = self.k[i * kv_dim + d]`. The cache grows by one token
//! per `append`. `cur_len` is the number of tokens currently stored.
//!
//! Phase 6.E.1 scope: simplest possible contiguous-buffer layout. Capacity is
//! fixed at construction (`max_seq` tokens). Paged-attention style allocators
//! are a v1.1 concern (see ARCHITECTURE.md §3.5).

/// One layer's worth of K/V slots, sized for `max_seq` tokens × `kv_dim`
/// floats each.
#[derive(Clone, Debug)]
pub struct LayerKvCache {
    /// Flat buffer `[max_seq, kv_dim]` row-major.
    k: Vec<f32>,
    v: Vec<f32>,
    kv_dim: usize,
    max_seq: usize,
    cur_len: usize,
}

impl LayerKvCache {
    pub fn new(max_seq: usize, kv_dim: usize) -> Self {
        Self {
            k: vec![0.0; max_seq * kv_dim],
            v: vec![0.0; max_seq * kv_dim],
            kv_dim,
            max_seq,
            cur_len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.cur_len
    }

    pub fn is_empty(&self) -> bool {
        self.cur_len == 0
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Append one token's K and V vectors. `k_row.len()` and `v_row.len()`
    /// must equal `kv_dim`. Panics on overflow past `max_seq`.
    pub fn append(&mut self, k_row: &[f32], v_row: &[f32]) {
        assert_eq!(k_row.len(), self.kv_dim);
        assert_eq!(v_row.len(), self.kv_dim);
        assert!(
            self.cur_len < self.max_seq,
            "kv cache full (max_seq={})",
            self.max_seq
        );
        let off = self.cur_len * self.kv_dim;
        self.k[off..off + self.kv_dim].copy_from_slice(k_row);
        self.v[off..off + self.kv_dim].copy_from_slice(v_row);
        self.cur_len += 1;
    }

    /// Borrow the K rows that have been populated so far: `[cur_len, kv_dim]`.
    pub fn k_filled(&self) -> &[f32] {
        &self.k[..self.cur_len * self.kv_dim]
    }

    /// Borrow the V rows that have been populated so far.
    pub fn v_filled(&self) -> &[f32] {
        &self.v[..self.cur_len * self.kv_dim]
    }

    pub fn reset(&mut self) {
        self.cur_len = 0;
        // Zeroing isn't required — `k_filled`/`v_filled` slice by cur_len.
    }
}

/// One cache slot per transformer layer.
#[derive(Clone, Debug)]
pub struct KvCache {
    layers: Vec<LayerKvCache>,
}

impl KvCache {
    pub fn new(n_layers: usize, max_seq: usize, kv_dim: usize) -> Self {
        Self {
            layers: (0..n_layers)
                .map(|_| LayerKvCache::new(max_seq, kv_dim))
                .collect(),
        }
    }

    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn layer(&self, i: usize) -> &LayerKvCache {
        &self.layers[i]
    }

    pub fn layer_mut(&mut self, i: usize) -> &mut LayerKvCache {
        &mut self.layers[i]
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_and_read_back() {
        let mut c = LayerKvCache::new(4, 3);
        assert!(c.is_empty());
        c.append(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        c.append(&[7.0, 8.0, 9.0], &[10.0, 11.0, 12.0]);
        assert_eq!(c.len(), 2);
        assert_eq!(c.k_filled(), &[1.0, 2.0, 3.0, 7.0, 8.0, 9.0]);
        assert_eq!(c.v_filled(), &[4.0, 5.0, 6.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn reset_returns_empty() {
        let mut c = LayerKvCache::new(4, 2);
        c.append(&[1.0, 2.0], &[3.0, 4.0]);
        c.reset();
        assert!(c.is_empty());
        assert_eq!(c.k_filled(), &[] as &[f32]);
    }

    #[test]
    #[should_panic(expected = "kv cache full")]
    fn overflow_panics() {
        let mut c = LayerKvCache::new(1, 2);
        c.append(&[1.0, 2.0], &[3.0, 4.0]);
        c.append(&[5.0, 6.0], &[7.0, 8.0]); // overflow
    }

    #[test]
    fn multi_layer_cache_independent() {
        let mut c = KvCache::new(3, 4, 2);
        assert_eq!(c.n_layers(), 3);
        c.layer_mut(0).append(&[1.0, 2.0], &[3.0, 4.0]);
        c.layer_mut(1).append(&[5.0, 6.0], &[7.0, 8.0]);
        assert_eq!(c.layer(0).len(), 1);
        assert_eq!(c.layer(1).len(), 1);
        assert_eq!(c.layer(2).len(), 0);
    }
}
