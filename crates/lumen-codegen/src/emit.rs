//! Byte-level machine code emitter. Used by all CPU backends.

/// Append-only byte buffer with patch points.
#[derive(Default)]
pub struct Emitter {
    pub buf: Vec<u8>,
}

impl Emitter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn u8(&mut self, b: u8) {
        self.buf.push(b);
    }

    pub fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn bytes(&mut self, s: &[u8]) {
        self.buf.extend_from_slice(s);
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn patch_u32(&mut self, offset: usize, value: u32) {
        self.buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
}
