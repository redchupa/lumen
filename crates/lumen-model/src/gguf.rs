//! GGUF v3 reader.
//!
//! Spec: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>
//!
//! ## File layout
//!
//! ```text
//! magic      : 4 bytes "GGUF"
//! version    : u32 LE   (= 3)
//! n_tensors  : u64 LE
//! n_kv       : u64 LE
//!
//! kv_pairs   : KvPair × n_kv
//!   key      : GgufString   { len: u64 LE, bytes: [u8; len] }
//!   value_t  : u32 LE       (KvType)
//!   value    : payload depending on value_t
//!
//! tensors    : TensorInfo × n_tensors
//!   name     : GgufString
//!   n_dims   : u32 LE
//!   dims     : [u64 LE; n_dims]
//!   tensor_t : u32 LE       (GgmlType)
//!   offset   : u64 LE       (byte offset from the start of the data section)
//!
//! padding    : align to `general.alignment` (default 32)
//! data       : tensor payload bytes
//! ```
//!
//! Phase 5.D scope: parse the header, key-value pairs, and tensor table;
//! expose a slice of each tensor's raw bytes via [`GgufFile::tensor_data`].
//! No mmap yet — the whole file is loaded into a `Vec<u8>`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

const GGUF_MAGIC: &[u8] = b"GGUF";

/// KV value type tag, exactly as the GGUF spec defines.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum KvType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl KvType {
    fn from_u32(v: u32) -> Result<Self, GgufError> {
        Ok(match v {
            0 => KvType::U8,
            1 => KvType::I8,
            2 => KvType::U16,
            3 => KvType::I16,
            4 => KvType::U32,
            5 => KvType::I32,
            6 => KvType::F32,
            7 => KvType::Bool,
            8 => KvType::String,
            9 => KvType::Array,
            10 => KvType::U64,
            11 => KvType::I64,
            12 => KvType::F64,
            other => return Err(GgufError::BadKvType(other)),
        })
    }
}

/// Tensor dtype tag (subset of ggml's enum that Lumen cares about).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
}

impl GgmlType {
    fn from_u32(v: u32) -> Result<Self, GgufError> {
        Ok(match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            other => return Err(GgufError::UnsupportedTensorType(other)),
        })
    }

    /// Bytes for one storage unit (block for quantized formats, element for the
    /// scalar floats). Mirrors ggml's `ggml_type_size`.
    pub fn type_size(self) -> u64 {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
            GgmlType::Q4_0 | GgmlType::Q4_1 => 18,
            GgmlType::Q5_0 | GgmlType::Q5_1 => 22,
            GgmlType::Q8_0 | GgmlType::Q8_1 => 34,
        }
    }

    /// Elements per storage unit. 1 for unquantized scalars, 32 for the K-quants.
    pub fn block_size(self) -> u64 {
        match self {
            GgmlType::F32 | GgmlType::F16 => 1,
            _ => 32,
        }
    }
}

/// One key-value entry. Arrays are flattened into a single owned `Vec<u8>`
/// holding the raw bytes plus the element type tag.
#[derive(Clone, Debug)]
pub enum KvValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    /// `(element_type, raw_bytes)` — caller decodes based on element type.
    /// Element type is never `Array` (spec forbids nested).
    Array(KvType, Vec<u8>),
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: GgmlType,
    pub offset: u64,
}

impl TensorInfo {
    pub fn element_count(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Byte size of this tensor's payload in the GGUF data section.
    /// Mirrors ggml's `ggml_row_size(t, nelem)`.
    pub fn byte_size(&self) -> u64 {
        let nelem = self.element_count();
        nelem * self.dtype.type_size() / self.dtype.block_size()
    }
}

#[derive(thiserror::Error, Debug)]
pub enum GgufError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("truncated file: needed {needed} bytes at offset {at}, got {available}")]
    Truncated {
        needed: usize,
        at: usize,
        available: usize,
    },
    #[error("bad magic — expected `GGUF`, got {0:?}")]
    BadMagic([u8; 4]),
    #[error("unsupported version {0} (this build supports v3)")]
    UnsupportedVersion(u32),
    #[error("bad KV type tag {0}")]
    BadKvType(u32),
    #[error("unsupported tensor type tag {0}")]
    UnsupportedTensorType(u32),
    #[error("invalid utf-8 in GGUF string at offset {0}")]
    BadString(usize),
    #[error("tensor `{0}` not found")]
    NoSuchTensor(String),
}

/// A parsed GGUF file.
#[derive(Debug)]
pub struct GgufFile {
    bytes: Vec<u8>,
    metadata: HashMap<String, KvValue>,
    tensors: Vec<TensorInfo>,
    tensor_by_name: HashMap<String, usize>,
    data_offset: usize,
    alignment: u64,
}

impl GgufFile {
    /// Read and parse a GGUF file from disk.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let bytes = fs::read(path)?;
        Self::from_bytes(bytes)
    }

    /// Parse an in-memory GGUF buffer.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, GgufError> {
        let mut p = Parser::new(&bytes);
        // Header.
        let mut magic = [0u8; 4];
        magic.copy_from_slice(p.take(4)?);
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic(magic));
        }
        let version = p.u32()?;
        if version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let n_tensors = p.u64()? as usize;
        let n_kv = p.u64()? as usize;

        // Key-value pairs.
        let mut metadata = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = p.string()?;
            let tag = KvType::from_u32(p.u32()?)?;
            let value = p.kv_value(tag)?;
            metadata.insert(key, value);
        }

        // Tensor info table.
        let mut tensors = Vec::with_capacity(n_tensors);
        let mut tensor_by_name = HashMap::with_capacity(n_tensors);
        for idx in 0..n_tensors {
            let name = p.string()?;
            let n_dims = p.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(p.u64()?);
            }
            let dtype = GgmlType::from_u32(p.u32()?)?;
            let offset = p.u64()?;
            tensor_by_name.insert(name.clone(), idx);
            tensors.push(TensorInfo {
                name,
                dims,
                dtype,
                offset,
            });
        }

        // Padding to alignment.
        let alignment: u64 = metadata
            .get("general.alignment")
            .and_then(|v| match v {
                KvValue::U32(x) => Some(*x as u64),
                KvValue::U64(x) => Some(*x),
                _ => None,
            })
            .unwrap_or(32);
        let here = p.pos();
        let padded = (here as u64).div_ceil(alignment) * alignment;
        p.skip((padded as usize) - here)?;
        let data_offset = p.pos();

        Ok(Self {
            bytes,
            metadata,
            tensors,
            tensor_by_name,
            data_offset,
            alignment,
        })
    }

    pub fn metadata(&self) -> &HashMap<String, KvValue> {
        &self.metadata
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_by_name.get(name).map(|&i| &self.tensors[i])
    }

    /// Borrow the raw bytes of one tensor. Length matches `info.byte_size()`.
    pub fn tensor_data(&self, name: &str) -> Result<&[u8], GgufError> {
        let info = self
            .tensor(name)
            .ok_or_else(|| GgufError::NoSuchTensor(name.to_string()))?;
        let start = self.data_offset + info.offset as usize;
        let len = info.byte_size() as usize;
        let end = start + len;
        if end > self.bytes.len() {
            return Err(GgufError::Truncated {
                needed: end - start,
                at: start,
                available: self.bytes.len().saturating_sub(start),
            });
        }
        Ok(&self.bytes[start..end])
    }
}

// ============================================================================
// Parser — internal byte cursor over a `&[u8]`.
// ============================================================================

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a [u8]) -> Self {
        Self { src, pos: 0 }
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        if self.pos + n > self.src.len() {
            return Err(GgufError::Truncated {
                needed: n,
                at: self.pos,
                available: self.src.len() - self.pos,
            });
        }
        let s = &self.src[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn skip(&mut self, n: usize) -> Result<(), GgufError> {
        let _ = self.take(n)?;
        Ok(())
    }

    fn u8_(&mut self) -> Result<u8, GgufError> {
        Ok(self.take(1)?[0])
    }

    fn i8_(&mut self) -> Result<i8, GgufError> {
        Ok(self.take(1)?[0] as i8)
    }

    fn u16(&mut self) -> Result<u16, GgufError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn i16(&mut self) -> Result<i16, GgufError> {
        Ok(i16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn i32(&mut self) -> Result<i32, GgufError> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn i64(&mut self) -> Result<i64, GgufError> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn f64(&mut self) -> Result<f64, GgufError> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String, GgufError> {
        let len = self.u64()? as usize;
        let start = self.pos;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| GgufError::BadString(start))
    }

    fn kv_value(&mut self, tag: KvType) -> Result<KvValue, GgufError> {
        Ok(match tag {
            KvType::U8 => KvValue::U8(self.u8_()?),
            KvType::I8 => KvValue::I8(self.i8_()?),
            KvType::U16 => KvValue::U16(self.u16()?),
            KvType::I16 => KvValue::I16(self.i16()?),
            KvType::U32 => KvValue::U32(self.u32()?),
            KvType::I32 => KvValue::I32(self.i32()?),
            KvType::F32 => KvValue::F32(self.f32()?),
            KvType::U64 => KvValue::U64(self.u64()?),
            KvType::I64 => KvValue::I64(self.i64()?),
            KvType::F64 => KvValue::F64(self.f64()?),
            KvType::Bool => KvValue::Bool(self.u8_()? != 0),
            KvType::String => KvValue::String(self.string()?),
            KvType::Array => {
                let elem_tag = KvType::from_u32(self.u32()?)?;
                let len = self.u64()? as usize;
                // We don't decode each element here — just record the raw payload
                // so callers that know the schema can interpret it later. Arrays
                // of strings have variable-length elements, so we still need to
                // walk them to compute the right slice length.
                let start = self.pos;
                for _ in 0..len {
                    self.skip_kv_payload(elem_tag)?;
                }
                let end = self.pos;
                let raw = self.src[start..end].to_vec();
                KvValue::Array(elem_tag, raw)
            }
        })
    }

    fn skip_kv_payload(&mut self, tag: KvType) -> Result<(), GgufError> {
        match tag {
            KvType::U8 | KvType::I8 | KvType::Bool => self.skip(1),
            KvType::U16 | KvType::I16 => self.skip(2),
            KvType::U32 | KvType::I32 | KvType::F32 => self.skip(4),
            KvType::U64 | KvType::I64 | KvType::F64 => self.skip(8),
            KvType::String => {
                let n = self.u64()? as usize;
                self.skip(n)
            }
            KvType::Array => Err(GgufError::BadKvType(KvType::Array as u32)),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid GGUF v3 buffer by hand for round-trip testing.
    fn build_toy_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(GGUF_MAGIC);
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&2u64.to_le_bytes()); // n_kv

        // KV 1: "general.architecture" → "lumen.test"
        write_string(&mut buf, "general.architecture");
        buf.extend_from_slice(&(KvType::String as u32).to_le_bytes());
        write_string(&mut buf, "lumen.test");

        // KV 2: "general.alignment" → u32 32
        write_string(&mut buf, "general.alignment");
        buf.extend_from_slice(&(KvType::U32 as u32).to_le_bytes());
        buf.extend_from_slice(&32u32.to_le_bytes());

        // Tensor info: "weight" shape=[4, 8] dtype=F32 offset=0
        write_string(&mut buf, "weight");
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&4u64.to_le_bytes()); // dim 0
        buf.extend_from_slice(&8u64.to_le_bytes()); // dim 1
        buf.extend_from_slice(&(GgmlType::F32 as u32).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        // Pad to alignment 32.
        let pad = (32 - (buf.len() % 32)) % 32;
        buf.resize(buf.len() + pad, 0);

        // Tensor data: 4 × 8 = 32 floats.
        for i in 0..32u32 {
            buf.extend_from_slice(&(i as f32).to_le_bytes());
        }
        buf
    }

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        let bytes = s.as_bytes();
        buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(bytes);
    }

    #[test]
    fn parses_toy_header_and_metadata() {
        let buf = build_toy_gguf();
        let f = GgufFile::from_bytes(buf).expect("parse");
        assert_eq!(f.tensors().len(), 1);
        assert_eq!(f.alignment(), 32);
        let arch = f.metadata().get("general.architecture");
        assert!(matches!(arch, Some(KvValue::String(s)) if s == "lumen.test"));
    }

    #[test]
    fn tensor_metadata_is_correct() {
        let buf = build_toy_gguf();
        let f = GgufFile::from_bytes(buf).expect("parse");
        let info = f.tensor("weight").expect("weight");
        assert_eq!(info.dims, vec![4, 8]);
        assert_eq!(info.dtype, GgmlType::F32);
        assert_eq!(info.byte_size(), 4 * 8 * 4);
    }

    #[test]
    fn tensor_data_round_trip() {
        let buf = build_toy_gguf();
        let f = GgufFile::from_bytes(buf).expect("parse");
        let data = f.tensor_data("weight").expect("data");
        assert_eq!(data.len(), 32 * 4);
        for (i, chunk) in data.chunks_exact(4).enumerate() {
            let v = f32::from_le_bytes(chunk.try_into().unwrap());
            assert_eq!(v, i as f32);
        }
    }

    #[test]
    fn rejects_bad_magic() {
        let mut buf = build_toy_gguf();
        buf[0] = b'X';
        let err = GgufFile::from_bytes(buf).unwrap_err();
        assert!(matches!(err, GgufError::BadMagic(_)));
    }

    #[test]
    fn rejects_wrong_version() {
        let buf = build_toy_gguf();
        let mut buf = buf;
        buf[4..8].copy_from_slice(&99u32.to_le_bytes());
        let err = GgufFile::from_bytes(buf).unwrap_err();
        assert!(matches!(err, GgufError::UnsupportedVersion(99)));
    }
}
