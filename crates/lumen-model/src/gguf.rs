//! GGUF v3 reader. Phase 5 work — currently a stub.
//!
//! Spec: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

#[derive(thiserror::Error, Debug)]
pub enum GgufError {
    #[error("not implemented yet")]
    NotImplemented,
}

pub struct GgufFile;

impl GgufFile {
    pub fn open(_path: &str) -> Result<Self, GgufError> {
        Err(GgufError::NotImplemented)
    }
}
