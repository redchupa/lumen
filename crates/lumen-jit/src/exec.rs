//! Allocate W^X executable memory and turn `MachineCode` into a function pointer.
//! Phase 4 work. Currently a stub returning an error.

use lumen_codegen::MachineCode;

#[derive(thiserror::Error, Debug)]
pub enum ExecError {
    #[error("executable memory allocation not implemented yet")]
    NotImplemented,
}

/// Owns a region of executable memory. Drop unmaps it.
pub struct ExecRegion {
    _bytes: usize,
}

impl ExecRegion {
    /// Phase 4 will:
    ///   1. mmap with PROT_READ | PROT_WRITE
    ///   2. memcpy bytes
    ///   3. mprotect to PROT_READ | PROT_EXEC (W^X)
    ///   4. icache flush on ARM
    pub fn from_machine_code(_code: &MachineCode) -> Result<Self, ExecError> {
        Err(ExecError::NotImplemented)
    }
}
