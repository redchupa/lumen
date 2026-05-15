//! Executable memory regions.
//!
//! Copies raw bytes into a freshly allocated page, flips its permissions from
//! `rw-` to `r-x`, and hands back something callable. Drop unmaps the region.
//!
//! Two implementations:
//!   - `cfg(unix)`     — `mmap` + `mprotect`.
//!   - `cfg(windows)`  — `VirtualAlloc` + `VirtualProtect`.

use lumen_codegen::MachineCode;

#[derive(thiserror::Error, Debug)]
pub enum ExecError {
    #[error("allocation failed (errno = {0})")]
    Alloc(i32),
    #[error("permission change failed (errno = {0})")]
    Protect(i32),
}

/// Owns a region of executable memory. Drop unmaps it.
pub struct ExecRegion {
    ptr: *mut u8,
    len: usize,
    entry_offset: usize,
}

// The region is owned and not shared across threads.
unsafe impl Send for ExecRegion {}

impl ExecRegion {
    /// Allocate a W^X executable region and copy `code.bytes` into it. The
    /// function entry is at `code.entry_offset` from the start of the region.
    ///
    /// Steps:
    ///   1. Allocate page-aligned memory with RW.
    ///   2. memcpy bytes.
    ///   3. Re-protect to RX (W^X discipline).
    ///   4. Flush the instruction cache on architectures that need it.
    pub fn from_machine_code(code: &MachineCode) -> Result<Self, ExecError> {
        let len = code.bytes.len().max(1);
        let ptr = alloc_rw(len)?;
        // SAFETY: `ptr` points to `len` writable bytes we just allocated and
        // `code.bytes` has length <= `len`.
        unsafe {
            std::ptr::copy_nonoverlapping(code.bytes.as_ptr(), ptr, code.bytes.len());
        }
        make_executable(ptr, len)?;
        flush_icache(ptr, len);
        Ok(Self {
            ptr,
            len,
            entry_offset: code.entry_offset,
        })
    }

    /// Cast the entry point into a function pointer of type `F`.
    ///
    /// # Safety
    /// The caller asserts that `F` is the exact signature emitted by the
    /// backend and that the function obeys the platform calling convention.
    pub unsafe fn as_fn<F: Copy>(&self) -> F {
        debug_assert_eq!(std::mem::size_of::<F>(), std::mem::size_of::<*const u8>());
        let entry = unsafe { self.ptr.add(self.entry_offset) };
        // SAFETY: caller asserts the signature matches.
        unsafe { std::mem::transmute_copy::<*const u8, F>(&(entry as *const u8)) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }
}

impl Drop for ExecRegion {
    fn drop(&mut self) {
        // SAFETY: we own this region; ignore failures during teardown.
        unsafe {
            free_region(self.ptr, self.len);
        }
    }
}

// ============================================================================
// Unix backend
// ============================================================================

#[cfg(unix)]
mod imp {
    use super::ExecError;

    // libc signatures we need. Declared inline to avoid a dependency.
    extern "C" {
        fn mmap(
            addr: *mut libc_void,
            length: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut libc_void;
        fn mprotect(addr: *mut libc_void, length: usize, prot: i32) -> i32;
        fn munmap(addr: *mut libc_void, length: usize) -> i32;
        fn __errno_location() -> *mut i32;
    }
    #[allow(non_camel_case_types)]
    pub type libc_void = std::ffi::c_void;

    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const PROT_EXEC: i32 = 4;
    const MAP_PRIVATE: i32 = 2;
    #[cfg(target_os = "linux")]
    const MAP_ANONYMOUS: i32 = 0x20;
    #[cfg(target_os = "macos")]
    const MAP_ANONYMOUS: i32 = 0x1000;
    const MAP_FAILED: *mut libc_void = !0usize as *mut libc_void;

    pub fn alloc_rw(len: usize) -> Result<*mut u8, ExecError> {
        // SAFETY: standard mmap call; MAP_FAILED check handles failure.
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr == MAP_FAILED {
            // SAFETY: errno is thread-local on Linux/macOS.
            let err = unsafe { *__errno_location() };
            return Err(ExecError::Alloc(err));
        }
        Ok(ptr.cast())
    }

    pub fn make_executable(ptr: *mut u8, len: usize) -> Result<(), ExecError> {
        // SAFETY: we own `ptr..ptr+len`.
        let rc = unsafe { mprotect(ptr.cast(), len, PROT_READ | PROT_EXEC) };
        if rc != 0 {
            let err = unsafe { *__errno_location() };
            return Err(ExecError::Protect(err));
        }
        Ok(())
    }

    pub fn flush_icache(_ptr: *mut u8, _len: usize) {
        // On x86_64 the instruction cache is coherent with stores from the same core.
        // On ARM64 we'd issue `dc cvau` + `ic ivau` + `dsb ish`.
    }

    pub unsafe fn free_region(ptr: *mut u8, len: usize) {
        let _ = unsafe { munmap(ptr.cast(), len) };
    }
}

// ============================================================================
// Windows backend
// ============================================================================

#[cfg(windows)]
#[allow(clippy::upper_case_acronyms)] // Win32 API type names are canonically all-caps
mod imp {
    use super::ExecError;

    type LPVOID = *mut std::ffi::c_void;
    #[allow(non_camel_case_types)]
    type SIZE_T = usize;
    type DWORD = u32;
    type BOOL = i32;
    type HANDLE = *mut std::ffi::c_void;
    type PDWORD = *mut DWORD;

    const MEM_COMMIT: DWORD = 0x1000;
    const MEM_RESERVE: DWORD = 0x2000;
    const MEM_RELEASE: DWORD = 0x8000;
    const PAGE_READWRITE: DWORD = 0x04;
    const PAGE_EXECUTE_READ: DWORD = 0x20;

    extern "system" {
        fn VirtualAlloc(
            lpAddress: LPVOID,
            dwSize: SIZE_T,
            flAllocationType: DWORD,
            flProtect: DWORD,
        ) -> LPVOID;
        fn VirtualProtect(
            lpAddress: LPVOID,
            dwSize: SIZE_T,
            flNewProtect: DWORD,
            lpflOldProtect: PDWORD,
        ) -> BOOL;
        fn VirtualFree(lpAddress: LPVOID, dwSize: SIZE_T, dwFreeType: DWORD) -> BOOL;
        fn GetCurrentProcess() -> HANDLE;
        fn FlushInstructionCache(hProcess: HANDLE, lpBaseAddress: LPVOID, dwSize: SIZE_T) -> BOOL;
        fn GetLastError() -> DWORD;
    }

    pub fn alloc_rw(len: usize) -> Result<*mut u8, ExecError> {
        // SAFETY: VirtualAlloc with a NULL address picks any address.
        let ptr = unsafe {
            VirtualAlloc(
                std::ptr::null_mut(),
                len,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
        };
        if ptr.is_null() {
            let err = unsafe { GetLastError() } as i32;
            return Err(ExecError::Alloc(err));
        }
        Ok(ptr.cast())
    }

    pub fn make_executable(ptr: *mut u8, len: usize) -> Result<(), ExecError> {
        let mut old: DWORD = 0;
        let rc = unsafe { VirtualProtect(ptr.cast(), len, PAGE_EXECUTE_READ, &mut old) };
        if rc == 0 {
            let err = unsafe { GetLastError() } as i32;
            return Err(ExecError::Protect(err));
        }
        Ok(())
    }

    pub fn flush_icache(ptr: *mut u8, len: usize) {
        // SAFETY: documented Win32 API.
        unsafe {
            let _ = FlushInstructionCache(GetCurrentProcess(), ptr.cast(), len);
        }
    }

    pub unsafe fn free_region(ptr: *mut u8, _len: usize) {
        unsafe {
            let _ = VirtualFree(ptr.cast(), 0, MEM_RELEASE);
        }
    }
}

use imp::*;

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_codegen::MachineCode;

    /// Smoke test: emit a 1-byte `ret` and call it as `fn()`.
    #[test]
    fn alloc_protect_call_ret_only() {
        let code = MachineCode {
            bytes: vec![0xC3],
            entry_offset: 0,
        };
        let region = ExecRegion::from_machine_code(&code).expect("alloc");
        // SAFETY: `ret` matches an empty `extern "C" fn()`.
        unsafe {
            let f: unsafe extern "C" fn() = region.as_fn();
            f();
        }
    }
}
