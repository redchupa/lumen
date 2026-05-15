//! x86_64 backend. Emits native machine code for Lumen IR.
//!
//! Phase 2.B scope: only the `matmul(Param, Param) -> Return` pattern with
//! rank-2 fp32 tensors. Triple loop, scalar `mulss`/`addss`. No SIMD, no
//! tiling — those land in Phase 3.
//!
//! Calling convention support:
//!   - Windows x64 (`extern "C"` on `target_os = "windows"`):
//!     RCX=p0, RDX=p1, R8=p2.
//!   - System V x86_64 (Linux / macOS):
//!     RDI=p0, RSI=p1, RDX=p2.
//!
//! Layout assumption: row-major, fp32, contiguous. Buffers are caller-owned.

use crate::avx2_enc::{vbroadcastss, vfmadd231ps_mem, vmovups_store, vxorps_zero, Ymm};
use crate::backend::{Backend, Capabilities, CodegenError, CodegenOpts, MachineCode};
use crate::emit::Emitter;
use crate::x86_64_enc::*;
use lumen_ir::ty::{DType, Dim, TensorType};
use lumen_ir::{Function, IrModule, Op, ValueId};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Abi {
    /// rcx, rdx, r8, r9
    WinX64,
    /// rdi, rsi, rdx, rcx
    SysV,
}

impl Abi {
    pub fn detect_host() -> Self {
        #[cfg(target_os = "windows")]
        {
            Abi::WinX64
        }
        #[cfg(not(target_os = "windows"))]
        {
            Abi::SysV
        }
    }

    pub fn param_reg(self, index: u32) -> Reg {
        match (self, index) {
            (Abi::WinX64, 0) => Reg::RCX,
            (Abi::WinX64, 1) => Reg::RDX,
            (Abi::WinX64, 2) => Reg::R8,
            (Abi::WinX64, 3) => Reg::R9,
            (Abi::SysV, 0) => Reg::RDI,
            (Abi::SysV, 1) => Reg::RSI,
            (Abi::SysV, 2) => Reg::RDX,
            (Abi::SysV, 3) => Reg::RCX,
            _ => panic!("more than 4 register params not supported"),
        }
    }
}

pub struct X86_64 {
    pub abi: Abi,
    /// If true, the backend may select AVX2/FMA paths when shapes allow.
    /// Default: enabled. Set to false to force scalar codegen for testing.
    pub avx2: bool,
}

impl X86_64 {
    pub fn host() -> Self {
        Self {
            abi: Abi::detect_host(),
            avx2: true,
        }
    }
    pub fn scalar_only() -> Self {
        Self {
            abi: Abi::detect_host(),
            avx2: false,
        }
    }
    pub fn with_abi(abi: Abi) -> Self {
        Self { abi, avx2: true }
    }
}

impl Backend for X86_64 {
    fn name(&self) -> &'static str {
        "x86_64"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    fn lower(&self, ir: &IrModule, _opts: &CodegenOpts) -> Result<MachineCode, CodegenError> {
        let f = ir.functions.first().ok_or(CodegenError::UnsupportedOp(
            "module has no functions".into(),
        ))?;

        let mut em = Emitter::new();
        emit_function(&mut em, f, self.abi, self.avx2)?;
        Ok(MachineCode {
            bytes: em.buf,
            entry_offset: 0,
        })
    }
}

// ============================================================================
// Function emission
// ============================================================================

/// Emit one function. Phase 2.B requires the body to be exactly
/// `Param, Param, MatMul, Return` (in any order matching that shape).
fn emit_function(em: &mut Emitter, f: &Function, abi: Abi, avx2: bool) -> Result<(), CodegenError> {
    // Identify the matmul op and its two operand params.
    let mut matmul_idx: Option<usize> = None;
    let mut return_src: Option<ValueId> = None;
    for (i, v) in f.values.iter().enumerate() {
        match &v.op {
            Op::MatMul { .. } => matmul_idx = Some(i),
            Op::Return { value } => return_src = Some(*value),
            Op::Param { .. } => {}
            _ => {
                return Err(CodegenError::UnsupportedOp(format!(
                    "x86_64 Phase 2.B: op {:?} not supported",
                    v.op
                )));
            }
        }
    }
    let matmul_idx = matmul_idx.ok_or(CodegenError::UnsupportedOp(
        "Phase 2.B requires a matmul function".into(),
    ))?;
    let return_src = return_src.ok_or(CodegenError::UnsupportedOp("missing return".into()))?;
    if return_src.0 as usize != matmul_idx {
        return Err(CodegenError::UnsupportedOp(
            "Phase 2.B: return must come directly from matmul".into(),
        ));
    }

    // Extract M, K, N from the operand shapes.
    let (lhs, rhs) = match &f.values[matmul_idx].op {
        Op::MatMul { lhs, rhs } => (*lhs, *rhs),
        _ => unreachable!(),
    };
    let (m, k) = static_2d(&f.values[lhs.0 as usize].ty)?;
    let (_, n) = static_2d(&f.values[rhs.0 as usize].ty)?;

    // Verify dtype.
    if f.values[lhs.0 as usize].ty.dtype != DType::F32
        || f.values[rhs.0 as usize].ty.dtype != DType::F32
    {
        return Err(CodegenError::UnsupportedOp(
            "Phase 2.B: only f32 matmul is supported".into(),
        ));
    }

    // Param ↔ register mapping. p0 / p1 / result.
    let l_param_idx = expect_param(&f.values[lhs.0 as usize].op)?;
    let r_param_idx = expect_param(&f.values[rhs.0 as usize].op)?;
    let p_lhs = abi.param_reg(l_param_idx);
    let p_rhs = abi.param_reg(r_param_idx);
    // Output buffer is implicit param `n_params` (the caller-supplied result).
    let p_out = abi.param_reg(f.params.len() as u32);

    // Automatic codegen synthesis. AVX2 path requires N % 8 == 0 so a full
    // 256-bit ymm tile lines up with the row stride.
    if avx2 && n % 8 == 0 {
        emit_matmul_avx2(em, p_lhs, p_rhs, p_out, m as i32, k as i32, n as i32, abi);
    } else {
        emit_matmul_body(em, p_lhs, p_rhs, p_out, m as i32, k as i32, n as i32, abi);
    }
    Ok(())
}

/// Pseudo-asm of what we emit:
///
/// ```text
///   ; prologue (Win x64): sub rsp, 40   (32 shadow + 8 align)
///   ; SysV: red zone; nothing needed for a leaf without spills.
///   xor r10, r10              ; i = 0
/// i_loop:
///   cmp r10, M
///   jge i_done
///   xor r11, r11              ; j = 0
/// j_loop:
///   cmp r11, N
///   jge j_done
///   xorps xmm0, xmm0          ; acc = 0
///   xor rcx_scratch, rcx_scratch  ; kk = 0
/// k_loop:
///   cmp rcx_scratch, K
///   jge k_done
///   ; load p_lhs[i*K + kk]
///   mov rax, r10
///   imul rax, rax, K
///   add rax, rcx_scratch
///   movss xmm1, [p_lhs + rax*4]
///   ; *= p_rhs[kk*N + j]
///   mov rdx_scratch, rcx_scratch
///   imul rdx_scratch, rdx_scratch, N
///   add rdx_scratch, r11
///   mulss xmm1, [p_rhs + rdx_scratch*4]
///   addss xmm0, xmm1
///   inc rcx_scratch
///   jmp k_loop
/// k_done:
///   ; store p_out[i*N + j] = acc
///   mov rax, r10
///   imul rax, rax, N
///   add rax, r11
///   movss [p_out + rax*4], xmm0
///   inc r11
///   jmp j_loop
/// j_done:
///   inc r10
///   jmp i_loop
/// i_done:
///   ; epilogue: add rsp, 40 (Win) ; ret
/// ```
///
/// Caller-saved scratch we use freely: RAX, RCX, RDX, R10, R11 (Win64 + SysV).
/// Some of these collide with `p_lhs/p_rhs/p_out` on Win64 (where RCX/RDX are
/// param regs). To keep things uniform, we first save the param pointers into
/// callee-saved-ish slots — we just spill them to stack at the prologue and
/// reload as needed. For Phase 2.B simplicity we copy params into R12/R13/R14
/// (callee-saved). That means we MUST save/restore those registers.
#[allow(clippy::too_many_arguments)] // matmul codegen genuinely needs all of these
fn emit_matmul_body(
    em: &mut Emitter,
    p_lhs: Reg,
    p_rhs: Reg,
    p_out: Reg,
    m: i32,
    k: i32,
    n: i32,
    abi: Abi,
) {
    // ---- prologue ----
    // Save the registers we clobber as if they were callee-saved.
    push_r64(em, Reg::R12);
    push_r64(em, Reg::R13);
    push_r64(em, Reg::R14);
    push_r64(em, Reg::R15);
    push_r64(em, Reg::RBX);
    // Stack now: 5*8=40 bytes pushed. On Win64 we also reserve shadow space.
    let extra = if abi == Abi::WinX64 { 40 } else { 8 }; // align to 16
    sub_ri32(em, Reg::RSP, extra);

    // Move param pointers into callee-saved regs so we can use RCX/RDX/R8 etc.
    mov_rr(em, Reg::R12, p_lhs); // R12 = lhs base
    mov_rr(em, Reg::R13, p_rhs); // R13 = rhs base
    mov_rr(em, Reg::R14, p_out); // R14 = out base

    // Loop counters: R15=i, RBX=j, RAX=kk (will be re-zeroed in k_loop entry)
    xor_rr(em, Reg::R15);
    let i_loop = em.len();
    cmp_ri32(em, Reg::R15, m);
    let jge_i = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    xor_rr(em, Reg::RBX);
    let j_loop = em.len();
    cmp_ri32(em, Reg::RBX, n);
    let jge_j = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    xorps_self(em, Xmm(0)); // acc = 0
    xor_rr(em, Reg::RCX); // kk = 0
    let k_loop = em.len();
    cmp_ri32(em, Reg::RCX, k);
    let jge_k = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    // a_idx = i * K + kk  (in RAX)
    mov_rr(em, Reg::RAX, Reg::R15);
    imul_rri32(em, Reg::RAX, Reg::RAX, k);
    add_rr(em, Reg::RAX, Reg::RCX);
    // xmm1 = lhs[RAX]
    movss_load(em, Xmm(1), Reg::R12, Some((Reg::RAX, Scale::S4)), 0);

    // b_idx = kk * N + j (in RDX)
    mov_rr(em, Reg::RDX, Reg::RCX);
    imul_rri32(em, Reg::RDX, Reg::RDX, n);
    add_rr(em, Reg::RDX, Reg::RBX);
    // xmm1 *= rhs[RDX]
    mulss_load(em, Xmm(1), Reg::R13, Some((Reg::RDX, Scale::S4)), 0);

    addss_xmm(em, Xmm(0), Xmm(1));
    inc_r(em, Reg::RCX);
    let jmp_k = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp_k, k_loop);

    // k_done:
    let k_done = em.len();
    patch_rel32(em, jge_k, k_done);

    // out_idx = i*N + j (RAX)
    mov_rr(em, Reg::RAX, Reg::R15);
    imul_rri32(em, Reg::RAX, Reg::RAX, n);
    add_rr(em, Reg::RAX, Reg::RBX);
    movss_store(em, Xmm(0), Reg::R14, Some((Reg::RAX, Scale::S4)), 0);

    inc_r(em, Reg::RBX);
    let jmp_j = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp_j, j_loop);

    // j_done:
    let j_done = em.len();
    patch_rel32(em, jge_j, j_done);

    inc_r(em, Reg::R15);
    let jmp_i = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp_i, i_loop);

    // i_done:
    let i_done = em.len();
    patch_rel32(em, jge_i, i_done);

    // ---- epilogue ----
    add_ri32(em, Reg::RSP, extra);
    pop_r64(em, Reg::RBX);
    pop_r64(em, Reg::R15);
    pop_r64(em, Reg::R14);
    pop_r64(em, Reg::R13);
    pop_r64(em, Reg::R12);
    ret(em);
}

/// AVX2 matmul. Requires `N % 8 == 0`. Processes 8 columns of the output at a
/// time using a single ymm accumulator.
///
/// ```text
/// for i in 0..M:
///   for j in 0..N step 8:
///     ymm_acc = 0
///     for kk in 0..K:
///       ymm0 = broadcast(lhs[i*K + kk])
///       ymm_acc += ymm0 * [rhs + (kk*N + j)*4]   (vfmadd231ps with mem operand)
///     store [out + (i*N + j)*4] = ymm_acc
/// ```
///
/// Same prologue/epilogue and same register convention as `emit_matmul_body`
/// so the two paths are interchangeable from the JIT's point of view.
#[allow(clippy::too_many_arguments)]
fn emit_matmul_avx2(
    em: &mut Emitter,
    p_lhs: Reg,
    p_rhs: Reg,
    p_out: Reg,
    m: i32,
    k: i32,
    n: i32,
    abi: Abi,
) {
    debug_assert!(n % 8 == 0, "AVX2 path requires N % 8 == 0");

    // ---- prologue ----
    push_r64(em, Reg::R12);
    push_r64(em, Reg::R13);
    push_r64(em, Reg::R14);
    push_r64(em, Reg::R15);
    push_r64(em, Reg::RBX);
    let extra = if abi == Abi::WinX64 { 40 } else { 8 };
    sub_ri32(em, Reg::RSP, extra);

    mov_rr(em, Reg::R12, p_lhs);
    mov_rr(em, Reg::R13, p_rhs);
    mov_rr(em, Reg::R14, p_out);

    // R15 = i, RBX = j (column block, stepped by 8), RCX = kk
    xor_rr(em, Reg::R15);
    let i_loop = em.len();
    cmp_ri32(em, Reg::R15, m);
    let jge_i = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    xor_rr(em, Reg::RBX);
    let j_loop = em.len();
    cmp_ri32(em, Reg::RBX, n);
    let jge_j = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    // ymm0 (acc) = 0
    vxorps_zero(em, Ymm(0));

    xor_rr(em, Reg::RCX); // kk = 0
    let k_loop = em.len();
    cmp_ri32(em, Reg::RCX, k);
    let jge_k = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    // ymm1 = broadcast(lhs[i*K + kk])
    // a_addr_index = i * K + kk  (in RAX)
    mov_rr(em, Reg::RAX, Reg::R15);
    imul_rri32(em, Reg::RAX, Reg::RAX, k);
    add_rr(em, Reg::RAX, Reg::RCX);
    vbroadcastss(em, Ymm(1), Reg::R12, Some((Reg::RAX, Scale::S4)), 0);

    // b_addr_index = kk * N + j  (in RDX)
    mov_rr(em, Reg::RDX, Reg::RCX);
    imul_rri32(em, Reg::RDX, Reg::RDX, n);
    add_rr(em, Reg::RDX, Reg::RBX);
    // ymm0 += ymm1 * [rhs + RDX*4]
    vfmadd231ps_mem(em, Ymm(0), Ymm(1), Reg::R13, Some((Reg::RDX, Scale::S4)), 0);

    inc_r(em, Reg::RCX);
    let jmp_k = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp_k, k_loop);

    let k_done = em.len();
    patch_rel32(em, jge_k, k_done);

    // store [out + (i*N + j)*4] = ymm0
    mov_rr(em, Reg::RAX, Reg::R15);
    imul_rri32(em, Reg::RAX, Reg::RAX, n);
    add_rr(em, Reg::RAX, Reg::RBX);
    vmovups_store(em, Ymm(0), Reg::R14, Some((Reg::RAX, Scale::S4)), 0);

    add_ri32(em, Reg::RBX, 8); // j += 8
    let jmp_j = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp_j, j_loop);

    let j_done = em.len();
    patch_rel32(em, jge_j, j_done);

    inc_r(em, Reg::R15);
    let jmp_i = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp_i, i_loop);

    let i_done = em.len();
    patch_rel32(em, jge_i, i_done);

    // epilogue
    add_ri32(em, Reg::RSP, extra);
    pop_r64(em, Reg::RBX);
    pop_r64(em, Reg::R15);
    pop_r64(em, Reg::R14);
    pop_r64(em, Reg::R13);
    pop_r64(em, Reg::R12);
    ret(em);
}

// Minimal push/pop encodings (the encoder module focuses on matmul ops; these
// are short, so we inline them here).
fn push_r64(em: &mut Emitter, r: Reg) {
    if r.high1() != 0 {
        em.u8(0x41); // REX.B
    }
    em.u8(0x50 + r.low3());
}

fn pop_r64(em: &mut Emitter, r: Reg) {
    if r.high1() != 0 {
        em.u8(0x41);
    }
    em.u8(0x58 + r.low3());
}

fn static_2d(t: &TensorType) -> Result<(u32, u32), CodegenError> {
    if t.shape.0.len() != 2 {
        return Err(CodegenError::UnsupportedOp(
            "matmul operand not rank-2".into(),
        ));
    }
    match (&t.shape.0[0], &t.shape.0[1]) {
        (Dim::Static(a), Dim::Static(b)) => Ok((*a, *b)),
        _ => Err(CodegenError::ShapeError(
            "dynamic dims not supported".into(),
        )),
    }
}

fn expect_param(op: &Op) -> Result<u32, CodegenError> {
    match op {
        Op::Param { index } => Ok(*index),
        _ => Err(CodegenError::UnsupportedOp(
            "Phase 2.B: matmul operands must be raw function parameters".into(),
        )),
    }
}

// ============================================================================
// Tests — emit ret-only for a tiny synthetic IR; full e2e lives in
// `tests/e2e_native_matmul.rs`.
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_param_regs() {
        assert_eq!(Abi::WinX64.param_reg(0), Reg::RCX);
        assert_eq!(Abi::WinX64.param_reg(1), Reg::RDX);
        assert_eq!(Abi::WinX64.param_reg(2), Reg::R8);
        assert_eq!(Abi::SysV.param_reg(0), Reg::RDI);
        assert_eq!(Abi::SysV.param_reg(1), Reg::RSI);
        assert_eq!(Abi::SysV.param_reg(2), Reg::RDX);
    }
}
