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

use crate::avx2_enc::{
    vbroadcastss, vbroadcastss_xmm, vcvtdq2ps, vcvtph2ps_xmm, vfmadd231ps_mem, vfmadd231ps_reg,
    vmovd_load, vmovups_load, vmovups_store, vmulps_reg, vpmovsxbd_load, vxorps_zero, Ymm,
};
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

/// Emit one function. Two patterns supported:
/// - matmul:    `Param, Param, MatMul, Return`           (Phase 2.B/3)
/// - dequant:   `Param, Dequantize, Return`              (Phase 5.B, Q8_0 only)
fn emit_function(em: &mut Emitter, f: &Function, abi: Abi, avx2: bool) -> Result<(), CodegenError> {
    // Try dequantize pattern first: a single Dequantize op whose source is a Param.
    if let Some(dq_idx) = f
        .values
        .iter()
        .position(|v| matches!(v.op, Op::Dequantize { .. }))
    {
        return emit_function_dequant(em, f, dq_idx, abi);
    }

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

    // Automatic codegen synthesis — tile size selection.
    // 1. 4×8 register tile when both M and N divide nicely (best throughput).
    // 2. 1×8 AVX2 vectorization when only N divides.
    // 3. Scalar fallback otherwise.
    if avx2 && m % 4 == 0 && n % 8 == 0 {
        emit_matmul_tile_4x8(em, p_lhs, p_rhs, p_out, m as i32, k as i32, n as i32, abi);
    } else if avx2 && n % 8 == 0 {
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

/// Emit a Q8_0 dequantize function.
///
/// Source IR: `Param(q8_0, [N])` → `Dequantize` → `Return`.
///
/// Input pointer (`p0`) addresses the start of `block_q8_0` blocks (18 bytes
/// per block: 2 bytes fp16 scale + ... wait, Q8_0 is 34 bytes: fp16 d + 32 i8).
/// Output pointer (`p1`) is the destination fp32 buffer.
///
/// Generated pseudo-asm (per block):
/// ```text
///   ; Load fp16 d from [r12 + i*34 + 0], convert to fp32, broadcast.
///   vmovd        xmm1, [r12 + i*34]
///   vcvtph2ps    xmm1, xmm1
///   vbroadcastss ymm0, xmm1
///   ; For each of 4 groups of 8:
///   vpmovsxbd    ymm1, [r12 + i*34 + 2 + g*8]   ; 8 i8 → 8 i32
///   vcvtdq2ps    ymm1, ymm1                     ; 8 i32 → 8 f32
///   vmulps       ymm1, ymm1, ymm0               ; * d
///   vmovups      [r13 + i*128 + g*32], ymm1     ; store 8 f32
/// ```
fn emit_function_dequant(
    em: &mut Emitter,
    f: &Function,
    dq_idx: usize,
    abi: Abi,
) -> Result<(), CodegenError> {
    use lumen_ir::ty::DType;

    let dq = &f.values[dq_idx];
    let Op::Dequantize { x } = &dq.op else {
        unreachable!()
    };
    let src_ty = &f.values[x.0 as usize].ty;
    if src_ty.dtype != DType::Q8_0 {
        return Err(CodegenError::UnsupportedOp(
            "x86_64 Phase 5.B: only Q8_0 dequantize is supported".into(),
        ));
    }
    if dq.ty.dtype != DType::F32 {
        return Err(CodegenError::UnsupportedOp(
            "Q8_0 dequant must produce f32 output".into(),
        ));
    }
    // We require x = Param(0). Output buffer is implicit param 1.
    let src_param = match &f.values[x.0 as usize].op {
        Op::Param { index } => *index,
        _ => {
            return Err(CodegenError::UnsupportedOp(
                "Phase 5.B: dequantize operand must be a Param".into(),
            ));
        }
    };

    // Total elements = product of shape dims.
    let n_elems: u32 = src_ty
        .shape
        .0
        .iter()
        .map(|d| match d {
            Dim::Static(v) => *v,
            Dim::Dynamic(_) => 0,
        })
        .product();
    if n_elems % 32 != 0 {
        return Err(CodegenError::ShapeError(
            "Q8_0 dequant requires element count divisible by 32".into(),
        ));
    }
    let num_blocks = (n_elems / 32) as i32;

    let p_src = abi.param_reg(src_param);
    let p_dst = abi.param_reg(f.params.len() as u32);

    emit_dequant_q8_body(em, p_src, p_dst, num_blocks, abi);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_dequant_q8_body(em: &mut Emitter, p_src: Reg, p_dst: Reg, num_blocks: i32, abi: Abi) {
    // ---- prologue ----
    push_r64(em, Reg::R12);
    push_r64(em, Reg::R13);
    push_r64(em, Reg::RBX);
    let extra = if abi == Abi::WinX64 { 40 } else { 8 };
    sub_ri32(em, Reg::RSP, extra);

    mov_rr(em, Reg::R12, p_src);
    mov_rr(em, Reg::R13, p_dst);

    // RBX = i (block index)
    xor_rr(em, Reg::RBX);
    let loop_start = em.len();
    cmp_ri32(em, Reg::RBX, num_blocks);
    let jge_end = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    // RAX = i * 34 (block byte offset within source)
    mov_rr(em, Reg::RAX, Reg::RBX);
    imul_rri32(em, Reg::RAX, Reg::RAX, 34);

    // ymm0 = broadcast(fp32(fp16 d at [R12 + RAX]))
    vmovd_load(em, Ymm(1), Reg::R12, Some((Reg::RAX, Scale::S1)), 0);
    vcvtph2ps_xmm(em, Ymm(1), Ymm(1));
    vbroadcastss_xmm(em, Ymm(0), Ymm(1));

    // RCX = i * 128 (dest byte offset = i * 32 * 4)
    mov_rr(em, Reg::RCX, Reg::RBX);
    imul_rri32(em, Reg::RCX, Reg::RCX, 128);

    // 4 groups of 8 elements each.
    for g in 0..4i32 {
        let qs_offset = 2 + g * 8; // 2-byte d + g*8 i8s
        vpmovsxbd_load(em, Ymm(1), Reg::R12, Some((Reg::RAX, Scale::S1)), qs_offset);
        vcvtdq2ps(em, Ymm(1), Ymm(1));
        vmulps_reg(em, Ymm(1), Ymm(1), Ymm(0));
        vmovups_store(em, Ymm(1), Reg::R13, Some((Reg::RCX, Scale::S1)), g * 32);
    }

    inc_r(em, Reg::RBX);
    let jmp_back = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp_back, loop_start);

    let end = em.len();
    patch_rel32(em, jge_end, end);

    // epilogue
    add_ri32(em, Reg::RSP, extra);
    pop_r64(em, Reg::RBX);
    pop_r64(em, Reg::R13);
    pop_r64(em, Reg::R12);
    ret(em);
}

/// AVX2 4×8 register-tile matmul. Requires `M % 4 == 0` and `N % 8 == 0`.
///
/// Strategy:
/// - 4 independent ymm accumulators (ymm0..ymm3), one per output row in the tile.
/// - Inner k loop: load 1 ymm of B (8 columns) once, then for each of the 4
///   rows broadcast A[i+r, kk] and `vfmadd231ps` the row accumulator.
/// - This breaks the FMA dependency chain into 4 independent streams, letting
///   the CPU's out-of-order engine schedule them in parallel (ILP ~4×).
///
/// ```text
/// for i_blk in 0..M step 4:
///   for j_blk in 0..N step 8:
///     ymm0..ymm3 = 0
///     for kk in 0..K:
///       ymm4 = load [rhs + (kk*N + j_blk)*4]
///       ymm5 = broadcast [lhs + (i_blk+0)*K*4 + kk*4]
///       ymm0 = ymm0 + ymm5 * ymm4
///       ymm5 = broadcast [lhs + (i_blk+1)*K*4 + kk*4]
///       ymm1 = ymm1 + ymm5 * ymm4
///       ymm5 = broadcast [lhs + (i_blk+2)*K*4 + kk*4]
///       ymm2 = ymm2 + ymm5 * ymm4
///       ymm5 = broadcast [lhs + (i_blk+3)*K*4 + kk*4]
///       ymm3 = ymm3 + ymm5 * ymm4
///     store ymm0..ymm3 to 4 consecutive rows of `out`
/// ```
#[allow(clippy::too_many_arguments)]
fn emit_matmul_tile_4x8(
    em: &mut Emitter,
    p_lhs: Reg,
    p_rhs: Reg,
    p_out: Reg,
    m: i32,
    k: i32,
    n: i32,
    abi: Abi,
) {
    debug_assert!(m % 4 == 0 && n % 8 == 0);

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

    // R15 = i_blk (steps by 4), RBX = j_blk (steps by 8), RCX = kk
    xor_rr(em, Reg::R15);
    let i_loop = em.len();
    cmp_ri32(em, Reg::R15, m);
    let jge_i = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    xor_rr(em, Reg::RBX);
    let j_loop = em.len();
    cmp_ri32(em, Reg::RBX, n);
    let jge_j = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    // Zero the 4 accumulators (ymm0..ymm3)
    vxorps_zero(em, Ymm(0));
    vxorps_zero(em, Ymm(1));
    vxorps_zero(em, Ymm(2));
    vxorps_zero(em, Ymm(3));

    xor_rr(em, Reg::RCX);
    let k_loop = em.len();
    cmp_ri32(em, Reg::RCX, k);
    let jge_k = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    // Load B[kk, j_blk..j_blk+8] into ymm4.
    // b_idx = kk * N + j_blk  (in RDX)
    mov_rr(em, Reg::RDX, Reg::RCX);
    imul_rri32(em, Reg::RDX, Reg::RDX, n);
    add_rr(em, Reg::RDX, Reg::RBX);
    vmovups_load(em, Ymm(4), Reg::R13, Some((Reg::RDX, Scale::S4)), 0);

    // For each of the 4 rows, broadcast A and accumulate.
    // Row offset is constant within an inner k iteration but i_blk + r changes.
    for r in 0..4i32 {
        // a_idx = (i_blk + r) * K + kk  in RAX
        mov_rr(em, Reg::RAX, Reg::R15);
        if r != 0 {
            add_ri32(em, Reg::RAX, r);
        }
        imul_rri32(em, Reg::RAX, Reg::RAX, k);
        add_rr(em, Reg::RAX, Reg::RCX);
        vbroadcastss(em, Ymm(5), Reg::R12, Some((Reg::RAX, Scale::S4)), 0);
        // ymm[r] += ymm5 * ymm4
        vfmadd231ps_reg(em, Ymm(r as u8), Ymm(5), Ymm(4));
    }

    inc_r(em, Reg::RCX);
    let jmp_k = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp_k, k_loop);

    let k_done = em.len();
    patch_rel32(em, jge_k, k_done);

    // Store 4 accumulators to 4 consecutive output rows.
    for r in 0..4i32 {
        // out_idx = (i_blk + r) * N + j_blk  in RAX
        mov_rr(em, Reg::RAX, Reg::R15);
        if r != 0 {
            add_ri32(em, Reg::RAX, r);
        }
        imul_rri32(em, Reg::RAX, Reg::RAX, n);
        add_rr(em, Reg::RAX, Reg::RBX);
        vmovups_store(em, Ymm(r as u8), Reg::R14, Some((Reg::RAX, Scale::S4)), 0);
    }

    add_ri32(em, Reg::RBX, 8);
    let jmp_j = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp_j, j_loop);

    let j_done = em.len();
    patch_rel32(em, jge_j, j_done);

    add_ri32(em, Reg::R15, 4);
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
