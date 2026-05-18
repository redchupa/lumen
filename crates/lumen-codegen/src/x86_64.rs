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
    vaddps_reg, vaddps_zmm_reg, vaddss_xmm, vbroadcastss, vbroadcastss_xmm, vbroadcastss_zmm_xmm,
    vcvtdq2ps, vcvtdq2ps_zmm, vcvtph2ps_xmm, vcvtsi2ss_xmm_r32, vdivss_xmm, vextractf128_xmm,
    vextractf32x8_zmm, vfmadd231ps_mem, vfmadd231ps_reg, vfmadd231ps_zmm_mem, vhaddps_ymm,
    vmovd_load, vmovdqu_load, vmovups_load, vmovups_store, vmulps_reg, vmulps_zmm_reg, vmulss_xmm,
    vpcmpeqd_reg, vpdpbusd_evex_reg, vpdpbusd_vex_reg, vpmaddubsw_reg, vpmaddwd_reg,
    vpmovsxbd_load, vpmovsxbd_zmm_load, vpsignb_reg, vpsrlw_imm8, vsqrtss_xmm, vxorps_zero,
    vxorps_zmm_zero, Ymm, Zmm,
};
// Note: vpaddd_reg is exported but not used in any kernel yet (it was added
// alongside the other AVX2 integer ops in Phase 7.M for future kernels).
#[allow(unused_imports)]
use crate::avx2_enc::vpaddd_reg;

/// Which VNNI encoding to use for `vpdpbusd`. Selected by the caller based
/// on runtime CPU feature detection.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum VnniForm {
    /// AVX-VNNI (Intel Tiger Lake / Alder Lake+). 5-byte VEX form.
    Vex256,
    /// AVX-512 VNNI (Skylake-X / SPR / Zen 4). 6-byte EVEX form, works on
    /// any AVX-512 capable CPU even if the `avxvnni` CPUID bit is clear.
    Evex256,
}

impl VnniForm {
    fn emit(self, em: &mut Emitter, acc: Ymm, a: Ymm, b: Ymm) {
        match self {
            VnniForm::Vex256 => vpdpbusd_vex_reg(em, acc, a, b),
            VnniForm::Evex256 => vpdpbusd_evex_reg(em, acc, a, b),
        }
    }
}
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

    fn lower(&self, ir: &IrModule, opts: &CodegenOpts) -> Result<MachineCode, CodegenError> {
        let f = ir.functions.first().ok_or(CodegenError::UnsupportedOp(
            "module has no functions".into(),
        ))?;

        let mut em = Emitter::new();
        emit_function(&mut em, f, self.abi, self.avx2, opts.vnni, opts.use_avx512)?;
        Ok(MachineCode {
            bytes: em.buf,
            entry_offset: 0,
        })
    }
}

// ============================================================================
// Function emission
// ============================================================================

/// Emit one function. Four patterns supported:
/// - quant matmul: `Param(q8_0), Param(f32), Dequantize, MatMul, Return`  (Phase 5.C)
/// - matmul:       `Param, Param, MatMul, Return`                         (Phase 2.B/3)
/// - dequant:      `Param, Dequantize, Return`                            (Phase 5.B)
/// - rms norm:     `Param(f32, [H]), Param(f32, [H]), RmsNorm, Return`    (Phase 6.C.2)
fn emit_function(
    em: &mut Emitter,
    f: &Function,
    abi: Abi,
    avx2: bool,
    vnni: Option<VnniForm>,
    use_avx512: bool,
) -> Result<(), CodegenError> {
    let has_rms = f.values.iter().any(|v| matches!(v.op, Op::RmsNorm { .. }));
    if has_rms {
        return emit_function_rms_norm(em, f, abi);
    }

    let has_dequant = f
        .values
        .iter()
        .any(|v| matches!(v.op, Op::Dequantize { .. }));
    let has_matmul = f.values.iter().any(|v| matches!(v.op, Op::MatMul { .. }));

    // Pattern 1: dequant fused with matmul → on-the-fly quant matmul.
    if has_dequant && has_matmul {
        return emit_function_quant_matmul_q8(em, f, abi, vnni, use_avx512);
    }

    // Pattern 2: standalone dequant.
    if has_dequant {
        let dq_idx = f
            .values
            .iter()
            .position(|v| matches!(v.op, Op::Dequantize { .. }))
            .unwrap();
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
    // 1. 4×8 register tile when both M and N divide nicely (best prefill throughput).
    // 2. 1×N 4-accumulator AVX2 when M == 1 and N % 32 == 0 (best decode throughput —
    //    breaks the single-accumulator FMA latency chain that bound the 1×8 path).
    // 3. 1×8 AVX2 vectorization when only N % 8 divides.
    // 4. Scalar fallback otherwise.
    if avx2 && m % 4 == 0 && n % 8 == 0 {
        emit_matmul_tile_4x8(em, p_lhs, p_rhs, p_out, m as i32, k as i32, n as i32, abi);
    } else if avx2 && m == 1 && n % 32 == 0 {
        emit_matmul_avx2_1xn_4acc(em, p_lhs, p_rhs, p_out, k as i32, n as i32, abi);
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

/// Emit a fused quantized matmul: Q8_0 weights × F32 activations → F32 output.
///
/// Source IR:
/// ```text
///   v0 = Param(0): tensor<q8_0, [M, K]>     # weights (quantized)
///   v1 = Param(1): tensor<f32,  [K, N]>     # activations
///   v2 = Dequantize v0 : tensor<f32, [M, K]>
///   v3 = MatMul v2, v1 : tensor<f32, [M, N]>
///   return v3
/// ```
///
/// Lumen recognizes this as one fused kernel and emits a single loop nest
/// that *never* materializes the full dequantized weight matrix. K must be
/// a multiple of 32 (Q8_0 block size). N must be a multiple of 8 (AVX2 row).
///
/// Generated pseudo-asm:
/// ```text
///   for i in 0..M:
///     for j in 0..N step 8:
///       ymm_acc = 0
///       for kb in 0..K/32:               ; block index
///         load fp16 d, convert -> xmm_d
///         for kk in 0..32:               ; in-block element
///           movsx eax, byte [weights + (i*KB + kb)*34 + 2 + kk]
///           vcvtsi2ss xmm_w, xmm_w, eax
///           vmulss    xmm_w, xmm_w, xmm_d
///           vbroadcastss ymm_w, xmm_w
///           vfmadd231ps ymm_acc, ymm_w, [activations + ((kb*32 + kk)*N + j)*4]
///       store ymm_acc -> out[i, j..j+8]
/// ```
fn emit_function_quant_matmul_q8(
    em: &mut Emitter,
    f: &Function,
    abi: Abi,
    vnni: Option<VnniForm>,
    use_avx512: bool,
) -> Result<(), CodegenError> {
    // Locate the ops and verify the pattern.
    let dq_indices: Vec<usize> = f
        .values
        .iter()
        .enumerate()
        .filter_map(|(i, v)| matches!(v.op, Op::Dequantize { .. }).then_some(i))
        .collect();
    let mm_idx = f
        .values
        .iter()
        .position(|v| matches!(v.op, Op::MatMul { .. }))
        .ok_or(CodegenError::UnsupportedOp("missing matmul".into()))?;

    let Op::MatMul { lhs, rhs } = f.values[mm_idx].op else {
        unreachable!()
    };

    // Pattern dispatch:
    //   1 Dequantize on LHS, F32 RHS  → Phase 5.C / 5.C-N=1 (Q8 × F32)
    //   2 Dequantize on both sides    → Phase 7.M (Q8 × Q8 int dot)
    let q8q8 = dq_indices.len() == 2
        && lhs == lumen_ir::ValueId(dq_indices[0] as u32)
        && rhs == lumen_ir::ValueId(dq_indices[1] as u32);

    if q8q8 {
        // Q8 × Q8 fused matmul (Phase 7.M).
        let Op::Dequantize { x: w_src } = f.values[dq_indices[0]].op else {
            unreachable!()
        };
        let Op::Dequantize { x: a_src } = f.values[dq_indices[1]].op else {
            unreachable!()
        };
        let w_ty = f.values[w_src.0 as usize].ty.clone();
        let a_ty = f.values[a_src.0 as usize].ty.clone();
        if w_ty.dtype != DType::Q8_0 || a_ty.dtype != DType::Q8_0 {
            return Err(CodegenError::UnsupportedOp(
                "Phase 7.M: Q8×Q8 path requires both operands Q8_0".into(),
            ));
        }
        let (m, k) = static_2d(&w_ty)?;
        let (k_a, n) = static_2d(&a_ty)?;
        if k != k_a {
            return Err(CodegenError::ShapeError(
                "Phase 7.M: matmul inner-dim mismatch".into(),
            ));
        }
        if k % 32 != 0 {
            return Err(CodegenError::ShapeError(
                "Phase 7.M: K must be a multiple of 32 (Q8_0 block size)".into(),
            ));
        }
        if n != 1 {
            return Err(CodegenError::ShapeError(
                "Phase 7.M: only N=1 (decode) Q8×Q8 kernel implemented".into(),
            ));
        }
        let w_param = match f.values[w_src.0 as usize].op {
            Op::Param { index } => index,
            _ => {
                return Err(CodegenError::UnsupportedOp(
                    "Q8×Q8 weight source must be Param".into(),
                ))
            }
        };
        let a_param = match f.values[a_src.0 as usize].op {
            Op::Param { index } => index,
            _ => {
                return Err(CodegenError::UnsupportedOp(
                    "Q8×Q8 activation source must be Param".into(),
                ))
            }
        };
        let p_w = abi.param_reg(w_param);
        let p_a = abi.param_reg(a_param);
        let p_out = abi.param_reg(f.params.len() as u32);
        // 3-way kernel pick:
        //   - VNNI + K_blocks % 4 == 0 → 4-accumulator unrolled (Phase 7.O)
        //   - VNNI only                → single-accumulator (Phase 7.N)
        //   - no VNNI                  → AVX2 vpsignb+vpmaddubsw+vpmaddwd (Phase 7.M)
        let k_blocks = (k / 32) as i32;
        match vnni {
            Some(form) if k_blocks % 4 == 0 => {
                emit_quant_matmul_q8q8_n1_body_vnni_4acc(
                    em, p_w, p_a, p_out, m as i32, k as i32, abi, form,
                );
            }
            Some(form) => {
                emit_quant_matmul_q8q8_n1_body_vnni(
                    em, p_w, p_a, p_out, m as i32, k as i32, abi, form,
                );
            }
            None => {
                emit_quant_matmul_q8q8_n1_body(em, p_w, p_a, p_out, m as i32, k as i32, abi);
            }
        }
        return Ok(());
    }

    // Single-Dequantize path (Phase 5.C and 5.C-N=1).
    let dq_idx = dq_indices.first().copied().unwrap();

    let Op::Dequantize { x: dq_src } = f.values[dq_idx].op else {
        unreachable!()
    };

    if lhs != lumen_ir::ValueId(dq_idx as u32) {
        return Err(CodegenError::UnsupportedOp(
            "Phase 5.C: matmul LHS must be the Dequantize result".into(),
        ));
    }

    let w_ty = f.values[dq_src.0 as usize].ty.clone();
    let a_ty = f.values[rhs.0 as usize].ty.clone();
    if w_ty.dtype != DType::Q8_0 {
        return Err(CodegenError::UnsupportedOp(
            "Phase 5.C: only Q8_0 weights supported".into(),
        ));
    }
    if a_ty.dtype != DType::F32 {
        return Err(CodegenError::UnsupportedOp(
            "Phase 5.C: activations must be f32".into(),
        ));
    }
    let (m, k) = static_2d(&w_ty)?;
    let (k_a, n) = static_2d(&a_ty)?;
    if k != k_a {
        return Err(CodegenError::ShapeError(
            "Phase 5.C: matmul inner-dim mismatch".into(),
        ));
    }
    if k % 32 != 0 {
        return Err(CodegenError::ShapeError(
            "Phase 5.C: K must be a multiple of 32 (Q8_0 block size)".into(),
        ));
    }
    // N == 1 is the decode shape (single-token autoregressive): handled by a
    // specialized K-vectorized + horizontal-reduce kernel.
    // N % 8 == 0 is the prefill / batched shape: handled by the original 5.C
    // path that vectorizes over output columns.
    if n != 1 && n % 8 != 0 {
        return Err(CodegenError::ShapeError(
            "Phase 5.C: N must be 1 (decode) or a multiple of 8 (prefill)".into(),
        ));
    }

    let w_param = match f.values[dq_src.0 as usize].op {
        Op::Param { index } => index,
        _ => {
            return Err(CodegenError::UnsupportedOp(
                "dequant source must be Param".into(),
            ))
        }
    };
    let a_param = match f.values[rhs.0 as usize].op {
        Op::Param { index } => index,
        _ => {
            return Err(CodegenError::UnsupportedOp(
                "matmul RHS must be Param".into(),
            ))
        }
    };

    let p_w = abi.param_reg(w_param);
    let p_a = abi.param_reg(a_param);
    let p_out = abi.param_reg(f.params.len() as u32);

    if n == 1 {
        // Phase 7.S: AVX-512 ZMM 2-acc variant when the host supports it.
        // Same dataflow as the 4-acc YMM body but with 16-lane FMAs (2 sub-
        // accumulators across the two halves of each 32-element Q8 block)
        // — half the inner instructions for the same compute, beneficial on
        // long-K matmuls (e.g. Qwen2 FFN-down with K_blocks=152).
        if use_avx512 {
            emit_quant_matmul_q8_n1_body_zmm(em, p_w, p_a, p_out, m as i32, k as i32, abi);
        } else {
            emit_quant_matmul_q8_n1_body(em, p_w, p_a, p_out, m as i32, k as i32, abi);
        }
    } else {
        emit_quant_matmul_q8_body(em, p_w, p_a, p_out, m as i32, k as i32, n as i32, abi);
    }
    Ok(())
}

/// Body of the Q8 × F32 fused matmul.
///
/// Stack layout (inside the function, after prologue + adjustment):
/// - R12 = weights base (q8_0 blocks)
/// - R13 = activations base (f32)
/// - R14 = output base (f32)
/// - R15 = i (row of weights / output)
/// - RBX = j (column block of output, step 8)
/// - R11 = kb (block index along K)
/// - RCX = kk (element within block, 0..32)
///
/// Block stride for weights = num_K_blocks per row × 34 bytes.
///   weight row byte offset (for row i) = i * (K/32) * 34.
///   block byte offset within row = kb * 34.
/// Stride for activations: each row is N floats = 4*N bytes.
#[allow(clippy::too_many_arguments)]
fn emit_quant_matmul_q8_body(
    em: &mut Emitter,
    p_w: Reg,
    p_a: Reg,
    p_out: Reg,
    m: i32,
    k: i32,
    n: i32,
    abi: Abi,
) {
    let k_blocks = k / 32;
    let row_w_bytes = k_blocks * 34;

    // ---- prologue ----
    push_r64(em, Reg::R12);
    push_r64(em, Reg::R13);
    push_r64(em, Reg::R14);
    push_r64(em, Reg::R15);
    push_r64(em, Reg::RBX);
    // R11 doesn't need preservation in Windows or SysV ABI (caller-saved).
    let extra = if abi == Abi::WinX64 { 40 } else { 8 };
    sub_ri32(em, Reg::RSP, extra);

    mov_rr(em, Reg::R12, p_w);
    mov_rr(em, Reg::R13, p_a);
    mov_rr(em, Reg::R14, p_out);

    xor_rr(em, Reg::R15); // i = 0
    let i_loop = em.len();
    cmp_ri32(em, Reg::R15, m);
    let jge_i = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    xor_rr(em, Reg::RBX); // j = 0
    let j_loop = em.len();
    cmp_ri32(em, Reg::RBX, n);
    let jge_j = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    // ymm0 = accumulator = 0
    vxorps_zero(em, Ymm(0));

    xor_rr(em, Reg::R11); // kb = 0
    let kb_loop = em.len();
    cmp_ri32(em, Reg::R11, k_blocks);
    let jge_kb = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    // weight row byte offset RDX = i * row_w_bytes
    mov_rr(em, Reg::RDX, Reg::R15);
    imul_rri32(em, Reg::RDX, Reg::RDX, row_w_bytes);
    // weight block byte offset RAX = RDX + kb*34
    mov_rr(em, Reg::RAX, Reg::R11);
    imul_rri32(em, Reg::RAX, Reg::RAX, 34);
    add_rr(em, Reg::RAX, Reg::RDX);
    // Now RAX = byte offset of this Q8_0 block within R12 base.

    // Load fp16 d at [R12 + RAX], convert to fp32 in xmm_d (xmm2)
    vmovd_load(em, Ymm(2), Reg::R12, Some((Reg::RAX, Scale::S1)), 0);
    vcvtph2ps_xmm(em, Ymm(2), Ymm(2));

    // 32 inner iterations
    xor_rr(em, Reg::RCX); // kk = 0
    let kk_loop = em.len();
    cmp_ri32(em, Reg::RCX, 32);
    let jge_kk = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    // movsx r9d, byte [R12 + RAX + 2 + RCX]
    // R12 base + RAX (block offset) + 2 (skip d) + RCX (in-block idx)
    // We can't have 3 registers, so combine: RAX_temp = RAX + RCX
    // Use R8 as scratch for the byte addr index.
    // Simpler: mov tmp = RAX; add tmp, RCX; load [R12 + tmp + 2]
    mov_rr(em, Reg::R8, Reg::RAX);
    add_rr(em, Reg::R8, Reg::RCX);
    movsx_r32_m8(em, Reg::R9, Reg::R12, Some((Reg::R8, Scale::S1)), 2);

    // vcvtsi2ss xmm1, xmm2 (any), R9d → xmm1 = (float)i32(R9), upper inherits from xmm2
    vcvtsi2ss_xmm_r32(em, Ymm(1), Ymm(2), Reg::R9);
    // vmulss xmm1, xmm1, xmm2 → xmm1 = w * d
    vmulss_xmm(em, Ymm(1), Ymm(1), Ymm(2));
    // vbroadcastss ymm1, xmm1 → broadcast scaled weight
    vbroadcastss_xmm(em, Ymm(1), Ymm(1));

    // Compute activation byte offset:
    //   act_idx = (kb*32 + kk) * N + j   (in floats)
    //   byte offset = act_idx * 4
    // act_row_idx (in floats) = (kb*32 + kk)*N  in R8 (reuse)
    mov_rr(em, Reg::R8, Reg::R11);
    imul_rri32(em, Reg::R8, Reg::R8, 32);
    add_rr(em, Reg::R8, Reg::RCX);
    imul_rri32(em, Reg::R8, Reg::R8, n);
    add_rr(em, Reg::R8, Reg::RBX);
    // FMA: ymm0 += ymm1 * [R13 + R8*4]
    vfmadd231ps_mem(em, Ymm(0), Ymm(1), Reg::R13, Some((Reg::R8, Scale::S4)), 0);

    inc_r(em, Reg::RCX);
    let jmp_kk = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp_kk, kk_loop);

    let kk_done = em.len();
    patch_rel32(em, jge_kk, kk_done);

    inc_r(em, Reg::R11);
    let jmp_kb = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp_kb, kb_loop);

    let kb_done = em.len();
    patch_rel32(em, jge_kb, kb_done);

    // Store output row: [R14 + (i*N + j)*4]
    mov_rr(em, Reg::RAX, Reg::R15);
    imul_rri32(em, Reg::RAX, Reg::RAX, n);
    add_rr(em, Reg::RAX, Reg::RBX);
    vmovups_store(em, Ymm(0), Reg::R14, Some((Reg::RAX, Scale::S4)), 0);

    add_ri32(em, Reg::RBX, 8);
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

/// Q8 × F32 fused matmul, N=1 specialization for decode.
///
/// Phase 7.G: replaces the 1-accumulator variant. Decode profiling showed
/// 88% of forward time was matmul, and within the Q8 N=1 kernel every
/// `vfmadd231ps` of one Q8 block (4 of them) accumulated into the same
/// `ymm0`, serializing 4 FMAs into a single dependency chain (4-cycle
/// latency × 4 FMAs = 16 cycles minimum per Q8 block).
///
/// This version uses **four independent accumulators** — one per 8-element
/// chunk of a Q8 block. The FMAs across a block now issue back-to-back, and
/// only the cross-block accumulation needs to wait. We also fold the
/// activation load into a memory-operand FMA, dropping one op per inner.
///
///   for i in 0..M:                              # output row
///     ymm0..ymm3 = 0                            # 4 × 8-wide accumulators
///     for kb in 0..K/32:                        # Q8_0 block index
///       load fp16 d → broadcast → ymm5
///       # 4 unrolled 8-wide chunks across the block, each into its own acc:
///       for (inner, acc) in [(0,0), (8,1), (16,2), (24,3)]:
///         vpmovsxbd ymm6, [w + 2 + inner]       ; 8 i8 → 8 i32
///         vcvtdq2ps ymm6, ymm6                  ; → 8 fp32
///         vmulps    ymm6, ymm6, ymm5            ; scaled weights
///         vfmadd231ps_mem ymm_acc, ymm6, [act + (kb*32+inner)*4]
///     # ymm0 += ymm1 + ymm2 + ymm3 (still 8-wide); then horizontal reduce.
///     vaddps ymm0, ymm0, ymm1
///     vaddps ymm2, ymm2, ymm3
///     vaddps ymm0, ymm0, ymm2
///     vhaddps × 2 → vextractf128 → vaddss → movss
///
/// The win versus the model's prior fp32 dequant-then-matmul path is
/// already captured (eliminating the dequant pass + transpose). This phase
/// adds another factor on top by hiding FMA latency the same way Phase 7.C
/// did for the fp32 kernel.
#[allow(clippy::too_many_arguments)]
fn emit_quant_matmul_q8_n1_body(
    em: &mut Emitter,
    p_w: Reg,
    p_a: Reg,
    p_out: Reg,
    m: i32,
    k: i32,
    abi: Abi,
) {
    let k_blocks = k / 32;
    let row_w_bytes = k_blocks * 34;

    // ---- prologue ----
    push_r64(em, Reg::R12);
    push_r64(em, Reg::R13);
    push_r64(em, Reg::R14);
    push_r64(em, Reg::R15);
    push_r64(em, Reg::RBX);
    let extra = if abi == Abi::WinX64 { 40 } else { 8 };
    sub_ri32(em, Reg::RSP, extra);

    mov_rr(em, Reg::R12, p_w);
    mov_rr(em, Reg::R13, p_a);
    mov_rr(em, Reg::R14, p_out);

    // R15 = i (output row), R11 = kb. R10 = kb*32 (activation index base).
    // RAX = absolute byte offset of current (i, kb) Q8 block.
    xor_rr(em, Reg::R15);
    let i_loop = em.len();
    cmp_ri32(em, Reg::R15, m);
    let jge_i = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    // 4 accumulators zeroed.
    vxorps_zero(em, Ymm(0));
    vxorps_zero(em, Ymm(1));
    vxorps_zero(em, Ymm(2));
    vxorps_zero(em, Ymm(3));

    xor_rr(em, Reg::R11); // kb = 0
    let kb_loop = em.len();
    cmp_ri32(em, Reg::R11, k_blocks);
    let jge_kb = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    // RAX = i*row_w_bytes + kb*34
    mov_rr(em, Reg::RAX, Reg::R15);
    imul_rri32(em, Reg::RAX, Reg::RAX, row_w_bytes);
    mov_rr(em, Reg::RDX, Reg::R11);
    imul_rri32(em, Reg::RDX, Reg::RDX, 34);
    add_rr(em, Reg::RAX, Reg::RDX);

    // ymm5 = broadcast(fp16_to_fp32(d))
    vmovd_load(em, Ymm(5), Reg::R12, Some((Reg::RAX, Scale::S1)), 0);
    vcvtph2ps_xmm(em, Ymm(5), Ymm(5));
    vbroadcastss_xmm(em, Ymm(5), Ymm(5));

    // R10 = kb * 32 (activation float index base for this block)
    mov_rr(em, Reg::R10, Reg::R11);
    imul_rri32(em, Reg::R10, Reg::R10, 32);

    // 4 unrolled 8-wide chunks. Each one accumulates into its own ymm acc,
    // so the 4 FMAs in this block are independent at the architectural level
    // and can issue back-to-back instead of serializing on ymm0.
    //
    // Register choice note: ymm0..ymm5 are caller-saved under both Win64 and
    // SysV. ymm6-ymm15 are *callee-saved on Win64* — clobbering them without
    // a save/restore corrupts the caller's float state across the call (this
    // exact bug surfaced as NaN logits on Qwen even though the unit test
    // passed: unit tests have no surrounding state to be poisoned).
    for (inner, acc) in [(0i32, 0u8), (8, 1), (16, 2), (24, 3)] {
        // ymm4 = sign-ext 8 i8 -> 8 i32 -> 8 fp32 -> * d
        vpmovsxbd_load(em, Ymm(4), Reg::R12, Some((Reg::RAX, Scale::S1)), 2 + inner);
        vcvtdq2ps(em, Ymm(4), Ymm(4));
        vmulps_reg(em, Ymm(4), Ymm(4), Ymm(5));
        // ymm_acc += ymm4 * [r13 + r10*4 + inner*4]  (memory-operand FMA)
        vfmadd231ps_mem(
            em,
            Ymm(acc),
            Ymm(4),
            Reg::R13,
            Some((Reg::R10, Scale::S4)),
            inner * 4,
        );
    }

    inc_r(em, Reg::R11);
    let jmp_kb = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp_kb, kb_loop);

    let kb_done = em.len();
    patch_rel32(em, jge_kb, kb_done);

    // Combine the 4 accumulators: ymm0 = ((y0 + y1) + (y2 + y3))
    vaddps_reg(em, Ymm(0), Ymm(0), Ymm(1));
    vaddps_reg(em, Ymm(2), Ymm(2), Ymm(3));
    vaddps_reg(em, Ymm(0), Ymm(0), Ymm(2));

    // Horizontal reduce ymm0 (8 fp32 lanes) -> scalar in xmm0[0].
    vhaddps_ymm(em, Ymm(0), Ymm(0), Ymm(0));
    vhaddps_ymm(em, Ymm(0), Ymm(0), Ymm(0));
    vextractf128_xmm(em, Ymm(1), Ymm(0), 1);
    vaddss_xmm(em, Ymm(0), Ymm(0), Ymm(1));

    // movss [r14 + r15*4], xmm0
    movss_store(em, Xmm(0), Reg::R14, Some((Reg::R15, Scale::S4)), 0);

    inc_r(em, Reg::R15);
    let jmp_i = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp_i, i_loop);

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

/// AVX-512 ZMM-wide variant of `emit_quant_matmul_q8_n1_body`. Phase 7.S.
///
/// Same Q8 weights × F32 activations → F32 output fused matmul (N=1
/// decode), but the inner loop processes a Q8 block (32 K-elements) as
/// 2 ZMM-16-lane chunks instead of 4 YMM-8-lane chunks. Two independent
/// fp32 sub-accumulators (zmm0, zmm1) one per chunk-index keep an
/// independent FMA chain for each half of every block.
///
/// Half the inner instructions per block (2 chunks vs 4) for the same
/// compute work — and each ZMM FMA processes 16 lanes, so back-end FMA
/// throughput stays the same. Net win comes from compute-bound shapes
/// (Qwen2 FFN-down with K_blocks=152 hits this).
///
/// Register plan (Win64-safe: zmm0..zmm5 only — same physical regs as
/// the caller-saved xmm0..xmm5):
///   zmm0, zmm1 = fp32 row sub-accumulators
///   zmm2 = broadcast(d_w)
///   zmm3 = weight i32 / fp32 scratch
///   xmm2/xmm4 = scalar scratch for the fp16→fp32 d_w (uses low lane of zmm2)
///
/// Requires AVX-512F + AVX-512BW (vpmovsxbd zmm m128 uses BW).
#[allow(clippy::too_many_arguments)]
fn emit_quant_matmul_q8_n1_body_zmm(
    em: &mut Emitter,
    p_w: Reg,
    p_a: Reg,
    p_out: Reg,
    m: i32,
    k: i32,
    abi: Abi,
) {
    let k_blocks = k / 32;
    let row_w_bytes = k_blocks * 34;

    // ---- prologue ----
    push_r64(em, Reg::R12);
    push_r64(em, Reg::R13);
    push_r64(em, Reg::R14);
    push_r64(em, Reg::R15);
    push_r64(em, Reg::RBX);
    let extra = if abi == Abi::WinX64 { 40 } else { 8 };
    sub_ri32(em, Reg::RSP, extra);

    mov_rr(em, Reg::R12, p_w);
    mov_rr(em, Reg::R13, p_a);
    mov_rr(em, Reg::R14, p_out);

    // R15 = i (output row), R11 = kb (block index), R10 = kb*32 (act idx base).
    xor_rr(em, Reg::R15);
    let i_loop = em.len();
    cmp_ri32(em, Reg::R15, m);
    let jge_i = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    // Two fp32 row sub-accumulators.
    vxorps_zmm_zero(em, Zmm(0));
    vxorps_zmm_zero(em, Zmm(1));

    xor_rr(em, Reg::R11);
    let kb_loop = em.len();
    cmp_ri32(em, Reg::R11, k_blocks);
    let jge_kb = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    // RAX = i*row_w_bytes + kb*34 (weight block byte offset)
    mov_rr(em, Reg::RAX, Reg::R15);
    imul_rri32(em, Reg::RAX, Reg::RAX, row_w_bytes);
    mov_rr(em, Reg::RDX, Reg::R11);
    imul_rri32(em, Reg::RDX, Reg::RDX, 34);
    add_rr(em, Reg::RAX, Reg::RDX);

    // R10 = kb*32 (activation float index base)
    mov_rr(em, Reg::R10, Reg::R11);
    imul_rri32(em, Reg::R10, Reg::R10, 32);

    // zmm2 = broadcast(fp16_to_fp32(d_w))
    vmovd_load(em, Ymm(2), Reg::R12, Some((Reg::RAX, Scale::S1)), 0);
    vcvtph2ps_xmm(em, Ymm(2), Ymm(2));
    vbroadcastss_zmm_xmm(em, Zmm(2), Zmm(2));

    // Chunk 0: K elements [0..16). zmm3 = (16 i8 → 16 i32 → 16 fp32) * d_w.
    vpmovsxbd_zmm_load(em, Zmm(3), Reg::R12, Some((Reg::RAX, Scale::S1)), 2);
    vcvtdq2ps_zmm(em, Zmm(3), Zmm(3));
    vmulps_zmm_reg(em, Zmm(3), Zmm(3), Zmm(2));
    // zmm0 += zmm3 * [r13 + r10*4 + 0] (16 fp32 activations)
    vfmadd231ps_zmm_mem(em, Zmm(0), Zmm(3), Reg::R13, Some((Reg::R10, Scale::S4)), 0);

    // Chunk 1: K elements [16..32). disp = 16 bytes (in weights) / 64 bytes (in acts).
    vpmovsxbd_zmm_load(em, Zmm(3), Reg::R12, Some((Reg::RAX, Scale::S1)), 2 + 16);
    vcvtdq2ps_zmm(em, Zmm(3), Zmm(3));
    vmulps_zmm_reg(em, Zmm(3), Zmm(3), Zmm(2));
    vfmadd231ps_zmm_mem(
        em,
        Zmm(1),
        Zmm(3),
        Reg::R13,
        Some((Reg::R10, Scale::S4)),
        64,
    );

    inc_r(em, Reg::R11);
    let jmp_kb = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp_kb, kb_loop);

    let kb_done = em.len();
    patch_rel32(em, jge_kb, kb_done);

    // Combine sub-accs: zmm0 += zmm1 (16 fp32 lanes).
    vaddps_zmm_reg(em, Zmm(0), Zmm(0), Zmm(1));

    // Horizontal reduce zmm0 (16 fp32 lanes) → scalar. Strategy:
    //   1. Extract upper 8 lanes to ymm1.
    //   2. Add to lower 8 lanes (zmm0 low half = ymm0).
    //   3. Run the existing YMM hsum tail (vhaddps × 2 + vextractf128 + vaddss).
    vextractf32x8_zmm(em, Ymm(1), Zmm(0), 1);
    // ymm0 = ymm0_low + ymm1 (using the existing YMM vaddps).
    vaddps_reg(em, Ymm(0), Ymm(0), Ymm(1));
    vhaddps_ymm(em, Ymm(0), Ymm(0), Ymm(0));
    vhaddps_ymm(em, Ymm(0), Ymm(0), Ymm(0));
    vextractf128_xmm(em, Ymm(1), Ymm(0), 1);
    vaddss_xmm(em, Ymm(0), Ymm(0), Ymm(1));
    movss_store(em, Xmm(0), Reg::R14, Some((Reg::R15, Scale::S4)), 0);

    inc_r(em, Reg::R15);
    let jmp_i = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp_i, i_loop);

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

/// Q8 weights × Q8 activations fused matmul, N=1 specialization.
///
/// Phase 7.M: this is the same shape as `emit_quant_matmul_q8_n1_body` but
/// the activations come in pre-quantized as Q8_0 blocks, so the inner
/// dot product can run in the integer domain instead of dequantizing per
/// block to fp32.
///
/// Per Q8 block of (32 weight i8s + 32 activation i8s):
/// ```text
///   load 32 i8 weights into ymm1
///   load 32 i8 acts    into ymm2
///   ymm3 = vpsignb(ymm1, ymm1)         ; |weight|         (u8)
///   ymm2 = vpsignb(ymm2, ymm1)         ; act * sign(w)   (i8)
///   ymm3 = vpmaddubsw(ymm3, ymm2)      ; 16 i16 pair-sums of u8 × i8
///   ymm3 = vpmaddwd(ymm3, ymm_ones16)  ; 8 i32 pair-sums of i16 × 1
///   ymm3 = vcvtdq2ps(ymm3)             ; 8 fp32 lanes (= int8 dot, partial)
///   scale = fp16_to_fp32(d_w) * fp16_to_fp32(d_a)   ; scalar
///   ymm4 = vbroadcastss(scale)
///   ymm0 = vfmadd231ps(ymm0, ymm3, ymm4)            ; fp32 row acc += scaled
/// ```
/// Per row epilogue: horizontal-sum ymm0 -> scalar, movss out[i].
///
/// Register plan (Win64-safe: ymm0..ymm5 only):
///   ymm0 = fp32 row accumulator
///   ymm1 = weight i8 vector (temp)
///   ymm2 = act i8 vector (temp), then sign-corrected
///   ymm3 = |w| / prod16 / prod32 / fp32 prod (temp)
///   ymm4 = d_w*d_a broadcast (temp)
///   ymm5 = ones16 constant (live across all rows; forged at function entry)
#[allow(clippy::too_many_arguments)]
fn emit_quant_matmul_q8q8_n1_body(
    em: &mut Emitter,
    p_w: Reg,
    p_a: Reg,
    p_out: Reg,
    m: i32,
    k: i32,
    abi: Abi,
) {
    let k_blocks = k / 32;
    let row_w_bytes = k_blocks * 34;

    // ---- prologue ----
    push_r64(em, Reg::R12);
    push_r64(em, Reg::R13);
    push_r64(em, Reg::R14);
    push_r64(em, Reg::R15);
    push_r64(em, Reg::RBX);
    let extra = if abi == Abi::WinX64 { 40 } else { 8 };
    sub_ri32(em, Reg::RSP, extra);

    mov_rr(em, Reg::R12, p_w);
    mov_rr(em, Reg::R13, p_a);
    mov_rr(em, Reg::R14, p_out);

    // Forge a 16-lane i16(1) constant in ymm5:
    //   vpcmpeqd ymm5,ymm5,ymm5 → all bits set (= -1 in every i32 lane)
    //   vpsrlw   ymm5, ymm5, 15 → each i16 lane = 0x0001
    // Stays in ymm5 for the entire function (no spills needed).
    vpcmpeqd_reg(em, Ymm(5), Ymm(5), Ymm(5));
    vpsrlw_imm8(em, Ymm(5), Ymm(5), 15);

    // R15 = i (output row), R11 = kb (block index).
    xor_rr(em, Reg::R15);
    let i_loop = em.len();
    cmp_ri32(em, Reg::R15, m);
    let jge_i = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    // fp32 row accumulator (8 lanes).
    vxorps_zero(em, Ymm(0));

    xor_rr(em, Reg::R11);
    let kb_loop = em.len();
    cmp_ri32(em, Reg::R11, k_blocks);
    let jge_kb = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    // RAX = i*row_w_bytes + kb*34   (weight block byte offset)
    mov_rr(em, Reg::RAX, Reg::R15);
    imul_rri32(em, Reg::RAX, Reg::RAX, row_w_bytes);
    mov_rr(em, Reg::RDX, Reg::R11);
    imul_rri32(em, Reg::RDX, Reg::RDX, 34);
    add_rr(em, Reg::RAX, Reg::RDX);

    // R10 = kb*34   (act block byte offset)
    mov_rr(em, Reg::R10, Reg::R11);
    imul_rri32(em, Reg::R10, Reg::R10, 34);

    // Load 32 i8 weight bytes (skip 2-byte d header) and 32 i8 act bytes.
    vmovdqu_load(em, Ymm(1), Reg::R12, Some((Reg::RAX, Scale::S1)), 2);
    vmovdqu_load(em, Ymm(2), Reg::R13, Some((Reg::R10, Scale::S1)), 2);

    // vpsignb trick: |w| (u8) and a*sign(w) (i8) for the u8×i8 maddubsw.
    vpsignb_reg(em, Ymm(3), Ymm(1), Ymm(1));
    vpsignb_reg(em, Ymm(2), Ymm(2), Ymm(1));

    // 32 (u8 × i8) → 16 i16 pair-sums.
    vpmaddubsw_reg(em, Ymm(3), Ymm(3), Ymm(2));
    // 16 i16 × 1 → 8 i32 pair-sums (effectively 4-wide reduction).
    vpmaddwd_reg(em, Ymm(3), Ymm(3), Ymm(5));
    // i32 → fp32 (8 lanes).
    vcvtdq2ps(em, Ymm(3), Ymm(3));

    // Compute d_w * d_a as a scalar fp32, broadcast to ymm4.
    //   ymm1 = fp16(d_w), then xmm1 = fp32(d_w)
    //   ymm2 = fp16(d_a), then xmm2 = fp32(d_a)
    //   xmm1 = xmm1 * xmm2 (scalar)
    //   ymm4 = broadcast(xmm1)
    vmovd_load(em, Ymm(1), Reg::R12, Some((Reg::RAX, Scale::S1)), 0);
    vcvtph2ps_xmm(em, Ymm(1), Ymm(1));
    vmovd_load(em, Ymm(2), Reg::R13, Some((Reg::R10, Scale::S1)), 0);
    vcvtph2ps_xmm(em, Ymm(2), Ymm(2));
    vmulss_xmm(em, Ymm(1), Ymm(1), Ymm(2));
    vbroadcastss_xmm(em, Ymm(4), Ymm(1));

    // ymm0 += ymm3 * ymm4
    vfmadd231ps_reg(em, Ymm(0), Ymm(3), Ymm(4));

    inc_r(em, Reg::R11);
    let jmp_kb = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp_kb, kb_loop);

    let kb_done = em.len();
    patch_rel32(em, jge_kb, kb_done);

    // Horizontal reduce ymm0 (8 fp32 lanes) -> scalar in xmm0[0], store.
    vhaddps_ymm(em, Ymm(0), Ymm(0), Ymm(0));
    vhaddps_ymm(em, Ymm(0), Ymm(0), Ymm(0));
    vextractf128_xmm(em, Ymm(1), Ymm(0), 1);
    vaddss_xmm(em, Ymm(0), Ymm(0), Ymm(1));
    movss_store(em, Xmm(0), Reg::R14, Some((Reg::R15, Scale::S4)), 0);

    inc_r(em, Reg::R15);
    let jmp_i = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp_i, i_loop);

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

/// Q8×Q8 fused matmul (N=1), VNNI variant. Same dataflow as
/// `emit_quant_matmul_q8q8_n1_body` but the i8×i8-then-pair-sum chain
/// (vpsignb → vpsignb → vpmaddubsw → vpmaddwd) collapses into one
/// `vpdpbusd` per block, dropping the critical path from ~10 cycles to
/// ~5 cycles per block.
///
/// vpdpbusd computes `acc_i32[i] += sum_4_pairs(u8 × i8)`, which is exactly
/// the i16-pair-sum-then-i32-pair-sum the AVX2 chain was doing in two
/// instructions. Caller still applies the vpsignb trick (|w| as u8, a*sign(w)
/// as i8) to handle our signed×signed Q8 multiplication.
///
/// Register plan (Win64-safe: ymm0..ymm5 only). Note: the `ones16` constant
/// used by the AVX2 path is no longer needed (vpdpbusd has the i16-pair-sum
/// baked in), freeing one ymm.
#[allow(clippy::too_many_arguments)]
fn emit_quant_matmul_q8q8_n1_body_vnni(
    em: &mut Emitter,
    p_w: Reg,
    p_a: Reg,
    p_out: Reg,
    m: i32,
    k: i32,
    abi: Abi,
    vnni: VnniForm,
) {
    let k_blocks = k / 32;
    let row_w_bytes = k_blocks * 34;

    // ---- prologue ----
    push_r64(em, Reg::R12);
    push_r64(em, Reg::R13);
    push_r64(em, Reg::R14);
    push_r64(em, Reg::R15);
    push_r64(em, Reg::RBX);
    let extra = if abi == Abi::WinX64 { 40 } else { 8 };
    sub_ri32(em, Reg::RSP, extra);

    mov_rr(em, Reg::R12, p_w);
    mov_rr(em, Reg::R13, p_a);
    mov_rr(em, Reg::R14, p_out);

    // R15 = i (output row), R11 = kb (block index).
    xor_rr(em, Reg::R15);
    let i_loop = em.len();
    cmp_ri32(em, Reg::R15, m);
    let jge_i = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    // fp32 row accumulator (8 lanes).
    vxorps_zero(em, Ymm(0));

    xor_rr(em, Reg::R11);
    let kb_loop = em.len();
    cmp_ri32(em, Reg::R11, k_blocks);
    let jge_kb = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    // Block byte offsets.
    mov_rr(em, Reg::RAX, Reg::R15);
    imul_rri32(em, Reg::RAX, Reg::RAX, row_w_bytes);
    mov_rr(em, Reg::RDX, Reg::R11);
    imul_rri32(em, Reg::RDX, Reg::RDX, 34);
    add_rr(em, Reg::RAX, Reg::RDX);

    mov_rr(em, Reg::R10, Reg::R11);
    imul_rri32(em, Reg::R10, Reg::R10, 34);

    // Load weight bytes and act bytes (skip 2-byte d header).
    vmovdqu_load(em, Ymm(1), Reg::R12, Some((Reg::RAX, Scale::S1)), 2);
    vmovdqu_load(em, Ymm(2), Reg::R13, Some((Reg::R10, Scale::S1)), 2);

    // |w| (u8) and a*sign(w) (i8) for the u8×i8 expected by vpdpbusd.
    vpsignb_reg(em, Ymm(3), Ymm(1), Ymm(1));
    vpsignb_reg(em, Ymm(2), Ymm(2), Ymm(1));

    // ymm5 accumulates 8 i32 partial dot-products for this block.
    // (Re-zeroed per block; we sum across blocks in fp32 via ymm0.)
    vxorps_zero(em, Ymm(5));
    vnni.emit(em, Ymm(5), Ymm(3), Ymm(2));

    // i32 → fp32 (8 lanes).
    vcvtdq2ps(em, Ymm(5), Ymm(5));

    // scalar d_w * d_a → broadcast.
    vmovd_load(em, Ymm(1), Reg::R12, Some((Reg::RAX, Scale::S1)), 0);
    vcvtph2ps_xmm(em, Ymm(1), Ymm(1));
    vmovd_load(em, Ymm(2), Reg::R13, Some((Reg::R10, Scale::S1)), 0);
    vcvtph2ps_xmm(em, Ymm(2), Ymm(2));
    vmulss_xmm(em, Ymm(1), Ymm(1), Ymm(2));
    vbroadcastss_xmm(em, Ymm(4), Ymm(1));

    // ymm0 += ymm5 * ymm4
    vfmadd231ps_reg(em, Ymm(0), Ymm(5), Ymm(4));

    inc_r(em, Reg::R11);
    let jmp_kb = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp_kb, kb_loop);

    let kb_done = em.len();
    patch_rel32(em, jge_kb, kb_done);

    // Horizontal reduce ymm0 (8 fp32 lanes) -> scalar in xmm0[0], store.
    vhaddps_ymm(em, Ymm(0), Ymm(0), Ymm(0));
    vhaddps_ymm(em, Ymm(0), Ymm(0), Ymm(0));
    vextractf128_xmm(em, Ymm(1), Ymm(0), 1);
    vaddss_xmm(em, Ymm(0), Ymm(0), Ymm(1));
    movss_store(em, Xmm(0), Reg::R14, Some((Reg::R15, Scale::S4)), 0);

    inc_r(em, Reg::R15);
    let jmp_i = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp_i, i_loop);

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

/// Q8×Q8 fused matmul (N=1), VNNI multi-accumulator variant. Phase 7.O.
///
/// The single-accumulator Phase 7.N kernel had ymm0 alone receiving every
/// block's `vfmadd231ps` — for K_blocks=152 (Qwen FFN-down) that's a 4-cycle
/// latency chain × 152 iterations = ~600 cycles per row regardless of FMA
/// throughput. This variant unrolls the kb loop by 4 and distributes the
/// per-block contributions across four independent fp32 sub-accumulators
/// (ymm0..ymm3), the same trick Phase 7.G applied to the fp32 matmul kernel.
///
/// Requires `K_blocks % 4 == 0` (dispatcher checks). For Qwen2.5-0.5B all
/// decode matmul shapes satisfy this (K_blocks ∈ {28, 152}).
///
/// Register plan adds an i32-dot scratch in ymm6, which is callee-saved on
/// Win64 — saved at `[rsp + extra]` in the prologue, restored in epilogue.
/// Everything else still fits in ymm0..ymm5.
#[allow(clippy::too_many_arguments)]
fn emit_quant_matmul_q8q8_n1_body_vnni_4acc(
    em: &mut Emitter,
    p_w: Reg,
    p_a: Reg,
    p_out: Reg,
    m: i32,
    k: i32,
    abi: Abi,
    vnni: VnniForm,
) {
    let k_blocks = k / 32;
    debug_assert!(k_blocks % 4 == 0, "4-acc kernel requires K_blocks % 4 == 0");
    let row_w_bytes = k_blocks * 34;

    // ---- prologue ----
    push_r64(em, Reg::R12);
    push_r64(em, Reg::R13);
    push_r64(em, Reg::R14);
    push_r64(em, Reg::R15);
    push_r64(em, Reg::RBX);
    let extra = if abi == Abi::WinX64 { 40 } else { 8 };
    // Reserve 32 bytes for the ymm6 save-area on top of `extra` (shadow/align).
    sub_ri32(em, Reg::RSP, extra + 32);
    // Save ymm6 (Win64 callee-saved; on SysV it's caller-saved but a few extra
    // bytes of stack don't hurt). Disp = extra puts the save area above the
    // shadow space (or align padding on SysV). RSP base requires SIB form
    // (encoded as `[rsp*1 + rsp + disp]` since index=RSP means "no index").
    vmovups_store(em, Ymm(6), Reg::RSP, Some((Reg::RSP, Scale::S1)), extra);

    mov_rr(em, Reg::R12, p_w);
    mov_rr(em, Reg::R13, p_a);
    mov_rr(em, Reg::R14, p_out);

    xor_rr(em, Reg::R15);
    let i_loop = em.len();
    cmp_ri32(em, Reg::R15, m);
    let jge_i = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    // 4 fp32 row sub-accumulators.
    vxorps_zero(em, Ymm(0));
    vxorps_zero(em, Ymm(1));
    vxorps_zero(em, Ymm(2));
    vxorps_zero(em, Ymm(3));

    xor_rr(em, Reg::R11); // kb_base = 0, stepping by 4
    let kb_loop = em.len();
    cmp_ri32(em, Reg::R11, k_blocks);
    let jge_kb = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    // Block-quad base offsets: RAX for weights, R10 for activations.
    mov_rr(em, Reg::RAX, Reg::R15);
    imul_rri32(em, Reg::RAX, Reg::RAX, row_w_bytes);
    mov_rr(em, Reg::RDX, Reg::R11);
    imul_rri32(em, Reg::RDX, Reg::RDX, 34);
    add_rr(em, Reg::RAX, Reg::RDX);

    mov_rr(em, Reg::R10, Reg::R11);
    imul_rri32(em, Reg::R10, Reg::R10, 34);

    // 4 unrolled blocks. Each goes into a different fp32 sub-accumulator,
    // breaking the FMA dependency chain into 4 independent streams.
    for (j, acc) in [0i32, 1, 2, 3].into_iter().enumerate() {
        let acc = Ymm(acc as u8);
        let block_disp = (j as i32) * 34;
        let q8_disp = block_disp + 2; // skip 2-byte fp16 d header

        // Load weight and activation i8 bytes.
        vmovdqu_load(em, Ymm(4), Reg::R12, Some((Reg::RAX, Scale::S1)), q8_disp);
        vmovdqu_load(em, Ymm(5), Reg::R13, Some((Reg::R10, Scale::S1)), q8_disp);
        // vpsignb trick: ymm5 = a*sign(w); then ymm4 = |w|. Order matters — the
        // first vpsignb still sees the original weight in ymm4.
        vpsignb_reg(em, Ymm(5), Ymm(5), Ymm(4));
        vpsignb_reg(em, Ymm(4), Ymm(4), Ymm(4));

        // Per-block i32 dot in ymm6 (vxorps is a zero-idiom → free on modern CPUs).
        vxorps_zero(em, Ymm(6));
        vnni.emit(em, Ymm(6), Ymm(4), Ymm(5));
        vcvtdq2ps(em, Ymm(6), Ymm(6));

        // scale = fp16_to_fp32(d_w) * fp16_to_fp32(d_a), broadcast to ymm4.
        vmovd_load(
            em,
            Ymm(4),
            Reg::R12,
            Some((Reg::RAX, Scale::S1)),
            block_disp,
        );
        vcvtph2ps_xmm(em, Ymm(4), Ymm(4));
        vmovd_load(
            em,
            Ymm(5),
            Reg::R13,
            Some((Reg::R10, Scale::S1)),
            block_disp,
        );
        vcvtph2ps_xmm(em, Ymm(5), Ymm(5));
        vmulss_xmm(em, Ymm(4), Ymm(4), Ymm(5));
        vbroadcastss_xmm(em, Ymm(4), Ymm(4));

        // sub-acc[j] += ymm6 * ymm4
        vfmadd231ps_reg(em, acc, Ymm(6), Ymm(4));
    }

    // kb_base += 4
    add_ri32(em, Reg::R11, 4);
    let jmp_kb = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp_kb, kb_loop);

    let kb_done = em.len();
    patch_rel32(em, jge_kb, kb_done);

    // Combine the 4 sub-accumulators: ymm0 = (ymm0+ymm1) + (ymm2+ymm3)
    vaddps_reg(em, Ymm(0), Ymm(0), Ymm(1));
    vaddps_reg(em, Ymm(2), Ymm(2), Ymm(3));
    vaddps_reg(em, Ymm(0), Ymm(0), Ymm(2));

    // Horizontal reduce ymm0 (8 fp32 lanes) -> scalar in xmm0[0], store.
    vhaddps_ymm(em, Ymm(0), Ymm(0), Ymm(0));
    vhaddps_ymm(em, Ymm(0), Ymm(0), Ymm(0));
    vextractf128_xmm(em, Ymm(1), Ymm(0), 1);
    vaddss_xmm(em, Ymm(0), Ymm(0), Ymm(1));
    movss_store(em, Xmm(0), Reg::R14, Some((Reg::R15, Scale::S4)), 0);

    inc_r(em, Reg::R15);
    let jmp_i = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp_i, i_loop);

    let i_done = em.len();
    patch_rel32(em, jge_i, i_done);

    // ---- epilogue ----
    // Restore ymm6 from save area (RSP base → SIB form, see prologue note).
    vmovups_load(em, Ymm(6), Reg::RSP, Some((Reg::RSP, Scale::S1)), extra);
    add_ri32(em, Reg::RSP, extra + 32);
    pop_r64(em, Reg::RBX);
    pop_r64(em, Reg::R15);
    pop_r64(em, Reg::R14);
    pop_r64(em, Reg::R13);
    pop_r64(em, Reg::R12);
    ret(em);
}

/// Emit an RMSNorm function: `y = (x * inv_rms) * weight`.
///
/// Source IR:
/// ```text
///   v0 = Param(0) : tensor<f32, [H]>     # input
///   v1 = Param(1) : tensor<f32, [H]>     # learned scale
///   v2 = RmsNorm v0, v1, eps : tensor<f32, [H]>
///   return v2
/// ```
///
/// Algorithm (two passes over the vector):
/// 1. Sum of squares → reduce to scalar → `inv_rms = 1 / sqrt(sum/H + eps)`
/// 2. `y[i] = x[i] * inv_rms * weight[i]`
///
/// Scope: 1-D input only, `H % 8 == 0` (one 8-wide ymm tile fits the row).
fn emit_function_rms_norm(em: &mut Emitter, f: &Function, abi: Abi) -> Result<(), CodegenError> {
    let rms_idx = f
        .values
        .iter()
        .position(|v| matches!(v.op, Op::RmsNorm { .. }))
        .unwrap();
    let (x_id, w_id, eps) = match &f.values[rms_idx].op {
        Op::RmsNorm { x, weight, eps } => (*x, *weight, *eps),
        _ => unreachable!(),
    };

    let x_ty = &f.values[x_id.0 as usize].ty;
    let w_ty = &f.values[w_id.0 as usize].ty;
    if x_ty.dtype != DType::F32 || w_ty.dtype != DType::F32 {
        return Err(CodegenError::UnsupportedOp(
            "Phase 6.C.2: rms_norm only supports f32 inputs".into(),
        ));
    }
    if x_ty.shape.0.len() != 1 || w_ty.shape.0.len() != 1 {
        return Err(CodegenError::UnsupportedOp(
            "Phase 6.C.2: rms_norm requires 1-D input and weight".into(),
        ));
    }
    let h = match x_ty.shape.0[0] {
        Dim::Static(v) => v,
        _ => {
            return Err(CodegenError::ShapeError(
                "dynamic dims not supported".into(),
            ))
        }
    };
    if h % 8 != 0 {
        return Err(CodegenError::ShapeError(
            "Phase 6.C.2: H must be a multiple of 8".into(),
        ));
    }

    let x_param = match f.values[x_id.0 as usize].op {
        Op::Param { index } => index,
        _ => {
            return Err(CodegenError::UnsupportedOp(
                "rms_norm x must be Param".into(),
            ))
        }
    };
    let w_param = match f.values[w_id.0 as usize].op {
        Op::Param { index } => index,
        _ => {
            return Err(CodegenError::UnsupportedOp(
                "rms_norm weight must be Param".into(),
            ))
        }
    };

    let p_x = abi.param_reg(x_param);
    let p_w = abi.param_reg(w_param);
    let p_out = abi.param_reg(f.params.len() as u32);

    emit_rms_norm_body(em, p_x, p_w, p_out, h as i32, eps, abi);
    Ok(())
}

/// Pseudo-asm. R12=x_base, R13=w_base, R14=out_base. RCX=i.
///
/// ```text
/// ; --- pass 1: sum of squares ---
/// vxorps ymm0, ymm0, ymm0
/// xor rcx, rcx
/// loop1:
///   cmp rcx, H
///   jge done1
///   vmovups ymm1, [r12 + rcx*4]
///   vfmadd231ps ymm0, ymm1, ymm1     ; ymm0 += x * x
///   add rcx, 8
///   jmp loop1
/// done1:
///
/// ; --- reduce ymm0 to scalar xmm0 ---
/// vhaddps   ymm0, ymm0, ymm0          ; sum adjacent pairs
/// vhaddps   ymm0, ymm0, ymm0          ; sum again
/// vextractf128 xmm1, ymm0, 1
/// vaddss    xmm0, xmm0, xmm1          ; xmm0[0] = full sum
///
/// ; --- compute inv_rms ---
/// mov   eax, 1/H bits
/// vmovd xmm1, eax
/// vmulss xmm0, xmm0, xmm1             ; xmm0 = mean(x²)
/// mov   eax, eps bits
/// vmovd xmm1, eax
/// vaddss xmm0, xmm0, xmm1             ; + eps
/// vsqrtss xmm0, xmm0, xmm0
/// mov   eax, 1.0 bits
/// vmovd xmm1, eax
/// vdivss xmm0, xmm1, xmm0             ; inv_rms = 1.0 / sqrt(...)
/// vbroadcastss ymm2, xmm0             ; ymm2 = (inv_rms × 8)
///
/// ; --- pass 2: y = x * inv_rms * weight ---
/// xor rcx, rcx
/// loop2:
///   cmp rcx, H
///   jge done2
///   vmovups ymm0, [r12 + rcx*4]      ; x
///   vmovups ymm1, [r13 + rcx*4]      ; weight
///   vmulps  ymm0, ymm0, ymm2         ; x * inv_rms
///   vmulps  ymm0, ymm0, ymm1         ; * weight
///   vmovups [r14 + rcx*4], ymm0
///   add rcx, 8
///   jmp loop2
/// done2:
/// ```
fn emit_rms_norm_body(
    em: &mut Emitter,
    p_x: Reg,
    p_w: Reg,
    p_out: Reg,
    h: i32,
    eps: f32,
    abi: Abi,
) {
    // ---- prologue ----
    push_r64(em, Reg::R12);
    push_r64(em, Reg::R13);
    push_r64(em, Reg::R14);
    let extra = if abi == Abi::WinX64 { 40 } else { 8 };
    sub_ri32(em, Reg::RSP, extra);

    mov_rr(em, Reg::R12, p_x);
    mov_rr(em, Reg::R13, p_w);
    mov_rr(em, Reg::R14, p_out);

    // pass 1: sum of squares into ymm0
    vxorps_zero(em, Ymm(0));
    xor_rr(em, Reg::RCX);
    let l1 = em.len();
    cmp_ri32(em, Reg::RCX, h);
    let jge1 = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);
    // ymm1 = load x[rcx..rcx+8]
    vmovups_load(em, Ymm(1), Reg::R12, Some((Reg::RCX, Scale::S4)), 0);
    // ymm0 += ymm1 * ymm1
    vfmadd231ps_reg(em, Ymm(0), Ymm(1), Ymm(1));
    add_ri32(em, Reg::RCX, 8);
    let jmp1 = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp1, l1);
    let done1 = em.len();
    patch_rel32(em, jge1, done1);

    // reduce ymm0 → scalar xmm0
    vhaddps_ymm(em, Ymm(0), Ymm(0), Ymm(0));
    vhaddps_ymm(em, Ymm(0), Ymm(0), Ymm(0));
    vextractf128_xmm(em, Ymm(1), Ymm(0), 1);
    vaddss_xmm(em, Ymm(0), Ymm(0), Ymm(1));

    // xmm0 *= 1/H
    let inv_h_bits = (1.0f32 / h as f32).to_bits();
    mov_ri32(em, Reg::RAX, inv_h_bits);
    vmovd_load_reg(em, Ymm(1), Reg::RAX); // xmm1 = inv_h
    vmulss_xmm(em, Ymm(0), Ymm(0), Ymm(1));

    // xmm0 += eps
    mov_ri32(em, Reg::RAX, eps.to_bits());
    vmovd_load_reg(em, Ymm(1), Reg::RAX);
    vaddss_xmm(em, Ymm(0), Ymm(0), Ymm(1));

    // xmm0 = sqrt(xmm0)
    vsqrtss_xmm(em, Ymm(0), Ymm(0), Ymm(0));

    // xmm0 = 1.0 / xmm0
    mov_ri32(em, Reg::RAX, 1.0f32.to_bits());
    vmovd_load_reg(em, Ymm(1), Reg::RAX);
    vdivss_xmm(em, Ymm(0), Ymm(1), Ymm(0));

    // broadcast to ymm2
    vbroadcastss_xmm(em, Ymm(2), Ymm(0));

    // pass 2: y = x * inv_rms * weight
    xor_rr(em, Reg::RCX);
    let l2 = em.len();
    cmp_ri32(em, Reg::RCX, h);
    let jge2 = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);
    vmovups_load(em, Ymm(0), Reg::R12, Some((Reg::RCX, Scale::S4)), 0);
    vmovups_load(em, Ymm(1), Reg::R13, Some((Reg::RCX, Scale::S4)), 0);
    vmulps_reg(em, Ymm(0), Ymm(0), Ymm(2));
    vmulps_reg(em, Ymm(0), Ymm(0), Ymm(1));
    vmovups_store(em, Ymm(0), Reg::R14, Some((Reg::RCX, Scale::S4)), 0);
    add_ri32(em, Reg::RCX, 8);
    let jmp2 = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp2, l2);
    let done2 = em.len();
    patch_rel32(em, jge2, done2);

    // epilogue
    add_ri32(em, Reg::RSP, extra);
    pop_r64(em, Reg::R14);
    pop_r64(em, Reg::R13);
    pop_r64(em, Reg::R12);
    ret(em);
}

/// Helper: `vmovd xmm_dst, r32_src` — register-to-xmm 4-byte move.
/// Encoding: VEX.128.66.0F.W0 6E /r  (reg form, ModR/M.mod = 11).
fn vmovd_load_reg(em: &mut Emitter, dst: Ymm, src: Reg) {
    // We use the avx2_enc::vmovd_load helper for the memory form already; the
    // register form needs ModR/M.mod = 11. Inline it.
    use crate::avx2_enc::{OpcodeMap, Prefix};
    // 2-byte VEX path (no extension bits needed for xmm0..xmm7 + rax..rdi).
    let r = dst.high1();
    let b = src.high1();
    let three_byte = b != 0;
    let inv_vvvv = 0b1111u8;
    if three_byte {
        em.u8(0xC4);
        let b1 = (((!r) & 1) << 7) | (1 << 6) | (((!b) & 1) << 5) | (OpcodeMap::M0F as u8);
        // W=0, vvvv=1111, L=0, pp=P66
        let b2 = (inv_vvvv << 3) | (Prefix::P66 as u8);
        em.u8(b1);
        em.u8(b2);
    } else {
        em.u8(0xC5);
        // R̄=~r, vvvv=1111, L=0, pp=P66
        let b1 = (((!r) & 1) << 7) | (inv_vvvv << 3) | (Prefix::P66 as u8);
        em.u8(b1);
    }
    em.u8(0x6E);
    em.u8(0b11_000_000 | (dst.low3() << 3) | src.low3());
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
/// Decode-specialized AVX2 matmul: M=1, N % 32 == 0.
///
/// During autoregressive decode every weight matmul has shape `(1, K) @ (K, N)`.
/// The original `emit_matmul_avx2` (M ≥ 1, 1 ymm accumulator per j-block of 8)
/// runs one `vfmadd231ps` per kk step into a single live accumulator. With
/// 4-cycle FMA latency that bottlenecks at ~1 FMA per 4 cycles regardless of
/// throughput — even though Skylake/Zen3+ can issue 1–2 FMAs per cycle.
///
/// This path processes 32 output columns per j step using 4 independent ymm
/// accumulators (ymm0..ymm3). Each kk iteration emits 4 FMAs into 4
/// independent dependency chains, so the CPU's out-of-order engine can pipeline
/// them and approach the 1 FMA/cycle issue rate.
///
/// ```text
/// for j in 0..N step 32:
///   ymm0..ymm3 = 0
///   for kk in 0..K:
///     ymm4 = broadcast lhs[kk]            ; M=1, so a_idx = kk
///     rdx  = kk*N + j
///     ymm0 += ymm4 * [rhs + rdx*4 +  0]
///     ymm1 += ymm4 * [rhs + rdx*4 + 32]
///     ymm2 += ymm4 * [rhs + rdx*4 + 64]
///     ymm3 += ymm4 * [rhs + rdx*4 + 96]
///   store ymm0..ymm3 -> out[j .. j+32]
/// ```
fn emit_matmul_avx2_1xn_4acc(
    em: &mut Emitter,
    p_lhs: Reg,
    p_rhs: Reg,
    p_out: Reg,
    k: i32,
    n: i32,
    abi: Abi,
) {
    debug_assert!(n % 32 == 0, "1xN-4acc path requires N % 32 == 0");

    // ---- prologue ----
    // Same callee-saved set as the other AVX2 paths so the stack layout is
    // identical from the JIT's point of view. We don't actually clobber R15
    // here (no outer i loop) but pushing it keeps the prologue/epilogue
    // structurally consistent across the family.
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

    // RBX = j (column block, steps by 32). No outer i loop — M = 1.
    xor_rr(em, Reg::RBX);
    let j_loop = em.len();
    cmp_ri32(em, Reg::RBX, n);
    let jge_j = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    // 4 accumulators = 0
    vxorps_zero(em, Ymm(0));
    vxorps_zero(em, Ymm(1));
    vxorps_zero(em, Ymm(2));
    vxorps_zero(em, Ymm(3));

    xor_rr(em, Reg::RCX); // kk = 0
    let k_loop = em.len();
    cmp_ri32(em, Reg::RCX, k);
    let jge_k = jcc_rel32_placeholder(em, Cond::GreaterOrEqual);

    // ymm4 = broadcast lhs[kk]   ; M = 1 ⇒ a_idx = kk = RCX
    vbroadcastss(em, Ymm(4), Reg::R12, Some((Reg::RCX, Scale::S4)), 0);

    // RDX = kk*N + j
    mov_rr(em, Reg::RDX, Reg::RCX);
    imul_rri32(em, Reg::RDX, Reg::RDX, n);
    add_rr(em, Reg::RDX, Reg::RBX);

    // 4 independent FMA chains, 8 lanes each, contiguous in B.
    vfmadd231ps_mem(em, Ymm(0), Ymm(4), Reg::R13, Some((Reg::RDX, Scale::S4)), 0);
    vfmadd231ps_mem(
        em,
        Ymm(1),
        Ymm(4),
        Reg::R13,
        Some((Reg::RDX, Scale::S4)),
        32,
    );
    vfmadd231ps_mem(
        em,
        Ymm(2),
        Ymm(4),
        Reg::R13,
        Some((Reg::RDX, Scale::S4)),
        64,
    );
    vfmadd231ps_mem(
        em,
        Ymm(3),
        Ymm(4),
        Reg::R13,
        Some((Reg::RDX, Scale::S4)),
        96,
    );

    inc_r(em, Reg::RCX);
    let jmp_k = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp_k, k_loop);

    let k_done = em.len();
    patch_rel32(em, jge_k, k_done);

    // Store ymm0..ymm3 -> out[0, j..j+32]. M = 1 ⇒ out_idx = j = RBX.
    vmovups_store(em, Ymm(0), Reg::R14, Some((Reg::RBX, Scale::S4)), 0);
    vmovups_store(em, Ymm(1), Reg::R14, Some((Reg::RBX, Scale::S4)), 32);
    vmovups_store(em, Ymm(2), Reg::R14, Some((Reg::RBX, Scale::S4)), 64);
    vmovups_store(em, Ymm(3), Reg::R14, Some((Reg::RBX, Scale::S4)), 96);

    add_ri32(em, Reg::RBX, 32);
    let jmp_j = jmp_rel32_placeholder(em);
    patch_rel32(em, jmp_j, j_loop);

    let j_done = em.len();
    patch_rel32(em, jge_j, j_done);

    // ---- epilogue ----
    add_ri32(em, Reg::RSP, extra);
    pop_r64(em, Reg::RBX);
    pop_r64(em, Reg::R15);
    pop_r64(em, Reg::R14);
    pop_r64(em, Reg::R13);
    pop_r64(em, Reg::R12);
    ret(em);
}

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
