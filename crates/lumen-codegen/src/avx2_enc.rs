//! AVX2 (256-bit) instruction encoder.
//!
//! AVX uses VEX prefixes instead of REX. Reference: Intel SDM Vol. 2A §2.3.
//!
//! ```text
//! 2-byte VEX: C5 [R̄ vvvv L pp]                  (only when X==0, B==0, W==0, opcode-map==0F)
//! 3-byte VEX: C4 [R̄ X̄ B̄ mmmmm] [W vvvv L pp]
//! ```
//!
//! Field meanings:
//! - R/X/B  : like REX.R/X/B (inverted in VEX, so 1 means low reg, 0 means high)
//! - W      : like REX.W (0 in nearly every AVX2 float op we emit)
//! - L      : 0 = 128-bit XMM operands, 1 = 256-bit YMM
//! - pp     : mandatory legacy prefix (00=none, 01=0x66, 10=0xF3, 11=0xF2)
//! - mmmmm  : opcode map (00001=0F, 00010=0F38, 00011=0F3A)
//! - vvvv   : 4-bit second source register (inverted: ymm0 = 1111)
//!
//! Phase 3.A scope: enough opcodes to vectorize the inner k-loop of matmul.

use crate::emit::Emitter;
use crate::x86_64_enc::{Reg, Scale};

/// 256-bit AVX register, 0..=15.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Ymm(pub u8);

impl Ymm {
    pub fn low3(self) -> u8 {
        self.0 & 0b111
    }
    pub fn high1(self) -> u8 {
        self.0 >> 3
    }
}

#[derive(Copy, Clone, Debug)]
pub enum OpcodeMap {
    M0F = 0b00001,   // 0F
    M0F38 = 0b00010, // 0F 38
    #[allow(dead_code)]
    M0F3A = 0b00011, // 0F 3A
}

#[derive(Copy, Clone, Debug)]
pub enum Prefix {
    None = 0b00,
    P66 = 0b01,
    PF3 = 0b10,
    #[allow(dead_code)]
    PF2 = 0b11,
}

/// Emit a 2-byte or 3-byte VEX prefix depending on which extension bits are set.
///
/// `r`, `x`, `b` are the raw extension bits (0 or 1, not inverted). This
/// function flips them as required by the VEX encoding.
#[allow(clippy::too_many_arguments)] // VEX has 8 logical fields, no way around it
fn emit_vex(
    em: &mut Emitter,
    r: u8,
    x: u8,
    b: u8,
    map: OpcodeMap,
    w: u8,
    vvvv: u8, // raw 4-bit register; we invert here
    l: u8,    // 0 = 128-bit, 1 = 256-bit
    pp: Prefix,
) {
    let inv_vvvv = (!vvvv) & 0b1111;
    let three_byte_needed = x != 0 || b != 0 || !matches!(map, OpcodeMap::M0F) || w != 0;
    if three_byte_needed {
        // 3-byte VEX
        em.u8(0xC4);
        let b1 = (((!r) & 1) << 7) | (((!x) & 1) << 6) | (((!b) & 1) << 5) | (map as u8);
        let b2 = (w << 7) | (inv_vvvv << 3) | (l << 2) | (pp as u8);
        em.u8(b1);
        em.u8(b2);
    } else {
        // 2-byte VEX: encodes R via inverted high bit
        em.u8(0xC5);
        let b1 = (((!r) & 1) << 7) | (inv_vvvv << 3) | (l << 2) | (pp as u8);
        em.u8(b1);
    }
}

/// `vxorps ymm_dst, ymm_dst, ymm_dst` — clears the register.
///
/// Encoding: VEX.NDS.256.0F.WIG 57 /r
pub fn vxorps_zero(em: &mut Emitter, dst: Ymm) {
    emit_vex(
        em,
        dst.high1(),
        0,
        dst.high1(),
        OpcodeMap::M0F,
        0,
        dst.0,
        1,
        Prefix::None,
    );
    em.u8(0x57);
    // ModR/M: mod=11, reg=dst.low3, rm=dst.low3
    em.u8(0b11_000_000 | (dst.low3() << 3) | dst.low3());
}

/// `vmovups ymm, [base + index*scale + disp32]` — 256-bit unaligned load.
///
/// Encoding: VEX.256.0F.WIG 10 /r
pub fn vmovups_load(em: &mut Emitter, dst: Ymm, base: Reg, index: Option<(Reg, Scale)>, disp: i32) {
    let x = index.map(|(i, _)| i.high1()).unwrap_or(0);
    emit_vex(
        em,
        dst.high1(),
        x,
        base.high1(),
        OpcodeMap::M0F,
        0,
        0,
        1,
        Prefix::None,
    );
    em.u8(0x10);
    emit_modrm_sib_disp32(em, dst.low3(), base, index, disp);
}

/// `vmovups [base + index*scale + disp32], ymm` — 256-bit unaligned store.
///
/// Encoding: VEX.256.0F.WIG 11 /r
pub fn vmovups_store(
    em: &mut Emitter,
    src: Ymm,
    base: Reg,
    index: Option<(Reg, Scale)>,
    disp: i32,
) {
    let x = index.map(|(i, _)| i.high1()).unwrap_or(0);
    emit_vex(
        em,
        src.high1(),
        x,
        base.high1(),
        OpcodeMap::M0F,
        0,
        0,
        1,
        Prefix::None,
    );
    em.u8(0x11);
    emit_modrm_sib_disp32(em, src.low3(), base, index, disp);
}

/// `vbroadcastss ymm, [base + index*scale + disp32]` — load one f32 and
/// replicate into all 8 lanes.
///
/// Encoding: VEX.256.66.0F38.W0 18 /r
pub fn vbroadcastss(em: &mut Emitter, dst: Ymm, base: Reg, index: Option<(Reg, Scale)>, disp: i32) {
    let x = index.map(|(i, _)| i.high1()).unwrap_or(0);
    emit_vex(
        em,
        dst.high1(),
        x,
        base.high1(),
        OpcodeMap::M0F38,
        0,
        0,
        1,
        Prefix::P66,
    );
    em.u8(0x18);
    emit_modrm_sib_disp32(em, dst.low3(), base, index, disp);
}

/// `vfmadd231ps ymm_acc, ymm_a, [base + index*scale + disp32]`
/// — `acc = acc + a * b` where b is loaded from memory.
///
/// Encoding: VEX.DDS.256.66.0F38.W0 B8 /r
/// dst = acc (ModR/M.reg), vvvv = a, rm = b
pub fn vfmadd231ps_mem(
    em: &mut Emitter,
    acc: Ymm,
    a: Ymm,
    base: Reg,
    index: Option<(Reg, Scale)>,
    disp: i32,
) {
    let x = index.map(|(i, _)| i.high1()).unwrap_or(0);
    emit_vex(
        em,
        acc.high1(),
        x,
        base.high1(),
        OpcodeMap::M0F38,
        0,
        a.0,
        1,
        Prefix::P66,
    );
    em.u8(0xB8);
    emit_modrm_sib_disp32(em, acc.low3(), base, index, disp);
}

/// `vfmadd231ps ymm_acc, ymm_a, ymm_b` — same but b is in a register.
pub fn vfmadd231ps_reg(em: &mut Emitter, acc: Ymm, a: Ymm, b: Ymm) {
    emit_vex(
        em,
        acc.high1(),
        0,
        b.high1(),
        OpcodeMap::M0F38,
        0,
        a.0,
        1,
        Prefix::P66,
    );
    em.u8(0xB8);
    em.u8(0b11_000_000 | (acc.low3() << 3) | b.low3());
}

/// `vmulps ymm_dst, ymm_src1, ymm_src2` — fallback when FMA isn't available
/// (we still emit it for testability).
///
/// Encoding: VEX.NDS.256.0F.WIG 59 /r
pub fn vmulps_reg(em: &mut Emitter, dst: Ymm, src1: Ymm, src2: Ymm) {
    emit_vex(
        em,
        dst.high1(),
        0,
        src2.high1(),
        OpcodeMap::M0F,
        0,
        src1.0,
        1,
        Prefix::None,
    );
    em.u8(0x59);
    em.u8(0b11_000_000 | (dst.low3() << 3) | src2.low3());
}

/// `vaddps ymm_dst, ymm_src1, ymm_src2`.
///
/// Encoding: VEX.NDS.256.0F.WIG 58 /r
pub fn vaddps_reg(em: &mut Emitter, dst: Ymm, src1: Ymm, src2: Ymm) {
    emit_vex(
        em,
        dst.high1(),
        0,
        src2.high1(),
        OpcodeMap::M0F,
        0,
        src1.0,
        1,
        Prefix::None,
    );
    em.u8(0x58);
    em.u8(0b11_000_000 | (dst.low3() << 3) | src2.low3());
}

/// `vmovd xmm, [base + index*scale + disp32]` — load 4 bytes into the low
/// lane of an XMM register. Used to bring a 16-bit fp16 (lower half) into the
/// FPU register file for subsequent `vcvtph2ps`.
///
/// Encoding: VEX.128.66.0F.W0 6E /r
pub fn vmovd_load(em: &mut Emitter, dst: Ymm, base: Reg, index: Option<(Reg, Scale)>, disp: i32) {
    let x = index.map(|(i, _)| i.high1()).unwrap_or(0);
    emit_vex(
        em,
        dst.high1(),
        x,
        base.high1(),
        OpcodeMap::M0F,
        0,
        0,
        0, // L=0 → 128-bit
        Prefix::P66,
    );
    em.u8(0x6E);
    emit_modrm_sib_disp32(em, dst.low3(), base, index, disp);
}

/// `vcvtph2ps xmm_dst, xmm_src` — convert the lower 4 fp16 lanes of `src` to
/// 4 fp32 lanes of `dst` (xmm form).
///
/// Encoding: VEX.128.66.0F38.W0 13 /r
pub fn vcvtph2ps_xmm(em: &mut Emitter, dst: Ymm, src: Ymm) {
    emit_vex(
        em,
        dst.high1(),
        0,
        src.high1(),
        OpcodeMap::M0F38,
        0,
        0,
        0, // L=0
        Prefix::P66,
    );
    em.u8(0x13);
    em.u8(0b11_000_000 | (dst.low3() << 3) | src.low3());
}

/// `vpmovsxbd ymm, [base + index*scale + disp32]` — sign-extend 8 i8 values
/// from memory into 8 i32 lanes of `dst`.
///
/// Encoding: VEX.256.66.0F38.W0 21 /r
pub fn vpmovsxbd_load(
    em: &mut Emitter,
    dst: Ymm,
    base: Reg,
    index: Option<(Reg, Scale)>,
    disp: i32,
) {
    let x = index.map(|(i, _)| i.high1()).unwrap_or(0);
    emit_vex(
        em,
        dst.high1(),
        x,
        base.high1(),
        OpcodeMap::M0F38,
        0,
        0,
        1, // L=1 → 256-bit
        Prefix::P66,
    );
    em.u8(0x21);
    emit_modrm_sib_disp32(em, dst.low3(), base, index, disp);
}

/// `vcvtdq2ps ymm_dst, ymm_src` — convert 8 i32 lanes to 8 fp32 lanes.
///
/// Encoding: VEX.256.0F.WIG 5B /r
pub fn vcvtdq2ps(em: &mut Emitter, dst: Ymm, src: Ymm) {
    emit_vex(
        em,
        dst.high1(),
        0,
        src.high1(),
        OpcodeMap::M0F,
        0,
        0,
        1,
        Prefix::None,
    );
    em.u8(0x5B);
    em.u8(0b11_000_000 | (dst.low3() << 3) | src.low3());
}

/// `vbroadcastss ymm_dst, xmm_src` — broadcast the lowest fp32 lane of `src`
/// to all 8 lanes of `dst`.
///
/// Encoding: VEX.256.66.0F38.W0 18 /r  (ModR/M.mod = 11, register form)
pub fn vbroadcastss_xmm(em: &mut Emitter, dst: Ymm, src: Ymm) {
    emit_vex(
        em,
        dst.high1(),
        0,
        src.high1(),
        OpcodeMap::M0F38,
        0,
        0,
        1,
        Prefix::P66,
    );
    em.u8(0x18);
    em.u8(0b11_000_000 | (dst.low3() << 3) | src.low3());
}

/// `vcvtsi2ss xmm_dst, xmm_src1, r32_src2` — convert i32 in `src2` to fp32 in
/// the low lane of `dst`. The upper lanes inherit from `src1`.
///
/// Encoding: VEX.NDS.LIG.F3.0F.W0 2A /r  (reg form)
pub fn vcvtsi2ss_xmm_r32(em: &mut Emitter, dst: Ymm, src1: Ymm, src2: Reg) {
    emit_vex(
        em,
        dst.high1(),
        0,
        src2.high1(),
        OpcodeMap::M0F,
        0,
        src1.0,
        0, // L=0 (LIG)
        Prefix::PF3,
    );
    em.u8(0x2A);
    em.u8(0b11_000_000 | (dst.low3() << 3) | src2.low3());
}

/// `vhaddps ymm_dst, ymm_src1, ymm_src2` — horizontal add of packed floats.
/// Used for the final reduce-step of an 8-wide sum-of-squares accumulator.
///
/// Encoding: VEX.NDS.256.F2.0F.WIG 7C /r
pub fn vhaddps_ymm(em: &mut Emitter, dst: Ymm, src1: Ymm, src2: Ymm) {
    emit_vex(
        em,
        dst.high1(),
        0,
        src2.high1(),
        OpcodeMap::M0F,
        0,
        src1.0,
        1,
        Prefix::PF2,
    );
    em.u8(0x7C);
    em.u8(0b11_000_000 | (dst.low3() << 3) | src2.low3());
}

/// `vextractf128 xmm_dst, ymm_src, imm8` — copy lane `imm8` (0 = lower,
/// 1 = upper) of a 256-bit `ymm` into an xmm register.
///
/// Encoding: VEX.256.66.0F3A.W0 19 /r ib
/// Operand mapping: ModR/M.reg = src (ymm), ModR/M.rm = dst (xmm).
pub fn vextractf128_xmm(em: &mut Emitter, dst: Ymm, src: Ymm, imm: u8) {
    emit_vex(
        em,
        src.high1(),
        0,
        dst.high1(),
        OpcodeMap::M0F3A,
        0,
        0,
        1,
        Prefix::P66,
    );
    em.u8(0x19);
    em.u8(0b11_000_000 | (src.low3() << 3) | dst.low3());
    em.u8(imm);
}

/// `vsqrtss xmm_dst, xmm_src1, xmm_src2` — scalar fp32 sqrt (low lane only).
///
/// Encoding: VEX.NDS.LIG.F3.0F.WIG 51 /r
pub fn vsqrtss_xmm(em: &mut Emitter, dst: Ymm, src1: Ymm, src2: Ymm) {
    emit_vex(
        em,
        dst.high1(),
        0,
        src2.high1(),
        OpcodeMap::M0F,
        0,
        src1.0,
        0,
        Prefix::PF3,
    );
    em.u8(0x51);
    em.u8(0b11_000_000 | (dst.low3() << 3) | src2.low3());
}

/// `vdivss xmm_dst, xmm_src1, xmm_src2` — scalar fp32 divide (low lane).
///
/// Encoding: VEX.NDS.LIG.F3.0F.WIG 5E /r
pub fn vdivss_xmm(em: &mut Emitter, dst: Ymm, src1: Ymm, src2: Ymm) {
    emit_vex(
        em,
        dst.high1(),
        0,
        src2.high1(),
        OpcodeMap::M0F,
        0,
        src1.0,
        0,
        Prefix::PF3,
    );
    em.u8(0x5E);
    em.u8(0b11_000_000 | (dst.low3() << 3) | src2.low3());
}

/// `vaddss xmm_dst, xmm_src1, xmm_src2` — scalar fp32 add (low lane).
///
/// Encoding: VEX.NDS.LIG.F3.0F.WIG 58 /r
pub fn vaddss_xmm(em: &mut Emitter, dst: Ymm, src1: Ymm, src2: Ymm) {
    emit_vex(
        em,
        dst.high1(),
        0,
        src2.high1(),
        OpcodeMap::M0F,
        0,
        src1.0,
        0,
        Prefix::PF3,
    );
    em.u8(0x58);
    em.u8(0b11_000_000 | (dst.low3() << 3) | src2.low3());
}

/// `vmulss xmm_dst, xmm_src1, xmm_src2` — scalar fp32 multiply (low lane only).
///
/// Encoding: VEX.NDS.LIG.F3.0F.WIG 59 /r
pub fn vmulss_xmm(em: &mut Emitter, dst: Ymm, src1: Ymm, src2: Ymm) {
    emit_vex(
        em,
        dst.high1(),
        0,
        src2.high1(),
        OpcodeMap::M0F,
        0,
        src1.0,
        0,
        Prefix::PF3,
    );
    em.u8(0x59);
    em.u8(0b11_000_000 | (dst.low3() << 3) | src2.low3());
}

// ============================================================================
// AVX2 integer ops (Phase 7.M — Q8-native int dot product)
// ============================================================================

/// `vmovdqu ymm, [base + index*scale + disp32]` — 256-bit unaligned integer
/// load. Same data path as `vmovups` on modern silicon but signals "integer"
/// to the rename engine; avoids the FP/INT bypass penalty when followed by
/// integer SIMD ops.
///
/// Encoding: VEX.256.F3.0F.WIG 6F /r
pub fn vmovdqu_load(em: &mut Emitter, dst: Ymm, base: Reg, index: Option<(Reg, Scale)>, disp: i32) {
    let x = index.map(|(i, _)| i.high1()).unwrap_or(0);
    emit_vex(
        em,
        dst.high1(),
        x,
        base.high1(),
        OpcodeMap::M0F,
        0,
        0,
        1,
        Prefix::PF3,
    );
    em.u8(0x6F);
    emit_modrm_sib_disp32(em, dst.low3(), base, index, disp);
}

/// `vpsignb ymm_dst, ymm_src1, ymm_src2` — for each byte lane:
///   if src2[i] < 0:  dst[i] = -src1[i]
///   if src2[i] == 0: dst[i] = 0
///   else:            dst[i] = src1[i]
///
/// Encoding: VEX.NDS.256.66.0F38.WIG 08 /r
pub fn vpsignb_reg(em: &mut Emitter, dst: Ymm, src1: Ymm, src2: Ymm) {
    emit_vex(
        em,
        dst.high1(),
        0,
        src2.high1(),
        OpcodeMap::M0F38,
        0,
        src1.0,
        1,
        Prefix::P66,
    );
    em.u8(0x08);
    em.u8(0b11_000_000 | (dst.low3() << 3) | src2.low3());
}

/// `vpmaddubsw ymm_dst, ymm_src1, ymm_src2` — multiply 32 (u8, i8) pairs and
/// horizontally sum adjacent pairs into 16 saturated i16 lanes:
///   dst_i16[i] = sat( src1_u8[2i]*src2_i8[2i] + src1_u8[2i+1]*src2_i8[2i+1] )
///
/// Note the asymmetry — `src1` is treated as unsigned bytes, `src2` as signed.
///
/// Encoding: VEX.NDS.256.66.0F38.WIG 04 /r
pub fn vpmaddubsw_reg(em: &mut Emitter, dst: Ymm, src1: Ymm, src2: Ymm) {
    emit_vex(
        em,
        dst.high1(),
        0,
        src2.high1(),
        OpcodeMap::M0F38,
        0,
        src1.0,
        1,
        Prefix::P66,
    );
    em.u8(0x04);
    em.u8(0b11_000_000 | (dst.low3() << 3) | src2.low3());
}

/// `vpmaddwd ymm_dst, ymm_src1, ymm_src2` — multiply 16 i16 pairs and sum
/// adjacent pairs into 8 i32 lanes:
///   dst_i32[i] = src1_i16[2i]*src2_i16[2i] + src1_i16[2i+1]*src2_i16[2i+1]
///
/// Encoding: VEX.NDS.256.66.0F.WIG F5 /r
pub fn vpmaddwd_reg(em: &mut Emitter, dst: Ymm, src1: Ymm, src2: Ymm) {
    emit_vex(
        em,
        dst.high1(),
        0,
        src2.high1(),
        OpcodeMap::M0F,
        0,
        src1.0,
        1,
        Prefix::P66,
    );
    em.u8(0xF5);
    em.u8(0b11_000_000 | (dst.low3() << 3) | src2.low3());
}

/// `vpaddd ymm_dst, ymm_src1, ymm_src2` — element-wise i32 add (8 lanes).
///
/// Encoding: VEX.NDS.256.66.0F.WIG FE /r
pub fn vpaddd_reg(em: &mut Emitter, dst: Ymm, src1: Ymm, src2: Ymm) {
    emit_vex(
        em,
        dst.high1(),
        0,
        src2.high1(),
        OpcodeMap::M0F,
        0,
        src1.0,
        1,
        Prefix::P66,
    );
    em.u8(0xFE);
    em.u8(0b11_000_000 | (dst.low3() << 3) | src2.low3());
}

/// `vpcmpeqd ymm_dst, ymm_src1, ymm_src2` — compare i32 lanes for equality,
/// setting each result lane to all-1s (= -1) or all-0s. The known idiom
/// `vpcmpeqd dst, dst, dst` produces an all-ones register cheaply (no input
/// dependency on the comparison).
///
/// Encoding: VEX.NDS.256.66.0F.WIG 76 /r
pub fn vpcmpeqd_reg(em: &mut Emitter, dst: Ymm, src1: Ymm, src2: Ymm) {
    emit_vex(
        em,
        dst.high1(),
        0,
        src2.high1(),
        OpcodeMap::M0F,
        0,
        src1.0,
        1,
        Prefix::P66,
    );
    em.u8(0x76);
    em.u8(0b11_000_000 | (dst.low3() << 3) | src2.low3());
}

/// `vpsrlw ymm_dst, ymm_src, imm8` — logical shift right each i16 lane by imm
/// bits. Used together with `vpcmpeqd self,self,self` to forge the 16-lane
/// `1` constant: all-ones >> 15 = 0x0001 per i16 lane.
///
/// Encoding: VEX.NDD.256.66.0F.WIG 71 /2 ib
/// (Note the /2 — the opcode extension goes in ModR/M.reg, and the
/// destination YMM is encoded in vvvv, not in ModR/M.reg.)
pub fn vpsrlw_imm8(em: &mut Emitter, dst: Ymm, src: Ymm, imm: u8) {
    emit_vex(
        em,
        0, // ModR/M.reg is the /2 opcode extension, not a real reg
        0,
        src.high1(),
        OpcodeMap::M0F,
        0,
        dst.0,
        1,
        Prefix::P66,
    );
    em.u8(0x71);
    // mod=11, reg=/2, rm=src.low3
    em.u8(0b11_000_000 | (2 << 3) | src.low3());
    em.u8(imm);
}

// ---- shared ModR/M+SIB+disp32 emission ----------------------------------

fn emit_modrm_sib_disp32(
    em: &mut Emitter,
    reg_low3: u8,
    base: Reg,
    index: Option<(Reg, Scale)>,
    disp: i32,
) {
    if let Some((idx, scale)) = index {
        em.u8(0b10_000_100 | (reg_low3 << 3));
        em.u8(((scale as u8) << 6) | (idx.low3() << 3) | base.low3());
        em.u32(disp as u32);
    } else {
        debug_assert!(base.low3() != 0b100, "use SIB form when base is RSP/R12");
        em.u8(0b10_000_000 | (reg_low3 << 3) | base.low3());
        em.u32(disp as u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc<F: Fn(&mut Emitter)>(f: F) -> Vec<u8> {
        let mut em = Emitter::new();
        f(&mut em);
        em.buf
    }

    /// `vxorps ymm0, ymm0, ymm0` — 2-byte VEX form because all extension bits are 0.
    /// Expected: C5 FC 57 C0
    ///   C5 FC: 2-byte VEX with vvvv=0000 inverted to 1111, L=1, pp=00, R̄=1
    ///   57   : opcode
    ///   C0   : ModR/M (mod=11, reg=000, rm=000)
    #[test]
    fn encodes_vxorps_ymm0() {
        assert_eq!(
            enc(|e| vxorps_zero(e, Ymm(0))),
            vec![0xC5, 0xFC, 0x57, 0xC0]
        );
    }

    /// `vmulps ymm0, ymm0, ymm1` — 2-byte VEX, dst=ymm0, src1=ymm0, src2=ymm1.
    /// Expected: C5 FC 59 C1
    #[test]
    fn encodes_vmulps_ymm0_ymm0_ymm1() {
        assert_eq!(
            enc(|e| vmulps_reg(e, Ymm(0), Ymm(0), Ymm(1))),
            vec![0xC5, 0xFC, 0x59, 0xC1]
        );
    }

    /// `vbroadcastss ymm0, [rcx + 0]` — 3-byte VEX (opcode map 0F 38).
    /// Expected: C4 E2 7D 18 81 00 00 00 00
    #[test]
    fn encodes_vbroadcastss_xmm0_from_rcx() {
        assert_eq!(
            enc(|e| vbroadcastss(e, Ymm(0), Reg::RCX, None, 0)),
            vec![0xC4, 0xE2, 0x7D, 0x18, 0x81, 0x00, 0x00, 0x00, 0x00]
        );
    }

    /// `vmovups ymm0, [rdx + 0]` — 2-byte VEX, opcode map 0F.
    /// Expected: C5 FC 10 82 00 00 00 00
    #[test]
    fn encodes_vmovups_load_from_rdx() {
        assert_eq!(
            enc(|e| vmovups_load(e, Ymm(0), Reg::RDX, None, 0)),
            vec![0xC5, 0xFC, 0x10, 0x82, 0x00, 0x00, 0x00, 0x00]
        );
    }

    /// `vfmadd231ps ymm2, ymm0, [r13 + rax*4 + 0]` — 3-byte VEX, REX.B set
    /// because r13 base is in the upper bank.
    #[test]
    fn encodes_vfmadd231ps_with_sib() {
        // Sanity: just make sure it produces a non-empty buffer with 0xB8 opcode.
        let bytes =
            enc(|e| vfmadd231ps_mem(e, Ymm(2), Ymm(0), Reg::R13, Some((Reg::RAX, Scale::S4)), 0));
        assert_eq!(bytes[0], 0xC4); // 3-byte VEX
        assert!(bytes.contains(&0xB8));
    }

    #[test]
    fn encodes_vcvtph2ps_xmm() {
        let bytes = enc(|e| vcvtph2ps_xmm(e, Ymm(0), Ymm(0)));
        assert_eq!(bytes[0], 0xC4); // 3-byte VEX (0F 38 map)
        assert!(bytes.contains(&0x13));
    }

    #[test]
    fn encodes_vpmovsxbd_load() {
        let bytes = enc(|e| vpmovsxbd_load(e, Ymm(1), Reg::RDX, None, 0));
        assert_eq!(bytes[0], 0xC4);
        assert!(bytes.contains(&0x21));
    }

    #[test]
    fn encodes_vcvtdq2ps() {
        let bytes = enc(|e| vcvtdq2ps(e, Ymm(1), Ymm(1)));
        assert!(bytes.contains(&0x5B));
    }

    #[test]
    fn encodes_vmovd_load() {
        let bytes = enc(|e| vmovd_load(e, Ymm(0), Reg::RCX, None, 0));
        assert!(bytes.contains(&0x6E));
    }

    #[test]
    fn encodes_vbroadcastss_xmm_to_ymm() {
        let bytes = enc(|e| vbroadcastss_xmm(e, Ymm(0), Ymm(0)));
        assert_eq!(bytes[0], 0xC4);
        assert!(bytes.contains(&0x18));
        // last byte is ModR/M with mod=11
        assert!(bytes[bytes.len() - 1] & 0b1100_0000 == 0b1100_0000);
    }

    #[test]
    fn encodes_vcvtsi2ss_xmm_r32() {
        // vcvtsi2ss xmm0, xmm0, eax
        let bytes = enc(|e| vcvtsi2ss_xmm_r32(e, Ymm(0), Ymm(0), Reg::RAX));
        // 2-byte VEX: C5 [R̄=1 vvvv=1111(~0) L=0 pp=10(F3)] = C5 FA, opcode 2A, ModR/M
        assert_eq!(bytes[0], 0xC5);
        assert!(bytes.contains(&0x2A));
    }

    #[test]
    fn encodes_vmulss_xmm() {
        // vmulss xmm0, xmm0, xmm0
        let bytes = enc(|e| vmulss_xmm(e, Ymm(0), Ymm(0), Ymm(0)));
        assert_eq!(bytes[0], 0xC5);
        assert!(bytes.contains(&0x59));
    }

    #[test]
    fn encodes_vhaddps_ymm() {
        // vhaddps ymm0, ymm0, ymm0
        let bytes = enc(|e| vhaddps_ymm(e, Ymm(0), Ymm(0), Ymm(0)));
        assert_eq!(bytes[0], 0xC5);
        assert!(bytes.contains(&0x7C));
    }

    #[test]
    fn encodes_vextractf128_xmm() {
        // vextractf128 xmm1, ymm0, 1
        let bytes = enc(|e| vextractf128_xmm(e, Ymm(1), Ymm(0), 1));
        assert_eq!(bytes[0], 0xC4); // 3-byte VEX (0F 3A)
        assert!(bytes.contains(&0x19));
        assert_eq!(*bytes.last().unwrap(), 1);
    }

    #[test]
    fn encodes_vsqrtss_xmm() {
        let bytes = enc(|e| vsqrtss_xmm(e, Ymm(0), Ymm(0), Ymm(0)));
        assert_eq!(bytes[0], 0xC5);
        assert!(bytes.contains(&0x51));
    }

    #[test]
    fn encodes_vdivss_xmm() {
        let bytes = enc(|e| vdivss_xmm(e, Ymm(0), Ymm(0), Ymm(0)));
        assert_eq!(bytes[0], 0xC5);
        assert!(bytes.contains(&0x5E));
    }

    #[test]
    fn encodes_vaddss_xmm() {
        let bytes = enc(|e| vaddss_xmm(e, Ymm(0), Ymm(0), Ymm(0)));
        assert_eq!(bytes[0], 0xC5);
        assert!(bytes.contains(&0x58));
    }

    #[test]
    fn encodes_vmovdqu_load() {
        // `vmovdqu ymm0, [rdx + 0]` — F3.0F map, opcode 6F.
        let bytes = enc(|e| vmovdqu_load(e, Ymm(0), Reg::RDX, None, 0));
        assert!(bytes[0] == 0xC5 || bytes[0] == 0xC4);
        assert!(bytes.contains(&0x6F));
    }

    #[test]
    fn encodes_vpsignb_reg() {
        // `vpsignb ymm0, ymm0, ymm0` — 66.0F38 map, opcode 08.
        let bytes = enc(|e| vpsignb_reg(e, Ymm(0), Ymm(0), Ymm(0)));
        assert_eq!(bytes[0], 0xC4); // 3-byte VEX (0F 38 map)
        assert!(bytes.contains(&0x08));
    }

    #[test]
    fn encodes_vpmaddubsw_reg() {
        // `vpmaddubsw ymm0, ymm0, ymm0` — 66.0F38 map, opcode 04.
        let bytes = enc(|e| vpmaddubsw_reg(e, Ymm(0), Ymm(0), Ymm(0)));
        assert_eq!(bytes[0], 0xC4); // 3-byte VEX (0F 38 map)
        assert!(bytes.contains(&0x04));
    }

    #[test]
    fn encodes_vpmaddwd_reg() {
        // `vpmaddwd ymm0, ymm0, ymm0` — 66.0F map, opcode F5.
        let bytes = enc(|e| vpmaddwd_reg(e, Ymm(0), Ymm(0), Ymm(0)));
        assert!(bytes[0] == 0xC5 || bytes[0] == 0xC4);
        assert!(bytes.contains(&0xF5));
    }

    #[test]
    fn encodes_vpaddd_reg() {
        // `vpaddd ymm0, ymm0, ymm0` — 66.0F map, opcode FE.
        let bytes = enc(|e| vpaddd_reg(e, Ymm(0), Ymm(0), Ymm(0)));
        assert!(bytes[0] == 0xC5 || bytes[0] == 0xC4);
        assert!(bytes.contains(&0xFE));
    }

    #[test]
    fn encodes_vpcmpeqd_reg() {
        // `vpcmpeqd ymm0, ymm0, ymm0` — 66.0F map, opcode 76.
        let bytes = enc(|e| vpcmpeqd_reg(e, Ymm(0), Ymm(0), Ymm(0)));
        assert!(bytes[0] == 0xC5 || bytes[0] == 0xC4);
        assert!(bytes.contains(&0x76));
    }

    #[test]
    fn encodes_vpsrlw_imm8() {
        // `vpsrlw ymm0, ymm0, 15` — 66.0F map, opcode 71 with /2 extension.
        let bytes = enc(|e| vpsrlw_imm8(e, Ymm(0), Ymm(0), 15));
        assert!(bytes[0] == 0xC5 || bytes[0] == 0xC4);
        assert!(bytes.contains(&0x71));
        assert_eq!(*bytes.last().unwrap(), 15);
    }
}
