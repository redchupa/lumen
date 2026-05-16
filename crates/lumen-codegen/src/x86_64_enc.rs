//! x86_64 instruction encoder. Only the subset Lumen actually emits.
//!
//! Encoding background (Intel SDM Vol. 2):
//!
//! ```text
//! [prefixes] [REX] [opcode] [ModR/M] [SIB] [displacement] [immediate]
//! ```
//!
//! - REX = 0x40 | (W<<3) | (R<<2) | (X<<1) | B
//!   - W=1  : 64-bit operand size
//!   - R    : extends ModR/M.reg to 4 bits  (high bit of `reg`)
//!   - X    : extends SIB.index             (high bit of `index`)
//!   - B    : extends ModR/M.rm or SIB.base (high bit of `rm`/`base`)
//! - ModR/M = (mod<<6) | (reg<<3) | rm
//! - SIB    = (scale<<6) | (index<<3) | base

use crate::emit::Emitter;

/// 64-bit general-purpose registers. Values are the 4-bit register numbers,
/// 0..=7 encode without a REX.B bit, 8..=15 require it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
#[allow(clippy::upper_case_acronyms)]
pub enum Reg {
    RAX = 0,
    RCX = 1,
    RDX = 2,
    RBX = 3,
    RSP = 4,
    RBP = 5,
    RSI = 6,
    RDI = 7,
    R8 = 8,
    R9 = 9,
    R10 = 10,
    R11 = 11,
    R12 = 12,
    R13 = 13,
    R14 = 14,
    R15 = 15,
}

impl Reg {
    pub fn idx(self) -> u8 {
        self as u8
    }
    /// Low 3 bits used in ModR/M.rm or SIB.base/index.
    pub fn low3(self) -> u8 {
        (self as u8) & 0b111
    }
    /// High bit, encoded in REX.B / REX.X / REX.R.
    pub fn high1(self) -> u8 {
        (self as u8) >> 3
    }
}

/// XMM registers, 0..=15.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Xmm(pub u8);

impl Xmm {
    pub fn low3(self) -> u8 {
        self.0 & 0b111
    }
    pub fn high1(self) -> u8 {
        self.0 >> 3
    }
}

/// 1-byte scale field used in SIB encoding.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Scale {
    S1 = 0,
    S2 = 1,
    S4 = 2,
    S8 = 3,
}

fn rex(w: bool, r: u8, x: u8, b: u8) -> u8 {
    0x40 | (u8::from(w) << 3) | ((r & 1) << 2) | ((x & 1) << 1) | (b & 1)
}

fn modrm(md: u8, reg: u8, rm: u8) -> u8 {
    ((md & 0b11) << 6) | ((reg & 0b111) << 3) | (rm & 0b111)
}

fn sib(scale: Scale, index: u8, base: u8) -> u8 {
    ((scale as u8) << 6) | ((index & 0b111) << 3) | (base & 0b111)
}

// ======================================================================
// Integer instructions
// ======================================================================

/// `mov dst, src` (64-bit register move).
pub fn mov_rr(em: &mut Emitter, dst: Reg, src: Reg) {
    em.u8(rex(true, src.high1(), 0, dst.high1()));
    em.u8(0x89);
    em.u8(modrm(0b11, src.low3(), dst.low3()));
}

/// `mov dst, imm64`.
pub fn mov_ri64(em: &mut Emitter, dst: Reg, imm: u64) {
    em.u8(rex(true, 0, 0, dst.high1()));
    em.u8(0xB8 + dst.low3());
    em.u64(imm);
}

/// `mov dst32, imm32` — 32-bit immediate. Useful for loading scalar fp32
/// constants (cast via `to_bits()`) before promoting to xmm with `vmovd`.
/// Encoding: `[REX.B] B8+r imm32`.
pub fn mov_ri32(em: &mut Emitter, dst: Reg, imm: u32) {
    if dst.high1() != 0 {
        em.u8(0x41); // REX.B
    }
    em.u8(0xB8 + dst.low3());
    em.u32(imm);
}

/// `xor dst, dst` (64-bit; sets dst = 0). Same as `xor edst, edst` but with REX.W.
pub fn xor_rr(em: &mut Emitter, dst: Reg) {
    em.u8(rex(true, dst.high1(), 0, dst.high1()));
    em.u8(0x31);
    em.u8(modrm(0b11, dst.low3(), dst.low3()));
}

/// `add dst, src` (64-bit).
pub fn add_rr(em: &mut Emitter, dst: Reg, src: Reg) {
    em.u8(rex(true, src.high1(), 0, dst.high1()));
    em.u8(0x01);
    em.u8(modrm(0b11, src.low3(), dst.low3()));
}

/// `add dst, imm32` (sign-extended to 64).
pub fn add_ri32(em: &mut Emitter, dst: Reg, imm: i32) {
    em.u8(rex(true, 0, 0, dst.high1()));
    em.u8(0x81);
    em.u8(modrm(0b11, 0, dst.low3()));
    em.u32(imm as u32);
}

/// `sub dst, imm32`.
pub fn sub_ri32(em: &mut Emitter, dst: Reg, imm: i32) {
    em.u8(rex(true, 0, 0, dst.high1()));
    em.u8(0x81);
    em.u8(modrm(0b11, 0b101, dst.low3()));
    em.u32(imm as u32);
}

/// `imul dst, src, imm32`.
pub fn imul_rri32(em: &mut Emitter, dst: Reg, src: Reg, imm: i32) {
    em.u8(rex(true, dst.high1(), 0, src.high1()));
    em.u8(0x69);
    em.u8(modrm(0b11, dst.low3(), src.low3()));
    em.u32(imm as u32);
}

/// `inc r64`.
pub fn inc_r(em: &mut Emitter, dst: Reg) {
    em.u8(rex(true, 0, 0, dst.high1()));
    em.u8(0xFF);
    em.u8(modrm(0b11, 0, dst.low3()));
}

/// `cmp dst, imm32`.
pub fn cmp_ri32(em: &mut Emitter, dst: Reg, imm: i32) {
    em.u8(rex(true, 0, 0, dst.high1()));
    em.u8(0x81);
    em.u8(modrm(0b11, 0b111, dst.low3()));
    em.u32(imm as u32);
}

/// Conditional jump near, 32-bit relative displacement. Returns the offset of
/// the displacement bytes so the caller can patch it once the target is known.
pub fn jcc_rel32_placeholder(em: &mut Emitter, cond: Cond) -> usize {
    em.u8(0x0F);
    em.u8(0x80 + cond as u8);
    let off = em.len();
    em.u32(0); // placeholder
    off
}

/// Unconditional jump near, 32-bit relative displacement. Returns disp offset.
pub fn jmp_rel32_placeholder(em: &mut Emitter) -> usize {
    em.u8(0xE9);
    let off = em.len();
    em.u32(0);
    off
}

/// Patch a previously-emitted rel32 displacement to land at `target_offset`.
pub fn patch_rel32(em: &mut Emitter, disp_offset: usize, target_offset: usize) {
    let next_ip = (disp_offset + 4) as i64;
    let rel = target_offset as i64 - next_ip;
    em.patch_u32(disp_offset, rel as u32);
}

#[derive(Copy, Clone, Debug)]
#[repr(u8)]
#[allow(dead_code)]
pub enum Cond {
    Below = 0x2,        // jb
    AboveOrEqual = 0x3, // jae
    Equal = 0x4,        // je
    NotEqual = 0x5,     // jne
    Less = 0xC,         // jl  (signed)
    GreaterOrEqual = 0xD,
    LessOrEqual = 0xE,
    Greater = 0xF,
}

/// `ret`.
pub fn ret(em: &mut Emitter) {
    em.u8(0xC3);
}

/// `movsx dst32, byte ptr [base + index*scale + disp32]` — sign-extend an 8-bit
/// value from memory into a 32-bit destination register. Encoding: `0F BE /r`.
pub fn movsx_r32_m8(em: &mut Emitter, dst: Reg, base: Reg, index: Option<(Reg, Scale)>, disp: i32) {
    // REX needed if any reg uses upper bank (dst high or base high or index high).
    let r = dst.high1();
    let x = index.map(|(i, _)| i.high1()).unwrap_or(0);
    let b = base.high1();
    if r != 0 || x != 0 || b != 0 {
        em.u8(0x40 | (r << 2) | (x << 1) | b);
    }
    em.u8(0x0F);
    em.u8(0xBE);
    if let Some((idx, scale)) = index {
        em.u8(modrm(0b10, dst.low3(), 0b100));
        em.u8(sib(scale, idx.low3(), base.low3()));
        em.u32(disp as u32);
    } else {
        debug_assert!(base.low3() != 0b100, "use SIB form when base is RSP/R12");
        em.u8(modrm(0b10, dst.low3(), base.low3()));
        em.u32(disp as u32);
    }
}

// ======================================================================
// SSE scalar single-precision instructions
// ======================================================================

/// `movss xmm, [base + index*scale + disp32]`. Pass `index = None` for no SIB.
pub fn movss_load(em: &mut Emitter, dst: Xmm, base: Reg, index: Option<(Reg, Scale)>, disp: i32) {
    em.u8(0xF3);
    emit_rex_for_sse_mem(em, dst, base, index);
    em.u8(0x0F);
    em.u8(0x10);
    emit_modrm_sib_disp32(em, dst.low3(), base, index, disp);
}

/// `movss [base + index*scale + disp32], xmm`.
pub fn movss_store(em: &mut Emitter, src: Xmm, base: Reg, index: Option<(Reg, Scale)>, disp: i32) {
    em.u8(0xF3);
    emit_rex_for_sse_mem(em, src, base, index);
    em.u8(0x0F);
    em.u8(0x11);
    emit_modrm_sib_disp32(em, src.low3(), base, index, disp);
}

/// `addss dst, [base + index*scale + disp32]`.
pub fn addss_load(em: &mut Emitter, dst: Xmm, base: Reg, index: Option<(Reg, Scale)>, disp: i32) {
    em.u8(0xF3);
    emit_rex_for_sse_mem(em, dst, base, index);
    em.u8(0x0F);
    em.u8(0x58);
    emit_modrm_sib_disp32(em, dst.low3(), base, index, disp);
}

/// `mulss dst, [base + index*scale + disp32]`.
pub fn mulss_load(em: &mut Emitter, dst: Xmm, base: Reg, index: Option<(Reg, Scale)>, disp: i32) {
    em.u8(0xF3);
    emit_rex_for_sse_mem(em, dst, base, index);
    em.u8(0x0F);
    em.u8(0x59);
    emit_modrm_sib_disp32(em, dst.low3(), base, index, disp);
}

/// `xorps dst, dst`. Used to zero an XMM register.
pub fn xorps_self(em: &mut Emitter, dst: Xmm) {
    if dst.high1() != 0 {
        em.u8(rex(false, dst.high1(), 0, dst.high1()));
    }
    em.u8(0x0F);
    em.u8(0x57);
    em.u8(modrm(0b11, dst.low3(), dst.low3()));
}

/// `mulss dst, src` (xmm-xmm).
pub fn mulss_xmm(em: &mut Emitter, dst: Xmm, src: Xmm) {
    em.u8(0xF3);
    if dst.high1() != 0 || src.high1() != 0 {
        em.u8(rex(false, dst.high1(), 0, src.high1()));
    }
    em.u8(0x0F);
    em.u8(0x59);
    em.u8(modrm(0b11, dst.low3(), src.low3()));
}

/// `addss dst, src` (xmm-xmm).
pub fn addss_xmm(em: &mut Emitter, dst: Xmm, src: Xmm) {
    em.u8(0xF3);
    if dst.high1() != 0 || src.high1() != 0 {
        em.u8(rex(false, dst.high1(), 0, src.high1()));
    }
    em.u8(0x0F);
    em.u8(0x58);
    em.u8(modrm(0b11, dst.low3(), src.low3()));
}

// ----- helpers ------------------------------------------------------------

fn emit_rex_for_sse_mem(em: &mut Emitter, reg: Xmm, base: Reg, index: Option<(Reg, Scale)>) {
    let r = reg.high1();
    let x = index.map(|(i, _)| i.high1()).unwrap_or(0);
    let b = base.high1();
    if r != 0 || x != 0 || b != 0 {
        em.u8(rex(false, r, x, b));
    }
}

fn emit_modrm_sib_disp32(
    em: &mut Emitter,
    reg_low3: u8,
    base: Reg,
    index: Option<(Reg, Scale)>,
    disp: i32,
) {
    // We always emit mod=10 (disp32) so encoding is uniform; the optimizer
    // can shrink to disp8 / no-disp later.
    if let Some((idx, scale)) = index {
        // Use SIB. rm = 0b100 = SIB.
        em.u8(modrm(0b10, reg_low3, 0b100));
        em.u8(sib(scale, idx.low3(), base.low3()));
        em.u32(disp as u32);
    } else {
        // No SIB. rm = base.low3().
        // Special case: when rm = 0b100 (RSP/R12) we'd accidentally request SIB;
        // for now, our backend never uses RSP/R12 as a base without an index,
        // so assert defensively.
        debug_assert!(base.low3() != 0b100, "use SIB form when base is RSP/R12");
        em.u8(modrm(0b10, reg_low3, base.low3()));
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

    #[test]
    fn encodes_ret() {
        assert_eq!(enc(ret), vec![0xC3]);
    }

    #[test]
    fn encodes_xor_rax_rax() {
        // 48 31 C0
        assert_eq!(enc(|e| xor_rr(e, Reg::RAX)), vec![0x48, 0x31, 0xC0]);
    }

    #[test]
    fn encodes_mov_rax_rcx() {
        // 48 89 C8  (mov rax, rcx — src is reg, dst is r/m)
        assert_eq!(
            enc(|e| mov_rr(e, Reg::RAX, Reg::RCX)),
            vec![0x48, 0x89, 0xC8]
        );
    }

    #[test]
    fn encodes_inc_rax() {
        assert_eq!(enc(|e| inc_r(e, Reg::RAX)), vec![0x48, 0xFF, 0xC0]);
    }

    #[test]
    fn encodes_cmp_rax_42() {
        // 48 81 F8 2A 00 00 00
        assert_eq!(
            enc(|e| cmp_ri32(e, Reg::RAX, 42)),
            vec![0x48, 0x81, 0xF8, 0x2A, 0x00, 0x00, 0x00]
        );
    }

    #[test]
    fn encodes_movss_load_rcx() {
        // movss xmm0, [rcx+0]; expected (no REX needed): F3 0F 10 81 00 00 00 00
        // We always emit disp32 form -> ModR/M=10 000 001
        assert_eq!(
            enc(|e| movss_load(e, Xmm(0), Reg::RCX, None, 0)),
            vec![0xF3, 0x0F, 0x10, 0x81, 0x00, 0x00, 0x00, 0x00]
        );
    }

    #[test]
    fn encodes_movss_store_r8() {
        // movss [r8+0], xmm0; needs REX.B for R8 base
        // F3 41 0F 11 80 00 00 00 00
        assert_eq!(
            enc(|e| movss_store(e, Xmm(0), Reg::R8, None, 0)),
            vec![0xF3, 0x41, 0x0F, 0x11, 0x80, 0x00, 0x00, 0x00, 0x00]
        );
    }

    #[test]
    fn encodes_jmp_rel32_placeholder_and_patch() {
        let mut em = Emitter::new();
        let off = jmp_rel32_placeholder(&mut em);
        let target = em.len();
        patch_rel32(&mut em, off, target); // jump to next instruction = 0 rel
                                           // E9 00 00 00 00
        assert_eq!(em.buf, vec![0xE9, 0x00, 0x00, 0x00, 0x00]);
    }
}
