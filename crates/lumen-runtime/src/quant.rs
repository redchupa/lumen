//! Quantization formats compatible with GGUF / llama.cpp.
//!
//! Phase 5.A scope: Q4_0 and Q8_0 reference implementations.
//!
//! ## Block layout
//!
//! `Q4_0` — 32 elements per block, 18 bytes:
//!
//! ```text
//! struct block_q4_0 {
//!     fp16 d;          // scale (delta)
//!     u8   qs[16];     // 32 packed 4-bit signed values
//! }
//! ```
//!
//! Each `qs[j]` byte stores two nibbles. The low nibble is element `[j+0]`,
//! the high nibble is element `[j+16]`. Both are interpreted as `(nibble - 8)`
//! (signed range -8..+7), then multiplied by the block's `d` scale.
//!
//! `Q8_0` — 32 elements per block, 34 bytes:
//!
//! ```text
//! struct block_q8_0 {
//!     fp16 d;
//!     i8   qs[32];
//! }
//! ```
//!
//! `y[i] = qs[i] * d`. No packing trickery.
//!
//! Layouts are byte-compatible with ggml's `block_q4_0` and `block_q8_0` so we
//! can mmap a GGUF file and reinterpret tensor data without copying.

/// Elements per quantization block. Standard ggml value.
pub const QK: usize = 32;

/// 18-byte Q4_0 block. Packed so it's exactly 18 bytes (matches ggml).
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct BlockQ4_0 {
    /// FP16 scale (bits, decoded via [`f16_bits_to_f32`]).
    pub d: u16,
    /// 32 × 4-bit signed values packed as 16 bytes.
    pub qs: [u8; 16],
}

const _: () = {
    assert!(std::mem::size_of::<BlockQ4_0>() == 18);
};

/// 34-byte Q8_0 block. Packed for ggml byte compatibility.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BlockQ8_0 {
    pub d: u16,
    pub qs: [i8; 32],
}

const _: () = {
    assert!(std::mem::size_of::<BlockQ8_0>() == 34);
};

/// Half-precision (FP16, IEEE 754 binary16) bit pattern → FP32.
///
/// Handles normals, zero, infinity, and NaN exactly. Subnormals are converted
/// via promotion (close enough for inference accuracy needs).
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (bits & 0x8000) as u32;
    let exp = (bits & 0x7C00) >> 10;
    let mant = (bits & 0x03FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // Signed zero.
            return f32::from_bits(sign << 16);
        }
        // Subnormal: value = mant * 2^-24. Convert by scaling.
        // 2^-24 = 5.960464477539063e-08
        let v = (mant as f32) * 5.960_464_5e-8_f32;
        return if sign != 0 { -v } else { v };
    }
    if exp == 0x1F {
        // Infinity or NaN: copy mant into the high bits.
        return f32::from_bits((sign << 16) | 0x7F800000 | (mant << 13));
    }
    // Normal: rebias exponent and shift mantissa.
    let exp_f32 = exp as u32 + (127 - 15);
    f32::from_bits((sign << 16) | (exp_f32 << 23) | (mant << 13))
}

/// Dequantize Q4_0 blocks into a flat fp32 buffer. `out.len()` must equal
/// `blocks.len() * QK`.
pub fn dequantize_q4_0(blocks: &[BlockQ4_0], out: &mut [f32]) {
    assert_eq!(out.len(), blocks.len() * QK, "output buffer size mismatch");
    for (i, block) in blocks.iter().enumerate() {
        let d_bits = block.d; // copy out of packed
        let d = f16_bits_to_f32(d_bits);
        let qs = block.qs; // copy
        let base = i * QK;
        for j in 0..16 {
            let lo = (qs[j] & 0x0F) as i32 - 8;
            let hi = ((qs[j] >> 4) & 0x0F) as i32 - 8;
            out[base + j] = (lo as f32) * d;
            out[base + j + 16] = (hi as f32) * d;
        }
    }
}

/// Dequantize Q8_0 blocks into a flat fp32 buffer.
pub fn dequantize_q8_0(blocks: &[BlockQ8_0], out: &mut [f32]) {
    assert_eq!(out.len(), blocks.len() * QK, "output buffer size mismatch");
    for (i, block) in blocks.iter().enumerate() {
        let d_bits = block.d;
        let d = f16_bits_to_f32(d_bits);
        let qs = block.qs;
        let base = i * QK;
        for j in 0..QK {
            out[base + j] = (qs[j] as f32) * d;
        }
    }
}

/// Quantize a fp32 slice into Q8_0 blocks. Used by tests to generate inputs
/// in a known-roundtripable form.
pub fn quantize_q8_0(input: &[f32], out: &mut [BlockQ8_0]) {
    assert_eq!(input.len(), out.len() * QK);
    for (i, block) in out.iter_mut().enumerate() {
        let chunk = &input[i * QK..(i + 1) * QK];
        let amax = chunk.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let d = amax / 127.0;
        let inv_d = if d != 0.0 { 1.0 / d } else { 0.0 };
        block.d = f32_to_f16_bits(d);
        for (j, &x) in chunk.iter().enumerate() {
            let q = (x * inv_d).round() as i32;
            block.qs[j] = q.clamp(-128, 127) as i8;
        }
    }
}

/// FP32 → FP16 bit pattern. Round-to-nearest-even with the common bit fiddle.
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 0xFF {
        // Inf or NaN.
        let m = if mant != 0 { 0x200 } else { 0 };
        return sign | 0x7C00 | m as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1F {
        // Overflow → inf.
        return sign | 0x7C00;
    }
    if new_exp <= 0 {
        // Subnormal/underflow. Approximate via flush-to-zero for simplicity.
        if new_exp < -10 {
            return sign;
        }
        // Restore the implicit leading 1 and right-shift.
        let m = (mant | 0x0080_0000) >> (1 - new_exp);
        // Round to nearest even.
        let rounded = (m + 0x1000) >> 13;
        return sign | rounded as u16;
    }
    // Normal: round mantissa with ties-to-even.
    let rounded = (mant + 0x1000) >> 13;
    let mut h_mant = (rounded as u16) & 0x03FF;
    let mut h_exp = new_exp as u16;
    if rounded & 0x0400 != 0 {
        // Mantissa overflowed into exponent.
        h_exp += 1;
        h_mant = 0;
        if h_exp >= 0x1F {
            return sign | 0x7C00;
        }
    }
    sign | (h_exp << 10) | h_mant
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_zero_one_inf() {
        assert_eq!(f16_bits_to_f32(0x0000), 0.0);
        // f16 1.0 = 0x3C00
        assert_eq!(f16_bits_to_f32(0x3C00), 1.0);
        // f16 -1.0 = 0xBC00
        assert_eq!(f16_bits_to_f32(0xBC00), -1.0);
        // f16 +inf = 0x7C00
        assert!(f16_bits_to_f32(0x7C00).is_infinite());
    }

    #[test]
    fn f32_to_f16_roundtrips() {
        for v in [0.0f32, 1.0, -1.0, 0.5, -0.5, 65504.0, -65504.0] {
            let bits = f32_to_f16_bits(v);
            let back = f16_bits_to_f32(bits);
            assert!(
                (v - back).abs() < (v.abs() + 1.0) * 1e-3,
                "v={} back={}",
                v,
                back
            );
        }
    }

    #[test]
    fn dequantize_q4_0_known_block() {
        // d = 1.0 (f16 0x3C00). qs[0] = 0x07 → low nibble 7-8 = -1, high nibble 0-8 = -8.
        // qs[1..16] all 0x88 → low 8-8=0, high 8-8=0.
        let mut block = BlockQ4_0 {
            d: 0x3C00,
            qs: [0x88; 16],
        };
        block.qs[0] = 0x07;
        let mut out = vec![0.0f32; 32];
        dequantize_q4_0(std::slice::from_ref(&block), &mut out);
        assert_eq!(out[0], -1.0); // low nibble of qs[0]
        assert_eq!(out[16], -8.0); // high nibble of qs[0]
        for j in 1..16 {
            assert_eq!(out[j], 0.0);
            assert_eq!(out[j + 16], 0.0);
        }
    }

    #[test]
    fn dequantize_q8_0_known_block() {
        // d = 0.5, qs all 4 → output 2.0 in all 32 slots.
        let mut qs = [0i8; 32];
        for q in qs.iter_mut() {
            *q = 4;
        }
        let block = BlockQ8_0 {
            d: f32_to_f16_bits(0.5),
            qs,
        };
        let mut out = vec![0.0f32; 32];
        dequantize_q8_0(std::slice::from_ref(&block), &mut out);
        for &v in &out {
            assert!((v - 2.0).abs() < 1e-3, "got {}", v);
        }
    }

    #[test]
    fn quantize_then_dequantize_q8_round_trips_within_tolerance() {
        // 32 floats in [-1, 1]. Quantize → Dequant should reproduce within
        // (range/254) ≈ 0.008.
        let input: Vec<f32> = (0..32).map(|i| ((i as f32) / 16.0) - 1.0).collect();
        let mut blocks = vec![BlockQ8_0 { d: 0, qs: [0; 32] }; 1];
        quantize_q8_0(&input, &mut blocks);
        let mut out = vec![0.0f32; 32];
        dequantize_q8_0(&blocks, &mut out);
        for (i, (g, w)) in out.iter().zip(input.iter()).enumerate() {
            assert!((g - w).abs() < 0.01, "i={}: got {} expected {}", i, g, w);
        }
    }
}
