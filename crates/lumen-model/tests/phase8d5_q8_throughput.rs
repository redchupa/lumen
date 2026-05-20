//! Phase 8.D.5: throughput diagnostic for the Q8 prefill kernel.
//!
//! Phase 8.D.3 measured pp32 at 22.65 tok/s — 2.9× slower per-token than
//! tg32. Phase 8.D.4 ruled out attention (1% of the wall). The remaining
//! candidates are:
//!
//!   1. The Q8 N>1 kernel emitting slow code (correct but slow).
//!   2. Per-matmul activation/output transpose in weight_matmul_jit_batched.
//!   3. Cold JIT cache, instruction icache, etc.
//!
//! This test isolates (1): we call the raw Q8 kernels at the shapes the
//! Qwen2.5-0.5B forward pass actually uses, with no transpose, no model
//! plumbing, and a tight warm-up loop. The comparison is:
//!
//!   A) get_or_compile_q8(M, K, 32) invoked ONCE per timed iter
//!   B) get_or_compile_q8(M, K,  1) invoked 32× per timed iter
//!
//! Same total fp32 work. If A is meaningfully faster than B then batching
//! pays for itself at the kernel level and the regression must live in the
//! transpose or in JIT caching. If A is not faster (or slower), the N>1
//! emit path itself is the problem and the next phase is codegen, not
//! plumbing.

use std::time::Instant;

use lumen_jit::MatmulJitCache;
use lumen_runtime::quant::{quantize_q8_0, BlockQ8_0};

const QK: usize = 32;

fn make_q8_weights(m: usize, k: usize) -> Vec<BlockQ8_0> {
    assert!(k % QK == 0);
    let raw: Vec<f32> = (0..m * k)
        .map(|i| ((i % 17) as f32) * 0.03 - 0.2)
        .collect();
    let mut blocks = vec![BlockQ8_0 { d: 0, qs: [0i8; 32] }; (m * k) / QK];
    quantize_q8_0(&raw, &mut blocks);
    blocks
}

/// Per-shape comparison. `m` and `k` come from the actual Qwen2.5-0.5B
/// projections; we only measure shapes that route through Q8 matmul on
/// the forward path.
fn bench_one_shape(label: &str, m: usize, k: usize) {
    let weights = make_q8_weights(m, k);
    let mut cache = MatmulJitCache::new();

    // Compile both kernels up front so neither timed loop pays JIT cost.
    let f_n32 = cache
        .get_or_compile_q8(m as u32, k as u32, 32)
        .expect("compile N=32");
    let f_n1 = cache
        .get_or_compile_q8(m as u32, k as u32, 1)
        .expect("compile N=1");

    // Activations. N=32 path takes a [K, 32] slab; N=1 takes [K, 1] —
    // we reuse the first column of the N=32 slab so the data lives in
    // the same cache lines for both paths.
    let act_n32: Vec<f32> = (0..k * 32)
        .map(|i| ((i % 11) as f32) * 0.05 - 0.1)
        .collect();
    let act_n1: Vec<f32> = act_n32[..k].to_vec();

    let mut out_n32 = vec![0.0f32; m * 32];
    let mut out_n1 = vec![0.0f32; m];

    // Warm-up: get pages resident, branch predictors trained, etc.
    for _ in 0..3 {
        unsafe {
            f_n32(
                weights.as_ptr() as *const u8,
                act_n32.as_ptr(),
                out_n32.as_mut_ptr(),
            );
        }
        for _ in 0..32 {
            unsafe {
                f_n1(
                    weights.as_ptr() as *const u8,
                    act_n1.as_ptr(),
                    out_n1.as_mut_ptr(),
                );
            }
        }
    }

    // Measure. 50 iters is enough to cover ~ms-scale noise on these shapes.
    let iters = 50;

    let t = Instant::now();
    for _ in 0..iters {
        // SAFETY: kernel was compiled for exactly these dims.
        unsafe {
            f_n32(
                weights.as_ptr() as *const u8,
                act_n32.as_ptr(),
                out_n32.as_mut_ptr(),
            );
        }
    }
    let dur_n32 = t.elapsed();

    let t = Instant::now();
    for _ in 0..iters {
        for _ in 0..32 {
            // SAFETY: kernel was compiled for these dims.
            unsafe {
                f_n1(
                    weights.as_ptr() as *const u8,
                    act_n1.as_ptr(),
                    out_n1.as_mut_ptr(),
                );
            }
        }
    }
    let dur_n1x32 = t.elapsed();

    let per_n32_us = dur_n32.as_secs_f64() * 1e6 / iters as f64;
    let per_n1x32_us = dur_n1x32.as_secs_f64() * 1e6 / iters as f64;
    let ratio = per_n1x32_us / per_n32_us;

    eprintln!(
        "{:<10} M={:>5} K={:>5}  N=32 once: {:>8.1}µs   N=1×32: {:>8.1}µs   speedup: {:.2}×",
        label, m, k, per_n32_us, per_n1x32_us, ratio
    );
}

/// Run the bench across every matmul shape Qwen2.5-0.5B uses in its
/// transformer layers + lm_head.
#[test]
#[ignore = "throughput diagnostic — runs <2s, only useful when investigating prefill regression"]
fn q8_prefill_throughput_per_shape() {
    eprintln!("Phase 8.D.5: raw Q8 kernel throughput (no transpose, no plumbing)");
    eprintln!("compares same total work: 1× N=32 call vs 32× N=1 calls\n");

    // Qwen2.5-0.5B: hidden=896, q_dim=896 (14×64), kv_dim=128 (2×64),
    //               ffn_hidden=4864, vocab=151936.
    // M = d_out, K = d_in. All five layer projections, in forward order.
    bench_one_shape("qkv (Q)", 896, 896);
    bench_one_shape("qkv (KV)", 128, 896);
    bench_one_shape("wo", 896, 896);
    bench_one_shape("gate/up", 4864, 896);
    bench_one_shape("down", 896, 4864);
    // lm_head intentionally skipped here — at prefill time it's still
    // M=1 only (we project just the last row), so it doesn't exercise
    // the prefill kernel.
}
