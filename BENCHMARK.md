# Lumen Benchmarks

Solid numbers, honestly reported. The runtime is at the bottom of its
performance curve — we just got to "real LLM inference runs end-to-end"
and only the Phase 3.C 4×8 register tile is wired into the main path so
far. ggml has had five-plus years of cache-blocking, quantized-matmul,
and microkernel tuning. The gap is meaningful and we own it.

## Test environment

- **Hardware**: Windows 11 Pro (host CPU detected by both binaries; both
  run in default release mode)
- **Model**: `qwen2.5-0.5b-instruct-q8_0.gguf` (Qwen2.5-0.5B-Instruct,
  Q8_0 quantization, ~640 MB on disk)
- **Threads**: `1` (single-thread on both sides for an apples-to-apples
  scalar-vs-scalar comparison; Lumen does not yet ship multi-thread
  matmul)
- **Lumen build**: `cargo test --release`
- **llama.cpp build**: `b9174-bin-win-cpu-x64` (official AVX2 CPU release)

## Headline numbers

| Path | Decode (tg32) tok/s | vs llama.cpp |
|---|---:|---:|
| Lumen naive Rust matmul | 2.91 | 14.2× slower |
| Lumen JIT (4×8 tile / 1×8 AVX2, 1-acc) | 4.43 | 9.3× slower |
| Lumen JIT (+ 1×N 4-acc decode path, Phase 7.C) | 5.08 | 8.1× slower |
| **Lumen JIT (+ Q8-native fused matmul, Phase 7.D)** | **17.97** | **2.30× slower** |
| llama.cpp (ggml) | 41.32 | 1.0× |

The 5.08 → 17.97 jump (**+3.5×, +254% vs the original 4.43**) comes from
keeping Q8_0 weights in their native layout end-to-end instead of
dequantizing to F32 at load time:

- weights stay 1× as many bytes (Q8: 1.0625 B/elt vs F32: 4 B/elt) — 4×
  the cache footprint freed
- the per-matmul fp32 dequant pass is gone
- a new decode kernel (`emit_quant_matmul_q8_n1_body`) reads Q8 blocks,
  rescales by the per-block fp16 `d`, and multiplies through F32
  activations in K-direction SIMD with a horizontal reduction tail
- `transpose_for_jit` drops from 1.94s to 700ns — the F32 weights that
  needed transposing are now Q8 weights that don't

Tokens remain bit-identical to the naive path (the new path is
mathematically equivalent to dequant-then-matmul; the fp32 dequant just
happens implicitly inside the inner loop).

llama.cpp also reports a `pp128 = 160.82 tok/s` for prompt processing
(batched 128 tokens). Lumen has no batched prefill yet — every prompt
token goes through the decode path one at a time, so the equivalent
column is just `decode tok/s × prompt_len`.

## What the numbers mean

**Lumen JIT is 1.52× faster than naive Rust matmul on the same forward
path** (2.91 → 4.43 tok/s). The improvement is bounded because:

1. **M = 1 in decode**: every weight matmul during single-token decode
   is shape `(1, K, N)`. That falls through to the 1×8 AVX2 lane; the
   4×8 register tile (which gets the synthetic 19× speedup vs scalar in
   Phase 3.C) needs `M % 4 == 0`. Prefill on ≥4 tokens at a time would
   exercise the tile and pull a wider gap.
2. **The naive Rust baseline is already hot**: LLVM in release mode
   auto-vectorizes the inner loop reasonably well. The cold-Rust →
   1×8-AVX2 jump is closer to 2× than 8×.
3. **Non-matmul ops carry weight**: attention (Q·K^T, softmax, attn·V),
   RoPE, RMSNorm, residual adds. Phase 6 uses the pure-Rust references
   for these, on the Hybrid-B principle that they're not the hot path
   anyway.

**llama.cpp is 9.3× faster than Lumen JIT.** The honest answer for why:

1. **Q8-native matmul**: ggml multiplies through the quantized layout
   directly. Lumen dequantizes Q8_0 → fp32 at load time and then does
   fp32 × fp32 matmul. We pay 4× the memory bandwidth and lose the
   `vpmaddubsw` / `vpdpbusd` integer SIMD instructions.
2. **Cache blocking**: ggml's GEMM tiles for L1 / L2 / L3. Lumen has
   register tiling (4×8) but no outer cache-block loop yet — performance
   stays high until the working set blows L2 and then drops.
3. **Specialized attention kernels**: ggml has hand-tuned attention
   passes (flash-style). Lumen runs the textbook formulation in Rust.
4. **More mature AVX2 use**: prefetch instructions, instruction
   scheduling that hits the FMA-2-per-cycle peak, etc. We get to ~50%
   of theoretical peak on a tight standalone matmul (Phase 3.C); ggml
   gets closer to 80%+ in the actual forward.

None of these are mysteries. They are the v1.0 → v1.1 work list.

## Methodology

### llama.cpp

```sh
llama-bench.exe -m qwen2.5-0.5b-instruct-q8_0.gguf -p 128 -n 32 -t 1
```

Reports two columns: `pp128` (prompt processing, 128 tokens batched)
and `tg32` (text generation, 32 tokens autoregressive). Both numbers
are mean ± stddev across multiple iterations.

### Lumen

Two `#[ignore]` integration tests in
`crates/lumen-model/tests/e2e_qwen_load.rs`:

```sh
cargo test -p lumen-model --release --test e2e_qwen_load \
    qwen_bench_tg32_naive -- --ignored --nocapture

cargo test -p lumen-model --release --test e2e_qwen_load \
    qwen_bench_tg32_jit -- --ignored --nocapture
```

Both load Qwen2.5-0.5B-Q8_0 from disk, encode "안녕" as a 2-token
prompt, then run `generate_greedy` / `generate_greedy_jit` for 32
decode tokens. Wall-clock time / 32 = the reported tok/s.

## Correctness across paths

A separate `#[ignore]` test
(`qwen_generate_jit_matches_naive_and_speed`) runs the same prompt
through both paths and asserts the token sequences match exactly. The
JIT path produces bit-identical tokens to the naive path — the speedup
isn't paid for in quality.

For Korean output specifically:

```
prompt: "안녕"
naive (1.63s / 3 tokens): ids [91145, 11, 134561] → "안녕하세요, 저는"
JIT   (1.12s / 3 tokens): ids [91145, 11, 134561] → "안녕하세요, 저는"
```

Same tokens; same text.

## Roadmap from here (where the next 10× lives)

1. **Cache blocking** for big matmul (lm_head 896 × 151,936, FFN
   gate/up/down 896 × 4864). Outer block loop over the inner 4×8
   register tile. Expected: tg32 → ~10 tok/s.
2. **Quantized matmul kernels**: Q8_0 × F32 fused (Phase 5.C is the
   prototype). Skip the fp32 dequant pass. Expected: another ~2× on
   memory-bound shapes.
3. **Specialized attention**: flash-style fused softmax. Expected:
   medium impact at this model size; big at longer context.
4. **AVX-512** path on capable CPUs. Expected: 1.5-2× depending on
   hardware (only some consumer CPUs have it).
5. **Multi-thread** prefill at minimum (decode parallelism is harder).
   Expected: near-linear up to physical core count.

The honest projection: with the first three items landed, Lumen should
sit in the same order of magnitude as llama.cpp on single-thread CPU
fp32-equivalent work. Catching it on Q8-native multi-thread is a
longer road.
