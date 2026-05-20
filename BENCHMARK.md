# Lumen Benchmarks

Solid numbers, honestly reported. v0.5.0 cycle compressed the gap to
ggml from 1.376× (v0.4.0) to 1.304×, with single-threaded Lumen now
**11% faster than single-threaded ggml** at decode. The 8-thread gap
that remains is a memory-bandwidth-utilization difference + Q8 N>1
kernel codegen that hasn't been microarchitecturally tuned yet —
diagnosed across 11 measurement cycles, infrastructure left in place
for the next milestone.

## Test environment

- **Hardware**: AMD Zen 4 (Ryzen / 7000-series), Windows 11 Pro
- **Model**: `qwen2.5-0.5b-instruct-q8_0.gguf` (Qwen2.5-0.5B-Instruct,
  Q8_0 quantization, ~670 MB on disk)
- **Threads**: explicit per-row; default Lumen pool caps at 8.
- **Lumen build**: `cargo test --release`
- **llama.cpp build**: `b9174-bin-win-cpu-x64` (official AVX2 CPU release,
  ggml-cpu-zen4.dll backend)

## Headline numbers (v0.5.0)

### Single-thread decode — Lumen beats ggml here

| Path | Decode (tg32) tok/s | vs llama.cpp |
|---|---:|---:|
| Lumen naive Rust matmul | 2.91 | 13.9× slower |
| Lumen JIT (4×8 tile / 1×8 AVX2, 1-acc) | 4.43 | 9.1× slower |
| Lumen JIT (+ Q8-native fused matmul, Phase 7.D) | 17.97 | 2.25× slower |
| Lumen JIT (+ 4-acc Q8 N=1 kernel, Phase 7.G) | 29.10 | 1.39× slower |
| Lumen JIT (+ AVX2 SiLU/mul + RoPE precompute, 7.H/7.I) | ~31 | 1.30× slower |
| Lumen JIT v0.4.0 (shape-aware VNNI/fp32 dispatch) | 41.85 | 1.04× faster |
| **Lumen JIT v0.5.0 (atomic ThreadPool, 1t)** | **45.64** | **1.13× faster (Lumen wins)** |
| llama.cpp (ggml, 1 thread) | 40.40 | 1.0× |

### Multi-thread decode (8 threads)

| Path | Decode (tg32) tok/s | vs ggml @ 8t |
|---|---:|---:|
| Lumen JIT v0.3.0 (rayon, M×K_blocks ≥ 100K parallel) | ~56 | 1.62× slower |
| Lumen JIT v0.4.0 (per-shape VNNI/fp32 dispatch) | 65.47 | 1.376× slower |
| **Lumen JIT v0.5.0 (atomic ThreadPool + 8.E.1 fix)** | **67.38** | **1.304× slower** |
| llama.cpp (ggml, 8 threads, ggml-cpu-zen4) | 87.86 | 1.0× |

### Thread scaling (v0.5.0)

| Threads | Lumen v0.5.0 | ggml | Winner |
|---|---:|---:|---|
| 1 | **45.64** | 40.40 | **Lumen +13%** |
| 2 | 60.78 | 62.55 | ggml +3% |
| 4 | 66.45 | 86.33 | ggml +30% |
| 8 | 67.38 | 87.86 | ggml +30% |

Lumen 1→8t scaling: 1.48× (efficiency 18.5%). ggml: 2.17× (27.2%).
The 8t gap is entirely scaling efficiency, not kernel quality.

### Prefill (Phase 8.E.1, opt-in `LUMEN_PREFILL=1`)

| Path | pp32 tok/s |
|---|---:|
| v0.4.0 (no prefill — decode-by-token) | 65 |
| Phase 8.D.3 first prefill attempt | 22.65 (regression, see 8.D retro) |
| **Phase 8.E.1 (N=1 fan-out fix)** | **54.15** |
| llama.cpp pp128 | 739 |

Prefill is opt-in via `LUMEN_PREFILL=1`. Default behaviour unchanged.
Real prefill win waits for 8.E.2 (N>1 codegen rewrite).

Apples-to-apples sanity check: Lumen ~56 multi-thread is **1.36× faster
than ggml's single-thread 41.32**. Single-thread Lumen (~31) is 1.32×
slower than single-thread ggml. So our multi-thread scaling (1.74×) lags
ggml's (2.20×). Reducing that gap is the next milestone — most likely
from (1) better per-thread kernel work-to-overhead ratio on the smaller
matmuls, (2) Q8-native instead of dequant-on-the-fly inside the FMA
loop, and (3) eventually AVX-512 where available.

Phase 7.G applied the same multi-accumulator trick from 7.C to the Q8
N=1 kernel: 4 independent ymm accumulators (one per 8-element chunk of
each Q8 block) instead of one chain through ymm0. All five Q8 matmul
shapes in the decode forward (gate_up, down, lm_head, qkv, wo) got
between **1.87× and 1.95× faster** — essentially matching the theoretical
~2× from breaking the 4-FMA dependency chain, bounded by memory bandwidth
on the weight stream.

The 5.08 → 17.97 jump (**+3.5×, +254% vs the original 4.43**) in 7.D came
from keeping Q8_0 weights in their native layout end-to-end instead of
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
