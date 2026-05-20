# Why prefill batching wasn't the answer (and what was) — Phase 8.D + 8.E.1

> Lumen build log #15 (EN). v0.5 cycle, third bundle. The hypothesis was plumbing;
> the real cause was one layer below in the kernel codegen. Tenth measurement-driven
> negative, first positive diagnosis, and a partial fix in the same cycle.

## TL;DR

- v0.5 standing after Phase 8.A: Lumen 8t **67.4 tok/s** vs ggml 8t **87.9 tok/s** — gap **1.304×**. Diagnosis: memory-bw utilization.
- Hypothesis: ggml hides the bw wall via prefill (`pp128 = 739 tok/s`). Adding prefill batching to Lumen should give a 5–10× speedup on prompts.
- Phase 8.D landed the plumbing — Q8 `N>1` correctness, `weight_matmul_jit_batched`, `forward_layer_prefill_jit`, chunked dispatch in `generate_greedy_jit`. All bit-exact vs decode-by-token.
- Measured **pp32 = 22.65 tok/s** vs `tg32 = 65 tok/s`. Per-token **2.9× regression**, opposite direction.
- First retro guess (attention `O(seq²)`) was wrong: a 50-line micro-bench showed attention is **0.66%** of pp32 wall.
- Real cause (Phase 8.D.5): the `N>1` Q8 codegen is **2.3–2.6× slower** than `N=1` called `N` times. The Phase 7.G 4-accumulator YMM tuning only lives on the `N=1` path.
- Phase 8.E.1 quick fix: dispatch `N>1` as a fan-out over `N` separate `N=1` calls. **pp32 = 22.65 → 54.15 tok/s (+2.39×)**. Still 17% below `tg32`; a proper `N>1` codegen rewrite is the next phase.
- Tokens remain bit-identical to v0.4.0 across every commit.

## Starting point

Phase 8.A reworked the ThreadPool (atomic claims, no per-task locking) and pulled the gap from `1.376× → 1.304×`. The 8t win was inside noise, but the diagnosis underneath was clear: **memory-bw utilization, not raw FLOPs**.

That made the next question obvious:

> If we're bw-bound, what happens on a workload that reuses each weight `N` times instead of once?

Decode streams the full **670 MB** of weights every token — DRAM read per token, hard wall. Prefill batches `N` prompt tokens into one matmul, so `weight_bytes / FLOP` drops by `N`. Same kernel, same hardware, different arithmetic intensity.

ggml on the same model, 8t:

| metric | ggml |
| --- | ---: |
| `tg128` (decode) | 87.9 tok/s |
| `pp128` (prefill) | 739 tok/s |

That's a **8.4× ratio**. ggml is essentially showing us where prefill weight-reuse lands when the kernel can take it.

Lumen v0.4.0:

| metric | Lumen |
| --- | ---: |
| `tg32` | 65 tok/s |
| `pp32` | 65 tok/s (no prefill — decode-by-token) |

ggml prefill was running about `11×` our decode. The clearest win candidate in the v0.5 bundle.

## The hypothesis

> **Prefill batching is the answer.**

Logic was trivial:

1. The matmul kernel already exists (decode path).
2. Feed it batched activations and let the inner loop process `N` rows.
3. Plumb it, and the 5–10× should fall out for free.

The first mistake of this phase: not stress-testing assumption #2 before committing to the plumbing.

## Phase 8.D — plumbing

### 8.D.1 — Q8 `N>1` kernel correctness — [`f44face`](https://github.com/redchupa/lumen/commit/f44face)

The JIT codegen had `N=1` assumptions sprinkled across multiple call sites. Generalize the row dimension. Validate against a naive Rust reference, bit-exact, across:

- 6 shapes: qkv (Q), qkv (KV), wo, gate, up, down
- 3 batch sizes: `N ∈ {1, 8, 32}`
- 3 integration tests: single-token forward, batched-token forward, per-layer parity

### 8.D.1-2 — `weight_matmul_jit_batched` wrapper — [`061579d`](https://github.com/redchupa/lumen/commit/061579d)

Old: `weight_matmul_jit(F32, weight, ...)`, single-row.
New: `weight_matmul_jit_batched(F32_2D[N×K], weight, ...)`, dispatches per-dtype (F32 / Q8) with an `N == 1` short-circuit so the decode path can never regress.

### 8.D.2 — `forward_layer_prefill_jit` — [`a8e7601`](https://github.com/redchupa/lumen/commit/a8e7601)

Counterpart to `forward_layer_jit`. Input `tokens[N×K]`, output `tokens[N×K]`. Attention needed a rewrite (`Q @ K^T` becomes `[N × seq_total]` not `[N × N]` because of the KV cache merge); everything else lifts cleanly to the `N` dimension.

Correctness: layer output is bit-exact vs `N` sequential decode-by-token passes.

### 8.D.3 — chunked dispatch + first measurement — [`9d46907`](https://github.com/redchupa/lumen/commit/9d46907)

`generate_greedy_jit` splits the prompt into `N=32` chunks, calls `forward_prefill_jit` per chunk, then enters the decode loop. Same shape as ggml's prompt processing.

Then I measured.

## Measurement — the regression

Zen 4, Qwen2.5-0.5B-Q8\_0, 8 threads, 32-token prompt:

```
pp32 (prefill 32 tokens + decode 1):  1.413 s → 22.65 tok/s
tg32 (decode 32 tokens):              0.492 s → 65.04 tok/s
```

Per-token: **prefill is 2.9× slower**. Expected 5–10× speedup, got 2.9× regression. Opposite direction.

The plumbing was clean and the output was correct, so the kernel underneath was the suspect. But I didn't stop and measure. The `9d46907` commit message included this line:

> "prefill regressed by 2.9× per-token. Likely attention `O(seq²)`. Phase 8.E will rewrite attention."

A speculative diagnosis written into the commit log without a micro-bench. **Second mistake of the bundle.** If I had moved on to "Phase 8.E", I would have spent it rewriting attention for a 1% factor.

## Wrong hypothesis — attention `O(seq²)` — [`e9d42d5`](https://github.com/redchupa/lumen/commit/e9d42d5)

Phase 8.D.4. Isolated `multi_head_attention` in a micro-bench, varied seq length only:

| seq | attention/call |
| ---: | ---: |
| 1 | 1.18 µs |
| 8 | 26.5 µs |
| 16 | 114 µs |
| 32 | 389 µs |
| 64 | 1442 µs |

Clean `O(seq²)`. Double seq, quadruple time. The hypothesis is correct in scaling. It's just dominated by an absurdly small constant.

```
pp32 attention budget:
  24 layers × 389 µs (seq=32) = 9.3 ms
pp32 wall total:               1413 ms
attention share:               0.66%
```

Removing attention entirely would move pp32 from `22.65 → 22.80 tok/s`. **Not attention.**

50-line bench, 5 minutes, one wrong direction killed. Had I trusted the commit-message guess and rolled into 8.E rewriting attention, I would have shipped a working attention rewrite and the same 2.9× regression.

## Real cause — `N>1` codegen is slower than `N=1 × N` — [`cf90aab`](https://github.com/redchupa/lumen/commit/cf90aab)

Phase 8.D.5. If attention is 0.66%, the matmuls own the remaining 95%+. So bench the matmul directly. Strip out transpose, activation quantization, dispatch — call **only the JIT'd weight matmul** in two modes:

- Mode A: one `N=32` call.
- Mode B: thirty-two `N=1` calls back-to-back (decode simulation).

Same shapes, same weights, same threads.

| shape | M | K | `N=32` once | `N=1 × 32` | ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| qkv (Q) | 896 | 896 | 2925 µs | 1156 µs | **2.53× slower** |
| qkv (KV) | 128 | 896 | 417 µs | 160 µs | 2.61× slower |
| wo | 896 | 896 | 2876 µs | 1181 µs | 2.43× slower |
| gate/up | 4864 | 896 | 15566 µs | 6803 µs | 2.29× slower |
| down | 896 | 4864 | 15915 µs | 6454 µs | 2.47× slower |

**Every shape: the `N>1` kernel is 2.3–2.6× slower than calling the `N=1` kernel `N` times.**

The micro-bench ratio (~2.5× mean) plus plumbing/quant overhead lands almost exactly on the observed `2.9×` regression at the model level. Hypothesis confirmed.

### Why

Looking inside the codegen:

- **`N=1` decode kernel** (Phase 7.G): 4-accumulator YMM tuning. Four independent FMA chains so Zen 4's FMA `4-cycle latency / 1-cycle throughput` is fully fed. That tuning gave us ~2× back when 7.G landed.
- **`N>1` prefill kernel** (Phase 8.D.1): generalized to the `N` axis with a straight `for rows { for cols { for k } }`. Zero microarchitectural tuning. Single accumulator. Each FMA waits on the previous one. Zen 4 stalls 4 cycles per FMA.

Same ISA, same hardware, same bytes touched. Dependency-chain length differs by 4×, throughput differs by 2.5×. Plumbing being correct doesn't help if the kernel underneath can't take the batch.

> **The plumbing was fine. The kernel beneath wasn't.**

## The fix — Phase 8.E.1 `N=1` fan-out dispatcher

Now I had a confirmed diagnosis and two choices:

1. Rewrite the `N>1` codegen with the 4-accumulator YMM pattern from 7.G. The Right Thing. Multi-week.
2. Dispatch `N>1` calls as `N` sequential `N=1` calls. A band-aid that should immediately match `N=1 × N` performance and surface whether the plumbing itself adds overhead.

Option 2 first, because the micro-bench already proved `N=1 × N` is faster than what we have, and it's a 10-line dispatcher change that unblocks the rest of the cycle.

```rust
// weight_matmul_jit_batched, simplified:
if n > 1 && dtype == Dtype::Q8_0 {
    // 8.E.1: fan out until N>1 codegen catches up with N=1 × N.
    for row in 0..n {
        weight_matmul_jit(&act[row], weight, &mut out[row], ...);
    }
    return;
}
// else: existing N>1 path (F32, or N==1 short-circuit)
```

Bit-exact vs the previous N>1 path. Same activation layout, same KV cache integration, same numerics.

### Measurement after the fix

Same machine, same model, same 8 threads, 32-token prompt:

| step | pp32 (tok/s) | vs decode |
| --- | ---: | ---: |
| Phase 8.D.3 (`N>1` codegen via row-major transpose) | 22.65 | 0.35× |
| **Phase 8.E.1 (`N=1` fan-out dispatcher)** | **54.15** | **0.83×** |
| `tg32` baseline | 65 | 1.00× |

**`pp32` recovered 2.39×.** Prefill is now `0.83×` of decode per token, not `0.35×`. Plumbing-only overhead (transpose, dispatch, activation quant) accounts for the remaining 17% gap vs decode-by-token.

Commits: [`0e011f1`](https://github.com/redchupa/lumen/commit/0e011f1), [`fe56490`](https://github.com/redchupa/lumen/commit/fe56490).

## Where this leaves us

Prefill is faster than the broken 8.D.3 number, but it's still slower than just decoding `N` tokens. **That's not a real prefill win** — it's the plumbing not actively hurting anymore. The actual weight-reuse upside from batching is still unrealized, because the kernel underneath still runs `N` separate streams over the weights.

Where ggml stands (`pp128 = 739 tok/s` on the same model) is what a properly tuned `N>1` codegen looks like. Reaching that requires lifting Phase 7.G's 4-accumulator pattern into the `N>1` kernel — Phase 8.E proper.

Two-stage hypothesis for the next phase:

- **8.E hypothesis 1**: extending the 4-accumulator YMM pattern to the `N>1` codegen drops the `N=32` kernel from `2.5×` slower to roughly parity with `N=1 × 32`.
- **8.E hypothesis 2**: at parity, the batched kernel exposes weight reuse, and `pp32` moves into the `100+ tok/s` range. The Lumen/ggml gap then gets re-measured under prefill conditions.

If hypothesis 1 fails, Phase 8.F explores a different microarchitectural pattern for batched matmul. If hypothesis 1 holds, hypothesis 2 validates automatically.

## Meta-pattern — don't write speculative diagnoses in commit messages

The biggest take-away from this bundle isn't a perf finding, it's a process one.

The `9d46907` commit message said "Likely attention `O(seq²)`. Phase 8.E will rewrite attention." That one sentence almost dictated the plan for the next phase. A 50-line micro-bench disproved it in 5 minutes — but only because I happened to read my own commit message critically the next day.

> **A commit message that states a cause without a measurement is a hypothesis, not a finding. Treat it like any other hypothesis: verify or kill before acting on it.**

This applies recursively. Code reviews catch wrong code. Nothing catches wrong narration of code. The author is the only person positioned to challenge their own retro guess, and the temptation is to ratify it.

Phases 8.D.4 and 8.D.5 are the cost of that one over-confident commit message. They both happened to land useful results — a clean falsification and a clean diagnosis — but in expectation, that's a phase of work spent verifying a one-line guess.

## v0.5 cycle — measurement-driven decisions so far

| phase | hypothesis | result |
| --- | --- | --- |
| 7.M | Q8 native int dot | net-neutral |
| 7.N | VNNI single-acc | -3.6% |
| 7.O | VNNI 4-acc | -2.7% (revived later in 7.P) |
| 7.R/S/T | AVX-512 ZMM on Zen 4 | -4.5% |
| 8.A | atomic ThreadPool | +9% 1t, +3% 8t (noise) |
| 8.B | L2-fit chunk cap | -3.5% (reverted) |
| 8.C | software prefetch | 8t -49% (reverted) |
| 8.D.3 | prefill batching → speedup | -2.9× pp32 |
| 8.D.3 retro | attention is the cause | 1% only — falsified |
| **8.D.5** | **`N>1` codegen is slower than `N=1 × N`** | **confirmed ✓** |
| **8.E.1** | **`N=1` fan-out recovers plumbing overhead** | **+2.39× pp32 ✓** |

Ten negatives + one positive diagnosis + one partial fix, in one cycle.

No release-worthy perf win on `main` yet — `tg32` is still ~67 tok/s on 8t, identical to v0.4.0 + 8.A. What changed is that for the first time this cycle, the next phase's goal is grounded in a confirmed kernel-level measurement rather than a guess at the system level.

## Decision / next phase

8.D infrastructure stays as-is: `forward_prefill_jit`, chunked dispatch in `generate_greedy_jit`, batched wrapper. Default decode path unchanged. Prefill API exposed with a doc note that `N>1` currently fans out to `N=1` calls.

Phase 8.E proper:

1. Lift the Phase 7.G 4-accumulator YMM pattern into the `N>1` codegen.
2. Re-run the per-shape micro-bench (`N=32` once vs `N=1 × 32`). Target: ratio ≤ 1.0.
3. If hit, re-measure pp32. Expected landing: roughly `100+ tok/s`, weight reuse finally visible.
4. Re-measure Lumen vs ggml under prefill (`pp128` vs `pp128`). That's the number that decides whether v0.5 has a real story.

Tokens remain bit-identical to v0.4.0 throughout.

## Code references

Repo: [github.com/redchupa/lumen](https://github.com/redchupa/lumen).

Key commits this bundle:

- [`f44face`](https://github.com/redchupa/lumen/commit/f44face) — 8.D.1 Q8 `N>1` kernel correctness
- [`061579d`](https://github.com/redchupa/lumen/commit/061579d) — 8.D.1-2 `weight_matmul_jit_batched` wrapper
- [`a8e7601`](https://github.com/redchupa/lumen/commit/a8e7601) — 8.D.2 `forward_layer_prefill_jit`
- [`9d46907`](https://github.com/redchupa/lumen/commit/9d46907) — 8.D.3 chunked dispatch + regression measurement
- [`e9d42d5`](https://github.com/redchupa/lumen/commit/e9d42d5) — 8.D.4 attention micro-bench (hypothesis falsified)
- [`cf90aab`](https://github.com/redchupa/lumen/commit/cf90aab) — 8.D.5 raw kernel micro-bench (cause confirmed)
- [`0e011f1`](https://github.com/redchupa/lumen/commit/0e011f1) — 8.E.1 `N=1` fan-out dispatcher
- [`fe56490`](https://github.com/redchupa/lumen/commit/fe56490) — 8.E.1 micro-bench follow-up + pp32 recovery measurement

`main` perf still tracks v0.4.0 + 8.A on decode. No v0.5 release until `N>1` codegen matches `N=1 × N` and the prefill story holds up against `pp128` numbers.
