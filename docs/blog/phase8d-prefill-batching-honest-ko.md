# Prefill batching이 답이 아니었던 이유 — Phase 8.D 후기

> Lumen 빌드로그 #15. v0.5 세 번째 묶음. 가설은 plumbing이었지만 진짜 원인은
> 한 층 아래 kernel codegen이었음. 측정-주도 의사결정 10번째 가설 부정.

## TL;DR

- v0.5 standing (Phase 8.A 후): Lumen 8t 67.4 tok/s, ggml 8t 87.9 tok/s — 격차 **1.304×**.
- 가설: ggml은 prefill (pp128 = 739 tok/s) 로 weight reuse를 살리는데 우리는 prefill 미지원 (decode-by-token = 65 tok/s). prefill batching 붙이면 5~10× 가속 가능.
- Phase 8.D.1 ~ 8.D.3: Q8 N>1 kernel 정답성 통과, `weight_matmul_jit_batched` wrapper, `forward_layer_prefill_jit`, `Model::forward_prefill_jit` + `generate_greedy_jit` chunked dispatch까지 plumbing 완성.
- 측정 (Zen 4, Qwen2.5-0.5B-Q8, 8t): **pp32 = 22.65 tok/s** vs tg32 baseline 65 tok/s. token당 **2.9× 더 느림 (회귀)**.
- 가설 1 (attention O(seq²) 가 원인) → 50줄 micro-bench로 부정. attention 24 layers × 389µs = 9.3ms = pp32 전체의 **0.66%**.
- 가설 2 (N>1 codegen 비효율) → raw kernel micro-bench로 확정. **N=32 kernel이 N=1 kernel × 32보다 2.3~2.6× 느림**. Phase 7.G 4-acc YMM 튜닝이 N=1 path에만 들어가 있었음.
- 결정: 8.D 인프라(plumbing) 유지, default 동작 변경 없음. 진정한 prefill 가속은 **Phase 8.E에서 N>1 codegen에 7.G 4-acc 패턴 확장**.

토큰은 모든 commit에서 v0.4.0과 bit-identical.

## 출발점

Phase 8.A에서 ThreadPool 재설계로 격차 1.376× → 1.304×. 하지만 8t는 noise 영역 win 이었고, 진짜 진단은 "memory bw 활용 효율 차이"였다. 다음 질문이 자연스럽게 따라온다.

> **bw bound라면, 같은 weight를 더 많이 재사용하는 워크로드에선 격차가 어떻게 변하나?**

decode는 token마다 weight 670 MB 전체를 stream한다. 그래서 매 token DRAM full read = 메모리 wall. 반면 prefill은 N개의 prompt token을 한 batch로 묶어서 같은 weight 한 번 읽고 N번 reuse. weight bytes / FLOP 비율이 N분의 1로 떨어진다.

ggml standing (같은 모델, 8t):
- tg128 (decode): 87.9 tok/s
- pp128 (prefill): **739 tok/s** — 8.4× 빠름

Lumen v0.4.0:
- tg32: 65 tok/s
- pp32: **decode-by-token = 65 tok/s** (prefill 미지원)

ggml prefill이 우리 decode 대비 ~11× 빠른 셈. prefill을 붙이면 잠재 5~10× 가속. v0.5 묶음의 가장 분명한 win 후보.

## Phase 8.D 가설

> **prefill batching이 답일 것이다.**

논리는 단순했다:
1. matmul kernel은 이미 있다 (decode path).
2. 같은 kernel에 batched activation을 넣으면 N>1 row 처리 = throughput 가속.
3. plumbing만 잘 깔면 5~10× 자동으로 따라온다.

이 가설을 점검하지 않고 그대로 plumbing 시작한 게 이번 phase의 첫 실수다.

## 진행 — plumbing 통과

### Phase 8.D.1 — Q8 N>1 kernel 정답성

JIT codegen이 N=1 (decode) path만 가정한 곳이 다수. N>1을 받게 일반화.

검증: 6개 형상 (qkv Q/KV, wo, gate, up, down) × N ∈ {1, 8, 32}. naive Rust reference와 bit-exact. 추가로 3개 통합 테스트 (single-token forward, batched-token forward, layer 단위 비교).

커밋 [f44face](https://github.com/redchupa/lumen).

### Phase 8.D.1-2 — `weight_matmul_jit_batched` wrapper

기존 `weight_matmul_jit(F32, weight, …)` 는 row=1 가정. `weight_matmul_jit_batched(F32_2D[N×K], weight, …)` 추가. dtype별 dispatch (F32/Q8) + rows=1 short-circuit (decode path는 절대 안 느려지게).

커밋 [061579d](https://github.com/redchupa/lumen).

### Phase 8.D.2 — `forward_layer_prefill_jit`

기존 `forward_layer_jit(token[K]) → token[K]` 의 prefill counterpart. 입력 `tokens[N×K]` → 출력 `tokens[N×K]`. attention만 새로 짜야 했다 (Q@K^T가 [N×N] 대신 [N×seq_total] 이라 KV cache 통합 필요). 나머지는 다 N차원으로 통과.

정답성: layer 출력이 N번의 decode-by-token 결과와 bit-exact.

커밋 [a8e7601](https://github.com/redchupa/lumen).

### Phase 8.D.3 — Model::forward_prefill_jit + chunked dispatch

`generate_greedy_jit` 가 prompt를 N=32 chunk로 나눠 `forward_prefill_jit` 호출, 마지막 chunk 후 decode 진입. chunked dispatch 패턴은 ggml과 동일.

커밋 [9d46907](https://github.com/redchupa/lumen).

여기까지 plumbing 완성. 정답성 모두 통과. 이제 측정.

## 측정 — 회귀

Zen 4, Qwen2.5-0.5B-Q8_0, 8t, 32-token prompt:

```
pp32 (prefill 32 tokens + decode 1):  1.413 초 → 22.65 tok/s
tg32 (decode 32 tokens):              0.492 초 → 65.04 tok/s
```

**prefill이 token당 2.9× 더 느림.**

기대했던 5~10× 가속 정반대 방향으로 갔다. plumbing은 정답성 통과인데 perf는 회귀. 가설 부정.

여기서 멈췄으면 안 됐다. commit [9d46907] message에 "attention 부분이 O(seq²) 라 prefill 느림 — Phase 8.E에서 손볼 것" 이라고 추측 적어놨다. 측정 안 하고 적어버린 추측. 이게 이번 묶음의 두 번째 실수.

## 가설 1 — attention O(seq²) (부정)

Phase 8.D.4. `multi_head_attention` 만 분리해서 micro-bench. seq 길이만 바꿔가며:

| seq | attention/call |
|---:|---:|
| 1 | 1.18 µs |
| 8 | 26.5 µs |
| 16 | 114 µs |
| 32 | 389 µs |
| 64 | 1442 µs |

깔끔한 O(seq²). seq 2× → 시간 4×. 가설 자체는 옳다.

하지만 절대값이 작다:
- pp32 = 32 token × 24 layers × 1 attention call/layer = 768 attention call (decode pass 제외)
- 24 layers × 389 µs (seq=32) = **9.3 ms**
- 전체 pp32 = 1413 ms
- attention / total = **0.66%**

attention 100% 제거해도 prefill은 22.65 → 22.80 tok/s. **attention 아님.**

50줄 micro-bench로 5분 만에 부정 가능했던 추측. commit message에 추측만 남겨놓고 다음 phase로 넘어갔으면, Phase 8.E를 attention 재작성에 통째로 태웠을 것. 측정 한 번에 잘못된 방향 한 phase 절약.

커밋 [e9d42d5](https://github.com/redchupa/lumen).

## 가설 2 — N>1 kernel codegen 비효율 (확정)

Phase 8.D.5. attention이 아니면 matmul. matmul이 prefill 시간의 95%대. 그 중 N=32 kernel이 어떻게 동작하는지를 raw로 본다.

micro-bench: transpose / activation quantize / dispatch overhead 전부 빼고 **순수 weight matmul JIT 함수만 호출**. N=32 짜리 1번 vs N=1 짜리 32번 (decode 시뮬레이션).

| Shape | M | K | N=32 once | N=1 × 32 | 비율 |
|---|---:|---:|---:|---:|---:|
| qkv (Q) | 896 | 896 | 2925 µs | 1156 µs | **2.53× 느림** |
| qkv (KV) | 128 | 896 | 417 µs | 160 µs | 2.61× 느림 |
| wo | 896 | 896 | 2876 µs | 1181 µs | 2.43× 느림 |
| gate/up | 4864 | 896 | 15566 µs | 6803 µs | 2.29× 느림 |
| down | 896 | 4864 | 15915 µs | 6454 µs | 2.47× 느림 |

**모든 shape에서 N=32 kernel이 N=1 kernel × 32 보다 2.3~2.6× 느림.**

이게 진짜 원인이다. plumbing이 batch를 잘 모아서 한 번에 넘겨도, 그 한 번이 32번 따로 도는 것보다 느리면 의미가 없다. pp32 결과 (2.9× 느림) 와 거의 정확히 일치 — micro-bench 비율 2.5× 평균에 plumbing/quant overhead 더하면 2.9×.

커밋 [cf90aab](https://github.com/redchupa/lumen).

## 진단

왜 N>1 kernel이 N=1 × 32보다 느린가? codegen 안을 보면 단서가 있다.

- **N=1 decode kernel** (Phase 7.G): 4-accumulator YMM 튜닝. FMA dependency chain을 4개로 분리해서 hide. Zen 4 FMA latency 4 cycle / throughput 1 cycle을 fully feed. 7.G 도입 시 ~2× 가속이 여기서 나옴.
- **N>1 prefill kernel**: Phase 8.D.1에서 N차원 일반화. inner-loop는 단순 nested loop (rows × cols × k). microarchitectural 튜닝 0. 단일 accumulator, FMA 결과를 바로 다음 FMA의 입력으로 쓰는 직선 chain. Zen 4가 4 cycle stall.

같은 ISA, 같은 hardware, 같은 weight bytes를 읽지만 dependency chain 길이가 달라서 throughput이 2.5× 차이.

> **Phase 8.D plumbing은 정답. 그 아래 N>1 codegen이 안 따라줬을 뿐.**

가설 자체는 옳았다 (batching → weight reuse → 가속). 다만 batching하려면 batched kernel이 decode kernel만큼 튜닝돼야 한다. 그 가정을 점검하지 않고 plumbing부터 깔았다.

## 메타 패턴 — commit message에 추측 적지 말 것

이번 묶음 가장 큰 학습은 perf 진단이 아니라 절차에 관한 것.

Phase 8.D.3 commit message:
> "prefill regressed by 2.9× per-token. Likely attention O(seq²). Phase 8.E will rewrite attention."

이 한 문장이 다음 phase를 한 갈래 잘못 보낼 뻔 했다. 50줄 micro-bench로 1%만 영향이라고 확정 가능했던 케이스인데, 측정 없이 추측만 적어놓고 8.E 계획까지 세웠다.

> **추측만 적힌 commit message는 다음 phase 시작 전 측정으로 검증해야 한다.**

8.D.4가 그 검증이고, 8.D.5가 진짜 원인 진단. 이 두 phase는 8.D.3 commit message가 "attention 의심" 이 아니라 "원인 미상, 다음에 측정" 이었으면 같은 결론에 더 빨리 도착했을 것.

추측은 가설이고, 가설은 측정으로 부정/확정해야 한다. 이걸 commit message에 단정형으로 적으면 미래의 자신이 검증 단계를 건너뛴다. 코드 진단도 그렇지만, 자기 자신의 추측에 대해서도 측정-주도 의사결정을 적용해야 한다.

## v0.5 cycle 패턴 누적

| Phase | 가설 | 측정 결과 |
|---|---|---|
| 7.M | Q8 native int dot | net-neutral |
| 7.N | VNNI single-acc | -3.6% |
| 7.O | VNNI 4-acc | -2.7% (7.P 부활) |
| 7.R/S/T | AVX-512 ZMM | -4.5% |
| 8.A | atomic ThreadPool | +9% 1t, +3% 8t (noise) |
| 8.B | L2-fit chunk cap | -3.5% (revert) |
| 8.C | software prefetch | 8t -49% (revert) |
| 8.D.3 | prefill batching → 가속 | -2.9× pp32 |
| 8.D.3 retro | attention 원인 | 1%만 (부정) |
| **8.D.5** | **N>1 codegen 원인** | **확정 ✓** |

10번 가설 부정 + 1번 명확 진단. 매번 인프라는 유지. 코드베이스는 매번 깔끔해지고, 격차 원인에 대한 이해도 매번 한 칸씩 더 좁혀짐.

명확한 perf win은 여전히 0개. 그런데 이번 묶음 끝에서 처음으로 "다음 phase에서 뭘 해야 하는지" 가 가설이 아니라 측정으로 확정됨. 그게 이번의 진짜 산출.

## 결정 / 다음 phase

8.D 인프라 (plumbing, `forward_prefill_jit`, chunked dispatch) 는 모두 유지. 단:
- default 동작 변경 없음. 짧은 prompt는 decode-by-token fallback (회귀 방지).
- prefill API는 expose하되 perf 회귀를 알리는 doc 추가.

진정한 prefill 가속은 **Phase 8.E**:
1. N>1 codegen에 Phase 7.G의 4-accumulator YMM 패턴 확장.
2. 같은 shape별 micro-bench로 2.5× → 1.0× (또는 그 이하) 까지 좁혔는지 검증.
3. 거기 도달하면 pp32 가 22.65 → 60+ tok/s 로 정상화. ggml pp128 740 tok/s 대비 격차는 그때 다시 측정.

가설은 두 단계로 분리됨:
- **8.E 가설 1**: N=1 4-acc 패턴을 N>1로 확장하면 N=1 kernel × N 수준에 도달.
- **8.E 가설 2**: 거기 도달하면 weight reuse 효과가 살아나서 prefill 본래 5~10× 가속이 실현됨.

가설 1이 부정되면 8.F는 batched matmul용 다른 microarchitectural 패턴 탐색. 가설 1이 확정되면 가설 2 자동 검증.

## 코드 참조

코드: [github.com/redchupa/lumen](https://github.com/redchupa/lumen).

이번 묶음 핵심 커밋:
- [f44face](https://github.com/redchupa/lumen/commit/f44face) — Phase 8.D.1 Q8 N>1 kernel 정답성
- [061579d](https://github.com/redchupa/lumen/commit/061579d) — Phase 8.D.1-2 `weight_matmul_jit_batched` wrapper
- [a8e7601](https://github.com/redchupa/lumen/commit/a8e7601) — Phase 8.D.2 `forward_layer_prefill_jit`
- [9d46907](https://github.com/redchupa/lumen/commit/9d46907) — Phase 8.D.3 chunked dispatch + 회귀 측정
- [e9d42d5](https://github.com/redchupa/lumen/commit/e9d42d5) — Phase 8.D.4 attention micro-bench (가설 1 부정)
- [cf90aab](https://github.com/redchupa/lumen/commit/cf90aab) — Phase 8.D.5 raw kernel micro-bench (가설 2 확정)

main branch perf는 v0.4.0 + 8.A 동일 (8t ~67 tok/s). v0.5 release 여전히 없음. Phase 8.E에서 N>1 codegen 4-acc 확장 후 perf 회복되면 그때 release 여부 결정.
