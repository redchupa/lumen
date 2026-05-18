# 1.4× 격차의 진짜 정체 — Phase 7.U + 8.A 후기

> Lumen 빌드로그 #14. v0.5 두 번째 묶음. 또 한 번 가설이 측정에 부정당함.
> 이번엔 lane width가 아니라 thread scaling이 의심받았고, 결국 진짜 범인은
> **메모리 대역폭 활용 효율**이라는 결론에 도달.

## TL;DR

- v0.4.0 standing: Lumen 8t 65.6 tok/s, ggml 8t 90.3 tok/s — **1.376× 격차**.
- Phase 7.U: 격차 원인 gap analysis. **1t에서 Lumen이 ggml보다 +3.6% 빠름**. 격차는 오직 thread scaling에서 발생.
- Phase 8.A: ThreadPool 재설계 (mpsc + Mutex → atomic counter + generation wake). 1t/2t 큰 win, **8t에선 +2.9% (noise 영역)**. dispatch overhead가 8t critical path 아니었음.
- Phase 8.B: chunk_rows L2-fit cap 가설 → **-3.5% 회귀, revert**.
- 반복 호출 벤치: cold→warm 차이 +2.8%만. Qwen2.5-0.5B 670MB weight >> Zen 4 L3 32MB. **매 token마다 DRAM 전체 stream**.
- 진단: 1.3× 격차 = **메모리 bw 활용 효율 차이** (Lumen 44 GB/s vs ggml 59 GB/s, DDR5-5200 80GB/s 대비 55% vs 74%).

토큰은 모든 commit에서 v0.4.0과 bit-identical.

## 출발점

이전 phase7t 블로그 결론: "ZMM 폭은 Zen 4에서 안 통한다. 격차 원인이 lane width가 아니다." 인프라만 남기고 default-off. v0.5 release 없음. 그래서 v0.5 두 번째 묶음의 첫 질문은 단순했다.

> **그럼 1.4× 격차의 진짜 원인은 뭐냐?**

v0.4.0 standing:
- Lumen 8t: 65.6 tok/s
- ggml 8t: 90.3 tok/s
- 격차 1.376×

가설 후보가 너무 많다. 매트멀 커널 효율? thread overhead? memory bw? cache miss? prefetch? 측정 없이는 어디 손댈지 모름. Phase 7.U는 그래서 측정 phase로 정의됨.

## Phase 7.U — Gap analysis

### Step breakdown

8t warm 32-token generation, 509ms total:

| Step | Time | % |
|---|---:|---:|
| gate_up_matmul | 201 ms | 39.7% |
| down_matmul | 109 ms | 21.3% |
| lm_head_matmul | 94 ms | 18.4% |
| qkv_matmul | 40 ms | 7.8% |
| wo_matmul | 32 ms | 6.3% |
| **matmul 합계** | **476 ms** | **93.5%** |
| attention / rope / norm | 33 ms | 6.5% |

→ 95%가 매트멀. attention 최적화는 무의미. 매트멀 외 다른 곳 손대도 ceiling 5%. **모든 답은 매트멀 안에 있음.**

### Thread scaling (Lumen vs ggml)

이게 진짜 충격이었다.

| Threads | Lumen v0.4 | ggml | 누가 빠른가 |
|---|---:|---:|---|
| 1 | 41.85 | 40.40 | **Lumen +3.6%** |
| 2 | 52.97 | 62.55 | ggml +18% |
| 4 | 63.72 | 86.33 | ggml +35% |
| 8 | 65.47 | 87.86 | ggml +34% |

**1t에서 Lumen이 ggml보다 빠름.** single-thread 매트멀 커널 자체는 우리가 안 진다. 7.P shape-aware kernel selection 효과로 보임.

격차는 thread 추가하면서 **벌어진다**.

- Lumen 1→8t scaling: 1.56× (효율 19.5%)
- ggml 1→8t scaling: 2.17× (효율 27.2%)

두 쪽 다 8t scaling 효율이 처참하다 (ggml조차 27%). 하지만 Lumen이 더 나쁨. 그래서 가설 1번:

> **격차 = thread dispatch overhead.** 8t에서 7개의 worker thread가 매 token (4개 매트멀 step × N layers × 32 tokens) 마다 wake/run/sync. 우리 ThreadPool이 ggml보다 무겁다.

### Step별 1t→4t 가속

| Step | 가속비 | 효율 |
|---|---:|---:|
| gate_up_matmul | 1.42× | 36% (최악) |
| down_matmul | 2.24× | 56% |
| lm_head_matmul | 1.97× | 49% |
| qkv_matmul | 1.14× | 29% |
| wo_matmul | — | — |

gate_up_matmul 가속 효율 36%가 의심스럽다. gate_up은 전체 39.7% 차지하는 가장 큰 step. 여기 효율 낮으면 전체가 낮아짐. 하지만 chunk_rows = M/threads 분할만 잘 되어 있으면 효율이 이렇게 낮을 수 없다. → 가설 2번:

> **chunk가 L2 cache에 안 들어감.** Zen 4 L2 = 1MB/core. gate_up chunk size 계산 → ~1.16MB. L2 못 fit. miss → DRAM round-trip → bw bound.

가설 1 (dispatch overhead) 와 가설 2 (L2 fit) 가 동시에 후보. Phase 8.A는 1번, 8.B는 2번을 친다.

## Phase 8.A — ThreadPool 재설계

기존 ThreadPool (Phase 7.L 작성):
- single `mpsc::channel<Box<dyn FnOnce>>`
- worker 쪽 `Mutex<Receiver>`로 task pull
- task per dispatch = `Box::new(...)` heap allocation
- tg32 8t = **3,840 dispatches × (1 box alloc + 1 mutex lock + 1 channel recv)**

mutex contention과 box allocation이 hot path에 있음. 단순 fork-join 패턴인데 mpsc channel은 over-engineered.

새 설계:
- `AtomicUsize next_task` counter
- worker는 `fetch_add(1)`로 자신의 task index 가져옴
- task closure는 **enqueue 시 한 번만 저장**, worker가 index로 접근
- generation-based wake (Condvar): 메인이 generation++, 모든 worker wake
- 비-task allocation, 비-task mutex (wake 시점 condvar 1번만)

이전 ThreadPool API 호환 (`parallel_for(n_tasks, f)`). 호출부 변경 0.

측정 (Zen 4, 8 tg32 runs each, mean):

| Threads | v0.4 | 8.A | 변화 |
|---|---:|---:|---:|
| 1 | 41.85 | 45.64 | **+9.1%** |
| 2 | 52.97 | 60.78 | **+14.7%** |
| 4 | 63.72 | 66.45 | +4.3% |
| 8 | 65.47 | 67.38 | **+2.9%** |

1t/2t는 시원하게 win. 1t에서 +9.1% 나오는 건 main thread도 ThreadPool로 dispatch하는 patten 때문 (main이 1개 chunk 잡고 worker N-1개 wake). dispatch overhead 줄어드니 1t에서도 효과.

ggml 격차: 1.376× → 1.304×.

**하지만 8t에서 +2.9%는 noise 영역.** 같은 8t에서 measurement run-to-run 변동이 1~2%. +2.9%면 실질 1.5% 정도일 수도. 이 한 사실이 시사하는 바:

> **dispatch overhead는 8t critical path가 아니었다.**

가설 1번 부분 부정. 1t/2t에서는 분명 dispatch overhead가 문제였지만, 8t에서는 다른 게 더 큰 병목이라는 것. 직관 (8t = 더 많은 dispatch = dispatch overhead가 더 큼) 이 또 부정됨. 8t에서는 각 dispatch가 한 chunk 큰 작업 잡고 가니, 절대적 dispatch 횟수는 같아도 dispatch cost / total cost 비율이 작아짐. 일리 있다.

ThreadPool 자체는 합당한 개선이라 채택 (1t/2t bench는 분명 win, 코드도 더 단순). 하지만 1.4× 격차의 본체는 아직 손도 못 댔음.

커밋 [3374278](https://github.com/redchupa/lumen).

## Phase 8.B — chunk_rows L2-fit cap (회귀 → revert)

가설 2번을 친다.

Zen 4 L2 = 1 MB/core. gate_up은 d_out = 9728 (gate + up concat), K = 896:
- chunk_rows = 9728 / 8 = 1216
- chunk당 weight: 1216 × 896 / 32 (Q8 block) × 34 byte ≈ **1.16 MB > 1 MB L2**

이론상 fit 안 됨. 더 작은 chunk로 cap 하면 L2-resident 가능. lm_head도 비슷 — 18.1 MB chunk가 L2 못 들어감.

`L2_BUDGET = 512 KB`로 cap. 적용 후:
- gate_up chunk: 1216 → **1102 rows** (둘레 마진 + activation 자리 확보)
- lm_head chunk: 18992 → **1102 rows**
- down/qkv/wo는 이미 fit, cap 영향 없음

측정:
```
v0.4 baseline 1t:  41.85 tok/s
8.B 1t:             40.10 tok/s   (-4.2%)
v0.4 baseline 8t:  65.47 tok/s
8.B 8t:             63.18 tok/s   (-3.5%)
```

전 thread에서 회귀. 1t에서도 -4.2%. **1t는 L2 fit 가설과 무관**해야 하는데 (single core가 L2 1MB 다 쓰니까) 회귀가 났다는 건, 회귀 원인이 **chunk_rows JIT 컴파일 커널 자체의 효율 차이**라는 것.

7.P의 shape-aware kernel selection은 chunk_rows를 보고 다른 커널을 합성한다. chunk_rows 1102 vs 608은 다른 kernel binary로 끝남. 그 합성된 커널의 inner-loop 효율이 단순히 다르다. L2 fit과 무관.

가설 2번 부정. **L2 fit 자체는 우리가 통제하는 차원이 아님 또는 효과가 거의 없음.** revert.

## 반복 호출 벤치 — memory-bw bound 확정

여기서 한 단계 물러서서 "근본적으로 우리 컴퓨터가 뭘 하고 있는지"를 묻기로.

같은 model + 같은 prompt + 같은 8t로 연속 5회 generate:

```
run 1: 64.17 tok/s  (cold)
run 2: 65.31 tok/s
run 3: 65.76 tok/s
run 4: 66.09 tok/s
run 5: 65.97 tok/s
```

cold → warm 차이 **+2.8%만**. 이게 뜻하는 바가 크다.

Qwen2.5-0.5B Q8 weight = **670 MB**. Zen 4 L3 cache = **32 MB**. 비율 21:1. weight의 ~5%만 L3에 들어감.

→ **매 token 생성마다 weight 670 MB를 DRAM에서 다시 stream해야 함.** L3 hit rate가 워낙 낮아 cold/warm 구분도 거의 없음.

8t로 32 token 생성에 485 ms (v0.4 baseline 기준):
- 32 token × 670 MB weight stream = **21.4 GB** read
- 485 ms 안에 → **44 GB/s** sustained bandwidth

DDR5-5200 dual-channel 이론값 = **83.2 GB/s**. 우리는 **53%** 사용.

ggml 같은 워크로드:
- 32 token / 364 ms → 21.4 GB / 0.364s = **59 GB/s** sustained
- **71%** 사용

격차 비율: 59 / 44 = **1.34×**. 우리가 측정해 온 격차 1.30~1.38×와 **거의 일치**.

> **결론: 1.3× 격차의 진짜 정체는 메모리 bw 활용 효율 차이.**

ggml은 prefetch (`vprefetcht0`, `prefetchnta`), streaming load/store, instruction scheduling 등으로 DRAM latency를 우리보다 잘 hide한다. 같은 매트멀 일 (=같은 byte를 같은 횟수 읽어야 함)이지만 ggml이 stall 시간을 더 줄임. 우리 매트멀 커널은 단순히 "vmovups + vfma" 시퀀스. memory side에서 우리가 못한 게 많다.

## 측정-주도 의사결정의 7번째 사례

| Phase | 가설 | 측정 | 결정 |
|---|---|---|---|
| 7.M | Q8 native int dot | net-neutral | infra |
| 7.N | VNNI single-acc | -3.6% | infra |
| 7.O | VNNI 4-acc | -2.7% | infra → 7.P 부활 |
| 7.R/S/T | AVX-512 ZMM lane width | -4.5% | infra |
| 7.U | gap = thread overhead | 부분 사실 | 다음 phase로 |
| 8.A | ThreadPool 재설계 | +2.9% (8t noise) | merge (1t/2t 명확 win) |
| **8.B** | **chunk L2 fit cap** | **-3.5%** | **revert** |

7개 phase 중 **명확한 perf win 0개, noise win 1개 (8.A 8t), 회귀 5개, 부정 1개**. 그런데 코드베이스는 매번 깔끔해지고, 격차 원인에 대한 이해는 단조 증가한다.

이번 묶음의 학습:
1. 1t에서 우리가 ggml보다 빠르다. **커널 자체는 안 진다.**
2. 격차는 thread scaling이 아니라 **memory bw 활용**에 있다.
3. dispatch overhead는 작은 thread 수에서만 의미 있는 비용이다. 8t에선 noise.
4. L2 fit 이론은 chunk_rows를 직접 통제할 수 없는 구조 (shape-aware kernel synthesis가 chunk_rows를 다양화) 에서는 손대기 어렵다.
5. **L3 << weight**라는 사실이 모든 cache 최적화의 근거를 흔든다. 매 token DRAM full stream이라면 cache 정책의 효과는 마진뿐.

직관: "thread 많을수록 dispatch overhead가 critical". 사실: **memory wall이 critical**.

## 다음 갈래

memory bw가 진짜 병목이라면, 직접 해야 할 일들:

1. **소프트웨어 prefetch.** 매트멀 inner loop에 `prefetcht0`, `prefetchnta` 삽입. ggml이 하는 일.
2. **Streaming store / non-temporal load.** output write에 `vmovntps`. weight read에 NT hint.
3. **Loop unrolling + interleaving** 으로 dependency chain 늘려 DRAM stall hide.
4. **모델 크기를 바꿔 측정.** Qwen2.5-1.5B (Q8 ~1.5GB) / -3B (~3GB) 에서 같은 격차가 나는가? bw bound면 격차가 비슷하게 유지될 것.
5. **다른 quantization (Q4_K_M ~ 350MB).** weight가 L3에 더 가까워지면 격차 줄어야 정상.

5번이 다음 phase 후보 1순위. weight를 줄여서 memory wall 영역에서 벗어났을 때 격차가 어떻게 변하는지가, 진단을 한 번 더 검증한다.

코드: [github.com/redchupa/lumen](https://github.com/redchupa/lumen).
이번 묶음 핵심 커밋: [3374278](https://github.com/redchupa/lumen/commit/3374278) (Phase 8.A ThreadPool), [3bad81f](https://github.com/redchupa/lumen/commit/3bad81f) (반복 호출 bench).

main branch perf는 v0.4.0 대비 1t/2t 명확 win, 8t는 +2.9% (noise). v0.5 release 여전히 없음. 다음 phase에서 memory side 손대 본 후 release 여부 결정.
