# v0.3.0 — 측정이 시키는 대로 갔더니 4× 더 빨라졌다

> Lumen 빌드로그 #10. Phase 7.E.0 ~ 7.J 합본.

## 결론

```
v0.2.0 tg32 (single thread):  17.97 tok/s   [어제]
v0.3.0 tg32 (single thread):  ~31 tok/s     [+72%]
v0.3.0 tg32 (8 threads):      ~56 tok/s     [+212% vs v0.2.0]

vs llama.cpp:
  Lumen 8t (56)  vs  ggml 1t (41.32):  Lumen 1.36× 빠름
  Lumen 8t (56)  vs  ggml 8t (90.90):  ggml  1.62× 빠름
```

토큰은 여전히 naive ↔ JIT bit-identical. `qwen_generate_jit_matches_naive_and_speed`
가 모든 커밋에서 그대로 통과.

## 무엇을 했나 — 한 번에

v0.2.0 직후 "다음 갈래는 attention vs multi-thread vs 매트멀 미세조정"
이라고 적었습니다. 그때 답은 "감으로 정하지 말고 측정". 여기서 출발해서
다섯 단계에 걸쳐 4× 빨라졌습니다.

| Phase | 무엇을 | 효과 |
|---|---|---|
| 7.E.0 | `StepTimer` 추가, decode 한 토큰을 17가지 bucket으로 분해 측정 | (계측 자체) |
| 7.G | Q8 N=1 매트멀 1-누산기 → 4-누산기 (7.C와 같은 trick, Q8에 적용) | matmul 1.9× |
| 7.H | SiLU + elementwise mul AVX2 + 벡터화된 expf | SiLU 87×, 전체 ~5% |
| 7.I | RoPE에서 `inv_freq`와 `sin_cos` 호이스팅 (per-head 중복 제거) | RoPE 3.7× |
| 7.J | rayon으로 큰 Q8 매트멀의 M 차원 멀티스레딩 | decode 1.81× |

전 단계가 데이터 → 결정 순서로 갔습니다. 측정 없었으면 7.E (flash
attention)에 들어갔을 텐데, 측정해보니 attention compute가 **0.9%**.
flash attention을 100배 빠르게 만들어도 0.9% 절감. 안 들어가길 잘함.

## 측정이 보여준 것 (Phase 7.E.0)

v0.2.0의 decode 시간 분포 (32 토큰):

```
layer/gate_up_matmul  692ms  36.1%
      lm_head_matmul  446ms  23.2%
   layer/down_matmul  414ms  21.6%
      layer/silu_mul  155ms   8.1%
    layer/qkv_matmul   83ms   4.3%
     layer/wo_matmul   65ms   3.4%
          layer/rope   43ms   2.2%
     layer/attention   17ms   0.9%    ← flash attention의 시작점
       (rest)              < 1%
```

**88.6%가 매트멀, 8.1%가 SiLU, 2.2%가 RoPE, 0.9%가 attention.**

원래 계획 7.E (flash)와 7.F (multi-thread prefill)는 측정 후 둘 다
폐기. 진짜 다음 단계는 (1) 매트멀을 더 짜내고, (2) SiLU/RoPE의
pure-Rust 비효율 잡고, (3) 큰 매트멀을 멀티스레드 — 이 순서.

## 7.G — 같은 trick, 다른 커널

v0.2.0의 Q8 N=1 매트멀은 4개 FMA를 같은 `ymm0`에 직렬로 쏟아붓고
있었습니다 (4 cycle latency × 4 FMA = 블록당 16 cycle 최소). Phase
7.C에서 fp32 매트멀에 했던 4-누산기 분산을 Q8 N=1에도 그대로:

```asm
; old: 1 accumulator
vfmadd231ps ymm0, ymm_w0, [act+0]
vfmadd231ps ymm0, ymm_w1, [act+32]    ; waits on ymm0 from prev
vfmadd231ps ymm0, ymm_w2, [act+64]    ; waits again
vfmadd231ps ymm0, ymm_w3, [act+96]    ; waits again

; new: 4 independent chains
vfmadd231ps ymm0, ymm_w, [act+0]      ; -> different accumulator each
vfmadd231ps ymm1, ymm_w, [act+32]
vfmadd231ps ymm2, ymm_w, [act+64]
vfmadd231ps ymm3, ymm_w, [act+96]
; final reduction: vaddps + vhaddps tail (once per row)
```

모든 Q8 매트멀이 **1.87-1.95× 빨라짐**. 메모리 대역폭에 도달했다는
신호 (이론 4× peak FMA throughput을 다 못 가져옴).

### Win64 ABI 디테일 (한 번 데인 거)

처음 짠 커널은 ymm6를 weight 스크래치로 썼습니다. 단위 테스트 통과,
Qwen e2e 실행하니까 logits 전부 NaN → argmax = 0 (모든 토큰 0번 ID).

원인: **Win64 ABI는 XMM6-XMM15가 callee-saved**. SysV(Linux/macOS)는
모두 caller-saved. 단위 테스트는 호출 전후 상태가 없으니까 못 잡았고,
실제 forward에서는 RMSNorm 같은 다른 코드가 xmm6에 들고 있던 fp32
값이 매트멀 호출 후 깨져서 NaN 생산. 수정은 ymm6 → ymm4 한 줄.

기록 가치 있는 함정: **단위 테스트는 ABI 보존 의무를 검증 안 함.
호출자가 같이 있는 e2e 테스트가 필수.**

## 7.H — pure Rust ops를 std::arch로

SiLU = `x * sigmoid(x) = x / (1 + exp(-x))`. 153ms / 32토큰 = 5ms/tok,
순수 Rust 스칼라.

AVX2 + FMA로 8 lane 동시 처리. 핵심은 `expf` 벡터화:

```rust
fn exp_ps_avx2_fma(x: __m256) -> __m256 {
    // 1. clamp to [-87, 87]
    // 2. n = round(x * log2(e))
    // 3. r = x - n * ln2          ; reduced argument via FMA
    // 4. exp(r) = degree-5 polynomial in r (Horner)
    // 5. 2^n = pack integer exponent bits, reinterpret as f32
    // 6. exp(x) = exp(r) * 2^n
}
```

외부 의존성 없이 std::arch::x86_64만 사용. polynomial 정확도 ~1e-7
relative — argmax-preserving sigmoid에는 충분.

**SiLU: 153ms → 1.76ms (87× 빠름).** mul도 같은 패턴으로 벡터화 (8-wide
mulps). 전체 decode에서 SiLU+mul 비중은 14% → 0.4%.

## 7.I — RoPE의 무료 트릭

RoPE는 token별로 `q`와 `k`에 회전 적용:
- inv_freq[i] = base^(-2i/head_dim) — `i, head_dim, base`에만 의존
- (sin, cos)[t][i] = (pos*inv_freq[i]).sin_cos() — `pos, i`에만 의존

기존 코드는 가장 안쪽 (t, h, i) 루프에서 매번 `powf` + `sin_cos` 호출.
Qwen2 디코드 (head_dim=64, n_heads=14, 1 position) → 14 × 32 = 448
중복 호출 per layer × 24 layer = 10,752 powf 호출 per token.

호이스팅해서 호출 횟수:
- `powf`: head_dim/2 = 32 회 per call (was n_heads × half = 448)
- `sin_cos`: half = 32 회 per call (was 448)

**RoPE: 43ms → 12ms (3.7×).** 코드는 그냥 두 개의 Vec 사전 계산.

## 7.J — rayon으로 큰 매트멀 멀티스레딩

7.G/H/I 끝나니 매트멀이 **95.9%**. 단일 스레드 한계 도달.
다음은 멀티스레드.

선택지:
1. std::thread + 매 호출마다 spawn — spawn cost 50μs × 8 thread × 169
   matmul calls/token = 67ms/token overhead. 망함.
2. 자체 thread pool — ~150 LOC, 까다로움.
3. rayon — 검증된 라이브러리, CLAUDE.md에서 허용된 deps 중 하나.

3번. rayon 추가, decode 매트멀에서 M (output row) 차원으로 분할.
각 thread가 같은 JIT 커널을 자기 row 청크에 호출.

핵심 디자인 결정: **언제 멀티스레드 켤지의 임계값**. 처음에는 `M ≥ 256`
으로 설정 → 잘 작동했지만 qkv (M=896, K=896) 같은 작은 매트멀에서
회귀 발생 (실측 49ms → 62ms, +27%). 원인: rayon 디스패치 overhead가
실제 compute보다 큼.

수정: `M × K_blocks ≥ 100K` 임계값. 총 work로 판단.
- gate_up (M=4864, K_blocks=28 = 136K): parallel ✓
- down (M=896, K_blocks=152 = 136K): parallel ✓
- wq (M=896, K_blocks=28 = 25K): **serial** (overhead 회피)
- lm_head (M=151936, K_blocks=28 = 4.25M): parallel ✓

`std::thread::available_parallelism()` 사용 (8 코어 제한, L3 contention
방지). pointer 전달은 `as usize` cast 후 closure 안에서 다시 cast —
raw pointer Send 문제 회피.

**결과: single-thread ~31 → 8-thread ~56 tok/s (+1.81×).**

## 정직한 ggml 비교

```
                       1 thread   8 threads
ggml (llama.cpp):       41.32      90.90
Lumen v0.3.0:           ~31        ~56

Lumen single → multi:   1.81× scaling
ggml single → multi:    2.20× scaling
```

같은 코어 수 비교에서 우리는 1.62× 느리고, 스케일링도 ggml이 더 좋음
(2.20 vs 1.81). 우리가 멀티스레드를 켰을 때 ggml의 single-thread는
넘었지만 (1.36× faster), 같은 thread count로는 여전히 격차.

남은 격차의 원인 추정:
1. **Per-call rayon dispatch** — 169개 매트멀 × ~5μs = 0.85ms/token =
   2.7% overhead. 자체 thread pool로 줄일 수 있음.
2. **Q8-native int dot product** — ggml은 `vpdpbusd` 같은 정수 SIMD로
   직접 int8 dot product. 우리는 dequant fp32 → fp32 FMA. 이론적으로
   2-4× 격차의 원인.
3. **No thread pinning** — ggml은 OS scheduling 신뢰 안 하고 자기가
   thread affinity 관리.
4. **No AVX-512** — 갖춰진 CPU 기준 추가 1.5-2× 가능.

## 무엇이 달라진 마음가짐

v0.2.0까지는 "이 ops 빠르게 만들어야지" 식 직관 기반. v0.3.0은
**"먼저 측정, 그 다음 결정"**. 7.E.0 한 commit이 7.E (flash attention)
한 phase 통째로 폐기시켰고, 7.J 임계값 첫 번째 cut도 측정 데이터로
바로 수정.

좋은 시스템이 빠른 게 아니라 **어디가 느린지 즉시 보이는 시스템이
좋은 시스템**입니다. v0.1.0의 BENCHMARK.md, v0.3.0의 StepTimer —
이게 v0.1 → v0.3에서 12.6× decode 향상을 가능하게 한 도구.

## v0.3.0 약속 / 안 약속

**약속:**
- Qwen2.5-0.5B-Q8_0 한국어 추론, **단일 스레드 ~31 / 8-thread ~56 tok/s**
- 토큰 시퀀스가 naive Rust path와 bit-identical
- 외부 추론 의존성 0개 (rayon만 추가됨, ggml/ONNX/candle/mlx 임포트 없음)

**안 약속:**
- ggml과 같은 자릿수 (8-thread에서 1.62× 격차)
- ARM64 (Phase 2.C 미진입)
- prefill batch (decode만 최적화)
- AVX-512 (지원 CPU에서 fallback)

## 다음

`v0.3.0 → v0.4.0`의 가능한 갈래:

1. **자체 thread pool로 rayon 대체** — 8 코어 만큼만 spawn하는 작은
   pool. rayon overhead 줄이기. 격차 1.62× → ~1.4× 예상.
2. **Q8-native int dot product** — `vpmaddubsw`나 `vpdpbusd`로 정수
   SIMD. 큰 일이지만 ggml의 핵심 기술. 격차 1.62× → ~1.0×.
3. **Token embeddings도 Q8** — 545MB → 144MB. 메모리 절약, decode
   영향은 작음.
4. **Prefill batch path** — decode와 다른 매트멀 형상 (M>1). 긴 프롬프트
   가속.

2번이 가장 큰 잠재력. 1번이 가장 빠른 win. 4번이 가장 useful for users.

코드: [github.com/redchupa/lumen](https://github.com/redchupa/lumen), 태그 `v0.3.0`.
