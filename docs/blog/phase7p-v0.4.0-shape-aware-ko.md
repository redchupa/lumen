# v0.4.0 — 직관 세 번 부정당하고, 하이브리드가 답이었다

> Lumen 빌드로그 #12. Phase 7.M ~ 7.P 합본.

## 결론

```
v0.3.0 tg32 (8 threads):  ~60 tok/s
v0.4.0 tg32 (8 threads):  ~65 tok/s    (+8%)

vs llama.cpp:
  Lumen v0.4.0 (8t) vs ggml -t 1 (41.32):  Lumen 1.58× faster
  Lumen v0.4.0 (8t) vs ggml -t 8 (90.90):  ggml  1.39× slower (was 1.45×)

v0.1.0 → v0.4.0: 4.43 → 65.3 tok/s = 14.7× decode
```

토큰은 v0.1.0부터 v0.4.0까지 모든 commit에서 naive Rust path와
bit-identical.

## 무엇이 v0.4.0의 단 4% win을 만들었나

요약: 네 phase 동안 **세 번** "직관적 win" 시도가 측정으로 부정됐고,
네 번째에서 **하이브리드 dispatch**로 답이 나왔습니다.

| Phase | 시도 | 측정 결과 | 결정 |
|---|---|---|---|
| 7.M | Q8×Q8 AVX2 int dot | bench 동등 (down +27% 회귀) | default-off, 인프라 유지 |
| 7.N | VNNI vpdpbusd (single-acc) | -3.6% on Zen 4 | default-off, 인프라 유지 |
| 7.O | VNNI 4-accumulator | -2.7% (down +27% 그대로) | default-off, 인프라 유지 |
| **7.P** | **K-shape별 dispatch** | **+4.5% bench** | **default-on** ✓ |

7.M, 7.N, 7.O에서 "정수 dot product가 더 빨라야 한다"는 직관을 따라
세 번 시도. 매번 측정이 부정. 처음엔 답답했지만, **세 번의 데이터가
모이니까 7.P가 가능**해졌습니다.

## 7.O에서 7.P로 — 데이터가 답을 지목

7.O에서 4-acc VNNI를 측정했을 때 per-step 분포:

```
gate_up (K_blocks=28):   229 → 195 ms  (-15%) ✓
lm_head (K_blocks=28):   99 → 90 ms   (-9%)  ✓
qkv     (K_blocks=28):   53 → 38 ms   (-28%) ✓
wo      (K_blocks=28):   35 → 30 ms   (-14%) ✓
down    (K_blocks=152): 112 → 142 ms  (+27%) ✗
```

5개 매트멀 중 4개가 **VNNI에서 이김**. 1개(down)가 **정확히 그만큼
잃음**. 합치면 wash. K_blocks=28 그룹과 K_blocks=152 그룹이 정반대
부호.

진단:
- VNNI 4-acc 커널: 블록당 14 instructions
- fp32 4-acc 커널: 블록당 4 instructions
- 짧은 inner loop (28 blocks/row)에서는 VNNI의 latency-short critical
  path가 fp32의 throughput 우위를 이김
- 긴 inner loop (152 blocks/row)에서는 VNNI의 더 많은 instructions이
  CPU OoO window를 saturate. fp32가 lean inner로 이김

답: **shape마다 다른 path를 쓰자.** Phase 7.P.

```rust
fn use_vnni_for_matmul(d_in: usize) -> bool {
    if !has_vnni() { return false; }
    (d_in / 32) <= 64  // empirical threshold between 28 and 152
}
```

forward의 모든 `if has_vnni()` 자리를 `if use_vnni_for_matmul(d_in)`로
교체. 각 매트멀이 자기 K_blocks에 따라 자기에 맞는 path 선택.

결과:

```
gate_up (28):  194 ms ← VNNI 유지
lm_head (28):   89 ms ← VNNI 유지
qkv     (28):   38 ms ← VNNI 유지
wo      (28):   30 ms ← VNNI 유지
down   (152):  103 ms ← fp32 4-acc (VNNI 회피)
Total:         487 ms (was 562ms in 7.L; -13.3%)
Bench tg32:    65.3 tok/s (was 62.5; +4.5%)
```

각 매트멀이 자기 winning path를 가져갔고, 전체적으로 가장 빠른
조합이 됨.

## 왜 7.M, 7.N, 7.O가 다 필요했나

각 phase가 "삽질"이 아니라 **다음 win의 인프라**였음:

- **7.M**: Q8×Q8 IR 패턴 + AVX2 정수 인코더 7개 (vpsignb, vpmaddubsw,
  vpmaddwd, vmovdqu, vpaddd, vpcmpeqd, vpsrlw_imm8) + 첫 Q8×Q8 codegen.
  → 7.P가 호출하는 fp32 fallback path도 이 인프라 위에서 작동.
- **7.N**: VEX/EVEX vpdpbusd 인코더 + 3-way CPU detection + VnniForm
  enum + CodegenOpts.vnni propagation.
  → 7.P의 VNNI 경로가 이 모든 걸 그대로 사용.
- **7.O**: 4-accumulator VNNI 커널 + ymm6 save/restore + 3-way codegen
  dispatch (4-acc / single-acc / AVX2).
  → 7.P의 짧은-K 매트멀이 모두 7.O 커널을 호출.

만약 7.M에서 "AVX2 int dot은 안 빠르네, 그만하자" 하고 멈췄으면 7.P의
하이브리드 dispatch는 존재할 수 없었음. 측정이 "이 길이 맞다"고 명확히
지목할 만큼의 데이터가 쌓이기 전까지 인프라를 계속 만드는 게 중요.

## 인프라 자산 한 화면

v0.4.0에 들어있는 코드 자산:

| Layer | 자산 |
|---|---|
| AVX2 정수 인코더 | vpsignb, vpmaddubsw, vpmaddwd, vmovdqu, vpaddd, vpcmpeqd, vpsrlw_imm8 |
| VNNI 인코더 | vpdpbusd VEX-256 (AVX-VNNI) + EVEX-256 (AVX-512 VNNI) |
| Codegen IR | Q8×Q8 fused matmul 패턴 (2× Dequantize + MatMul) |
| Codegen kernels | Q8×Q8 AVX2 single-acc, VNNI single-acc, VNNI 4-acc (Win64 callee-saved ymm6 처리) |
| Runtime dispatch | MatmulJitCache 3-way CPU detection (avx512vnni > avxvnni > AVX2) |
| Model integration | per-shape `use_vnni_for_matmul()` dispatch, activation quantization pipeline |
| Tests | byte-level encoder tests, 7-shape Q8×Q8 unit test, qwen_generate_jit_matches_naive_and_speed |

이 모든 게 default-on으로 실제 production path. 단 한 조각도 dead
code가 아님 (7.P가 모든 path를 실제 forward에서 호출).

## v0.4.0 정직한 standing

벤치 (Qwen2.5-0.5B-Q8_0, Windows AVX-512 Zen 4, 8 worker threads):

```
                       single thread   8 threads
ggml (llama.cpp):       41.32           90.90
Lumen v0.4.0:           ~33             ~65

  Lumen 8t (65) vs ggml -t 1 (41.32):  Lumen 1.58× FASTER
  Lumen 8t (65) vs ggml -t 8 (90.90):  ggml  1.39× faster
```

같은 thread 수에서 ggml이 1.39× 빠릅니다. 남은 격차의 진짜 원인:
**lane width**. ggml은 AVX-512 (ZMM, 512-bit), 우리는 AVX2 (YMM, 256-bit).
ZMM 폭에서는 한 instruction이 2× 처리.

이 격차를 좁히려면 EVEX 인코더 확장 + AVX-512 ZMM ops + 새 커널이
필요. v0.5의 큰 작업.

## v0.4.0 약속 / 안 약속

**약속:**
- Qwen2.5-0.5B-Q8_0 한국어 추론, **~65 tok/s (8 threads, Windows AVX-512)**
- 토큰이 naive Rust path와 bit-identical
- 외부 추론 의존성 0개 (production deps: `thiserror` 하나)
- CPU별 자동 best-path selection: AVX-512 VNNI > AVX-VNNI > AVX2 fp32
- shape별 자동 best-kernel selection: short-K → VNNI, long-K → fp32 4-acc

**약속하지 않음:**
- ggml 동등 (현재 8t에서 1.39× slower; AVX-512 ZMM 추가가 v0.5 작업)
- ARM64 (Phase 2.C 미진입)
- prefill batch (decode만 최적화)
- AVX-512 ZMM (v0.5)

## v0.4 → v0.5의 길

명확한 한 갈래:

**AVX-512 ZMM fp32 + ZMM VNNI** — EVEX 인코더 확장(현재 vpdpbusd만
EVEX), zmm 폭 fp32 FMA/load/store ops 추가, 새 ZMM 4-acc 커널들.
잠재 win: lane width 2× → 매트멀 처리량 2× → ggml과 같은 자릿수
도달 잠재력.

v0.5 한 phase로 끝낼 일은 아니고 sub-phases로 진행 예상.

코드: [github.com/redchupa/lumen](https://github.com/redchupa/lumen), 태그 `v0.4.0`.
