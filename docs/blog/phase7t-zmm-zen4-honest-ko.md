# Zen 4에서 AVX-512가 안 빠른 이유 — Phase 7.R ~ 7.T 후기

> Lumen 빌드로그 #13. v0.5의 초기 4 phases. perf release는 못 했지만
> 인프라가 살아 있고, 측정-주도 의사결정이 또 한 번 옳았음을 확인.

## TL;DR

- v0.5 목표는 **AVX-512 ZMM 폭(512-bit)으로 ggml 1.39× 격차 좁히기**.
- Phase 7.R: ZMM 인프라 (Zmm 타입, 일반 EVEX 인코더, 5 ZMM ops).
- Phase 7.S: ZMM Q8×F32 4-acc 커널 (down_matmul용).
- Phase 7.T: ZMM 4-acc Q8×Q8 커널 (vpsignb EVEX 부재 우회: vpmovsxbw + vpmaddwd).
- 모든 ZMM 커널이 정답성 통과, **Zen 4에선 측정 결과 -2.6% ~ -4.5% 회귀**.
- 진단: **Zen 4 AVX-512가 double-pumped 256-bit 구현**. ZMM ops가 두 개의
  256-bit µops로 분해됨. lane width win 안 됨.
- 결정: 인프라 유지, `host_supports_avx512_q8_kernel() → false` default,
  forced-AVX-512 unit test로 ZMM 코드 경로 CI 검증 유지.

토큰은 모든 commit에서 naive Rust path와 bit-identical.

## 출발점

v0.4.0 마치고 standing:
- Lumen 8t: ~65 tok/s
- ggml 8t: 90.9 tok/s
- 1.39× 격차

격차의 가장 큰 가설은 **lane width**: ggml AVX-512 (ZMM 512-bit) vs 우리
AVX2 (YMM 256-bit). 이론상 매트멀 throughput 2×.

v0.5 진입 = ZMM 폭으로 가자.

## Phase 7.R — EVEX 인프라

ZMM ops는 모두 EVEX-encoded. EVEX prefix는 VEX보다 복잡:
```
byte 0 = 0x62 (EVEX marker)
byte 1 (P0) = R̄ X̄ B̄ R̄' 0 0 m m       ; R/X/B/R' inverted; mm = opcode map
byte 2 (P1) = W vvvv 1 p p             ; vvvv inverted; pp = legacy prefix
byte 3 (P2) = z L' L b V̄' a a a         ; L'L = lane width; V̄' inverted
```

추가:
- `Zmm(u8)` 타입 (Ymm과 별도, type-safety)
- `emit_evex` 일반 helper (R/X/B/R'/V' bits 비-반전 입력, 헬퍼가 반전)
- 기존 `vpdpbusd_evex_reg` (Phase 7.N 수동 작성) 를 helper 통해 재구현
  — byte test로 동일 출력 검증 (regression check)
- 5개 ZMM-512 ops: `vxorps_zmm_zero`, `vmovups_zmm_load`,
  `vbroadcastss_zmm`, `vfmadd231ps_zmm_mem`, `vfmadd231ps_zmm_reg`

각각 byte-pattern unit test로 검증. 5개 새 테스트 추가 (36→41).

## Phase 7.S — ZMM Q8×F32 4-acc

Target: down_matmul (M=896, K=4864, K_blocks=152). Phase 7.P에서
`use_vnni_for_matmul()` 가 false 반환 → Q8×F32 path로 routing.

기존 YMM 4-acc (Phase 7.G):
- 4 sub-accs (ymm0..3), 8 lanes each
- 블록당 4 chunks of 8 elements, 각 chunk → 다른 sub-acc

ZMM 2-acc (7.S):
- 2 sub-accs (zmm0, zmm1), 16 lanes each
- 블록당 2 chunks of 16 elements, 각 chunk → 다른 sub-acc

같은 work (32 K-elements per block, M output rows), 절반 inner ops.

6 추가 ZMM 인코더: `vpmovsxbd_zmm_load`, `vcvtdq2ps_zmm`,
`vmulps_zmm_reg`, `vbroadcastss_zmm_xmm`, `vaddps_zmm_reg`,
`vextractf32x8_zmm`.

CodegenOpts에 `use_avx512: bool` 추가. `Backend::lower` → `emit_function`
→ `emit_function_quant_matmul_q8` 까지 전달.

측정 (Zen 4, 8 tg32 runs):

```
v0.4.0 baseline (all YMM):              mean 65.3 tok/s
7.S (ZMM Q8×F32 on down only):           mean 63.6 tok/s   (-2.6%)
Per-step profile:
  down_matmul: 103 → 99 ms / 32 tokens   (-4%)
```

down 자체는 미미하게 빨라졌는데 전체는 약간 느려짐. activation
quantization + EVEX overhead가 이득 잡아먹음.

## Phase 7.T — ZMM 4-acc Q8×Q8

처음 계획: Phase 7.O VNNI YMM 4-acc 커널을 ZMM으로 직접 확장.

벽: **`vpsignb`에 EVEX form이 없음.** SSSE3 시절 instruction. AVX-512가
바이트-수준 signed-sign manipulation까지 확장 안 함.

VNNI vpdpbusd는 `u8 × i8 → i32`. signed × signed (우리 Q8×Q8) 처리하려면
vpsignb trick (`|w|` as u8, `a*sign(w)` as i8) 필요. vpsignb 없으면 막힘.

다른 길 발견: **`vpmovsxbw + vpmaddwd`**:
- 둘 다 AVX-512BW에 EVEX-512 form 있음
- `vpmaddwd`는 signed i16 × signed i16 → i32 (symmetric in signedness)
- 32 signed i8 (Q8 block) → vpmovsxbw → 32 signed i16 → vpmaddwd → 16 i32

per block ~7-8 ops + scale, vs YMM VNNI 4-acc 14 ops. **거의 절반!**

추가 2 인코더: `vpmovsxbw_zmm_load`, `vpmaddwd_zmm_reg`.

Register plan (Win64-safe + 1 stack-saved):
- zmm0..3: 4 fp32 sub-accumulators
- zmm4: i16 weights / fp32 partials (sequential reuse)
- zmm5: i16 acts / scale broadcast (sequential reuse)
- ymm6: scalar d_a temp (stack-saved at prologue, restored at epilogue)

Phase 7.O의 ymm6 save/restore 패턴 재사용. RSP-base 메모리 액세스는
SIB 형식 필수.

측정 (Zen 4, 8 tg32 runs each):

```
v0.4.0 baseline (all YMM):                mean 65.3 tok/s
7.S only (Q8×F32 ZMM):                     mean 63.6 tok/s   (-2.6%)
7.T full (Q8×F32 ZMM + Q8×Q8 ZMM):         mean 62.4 tok/s   (-4.5%)
```

또 회귀. 누적 -4.5%.

## Zen 4에서 ZMM이 안 빠른 이유

세 번 측정 후 분명한 진단: **AMD Zen 4는 AVX-512를 "double-pumped"
256-bit datapath로 구현.**

내부적으로 ZMM op = 두 개의 256-bit µops 발급. 따라서:
- **Throughput**: 8 lanes/cycle (same as YMM)
- **Code size**: EVEX prefix 4-byte vs VEX 2-3 byte (오버헤드)
- **Dependency chain**: vpmovsxbw + vpmaddwd (3+5=8 cycle latency) vs
  VEX vpdpbusd (5 cycle) — longer
- **Lane width "win"이 사라짐.**

진짜 native 512-bit silicon — Intel Sapphire/Emerald/Granite Rapids,
AMD Zen 5+ — 에서는 다를 것. ZMM ops가 한 cycle에 16 lanes 처리하면
2× win 실현. 우리는 그 하드웨어가 없음.

## 그래도 결과: 인프라 유지

세 번 측정해서 부정적 결과로 끝났지만, **인프라는 모두 살아 있음**:

| 자산 | 위치 |
|---|---|
| Zmm 타입 + 일반 emit_evex helper | avx2_enc.rs |
| 12개 ZMM-512 EVEX 인코더 | avx2_enc.rs |
| Q8×F32 ZMM 2-acc 커널 | x86_64.rs |
| Q8×Q8 ZMM 4-acc 커널 (vpmovsxbw+vpmaddwd path) | x86_64.rs |
| CodegenOpts.use_avx512 + 4-way codegen dispatch | x86_64.rs |
| forced-avx512 unit test | matmul_cache.rs |

`host_supports_avx512_q8_kernel()` 한 줄을 native-512 silicon 감지로
바꾸면 즉시 활성화. CPUID flag만으로 구별 안 되니, CPU model 기반
heuristic 필요 (예: 가족/모델 번호로 Zen 5 vs Zen 4 구분).

## 측정-주도 의사결정의 네 번째 사례

| Phase | 시도 | 측정 | 결정 |
|---|---|---|---|
| 7.M | Q8 native AVX2 int dot | net-neutral | default-off, infra |
| 7.N | VNNI single-acc | -3.6% | default-off, infra |
| 7.O | VNNI 4-acc | -2.7% | default-off, infra → 7.P가 부활시킴 |
| **7.R+7.S+7.T** | **AVX-512 ZMM** | **-4.5%** | **default-off, infra** |

7.O의 인프라는 7.P에서 부활했음. 똑같이 7.R/S/T의 ZMM 인프라는 미래
어느 phase (또는 다른 호스트) 에서 활성화될 수 있음.

**직관: "더 큰 lane이 더 빠를 것"**. 사실에 대한 검증:
**Zen 4는 그렇지 않다**.

## v0.5 미release

이 phases들이 perf release를 만들지 못함. v0.5 태그는 없음. main은
v0.4.0 perf 그대로 유지 (~65 tok/s) + ZMM 인프라 추가.

다음 자연스러운 갈래:
1. **다른 하드웨어로 측정** — Intel SPR / Zen 5에서 ZMM 켜고 측정. win이면 정책 켜기.
2. **lane width 외 다른 격차 원인 조사** — perf counter로 thread overhead, memory bandwidth, 동시 요청 등 측정.
3. **다른 모델 / ARM64** — 다른 타깃으로 확장.

코드: [github.com/redchupa/lumen](https://github.com/redchupa/lumen), 태그 `v0.4.0` (unchanged).
