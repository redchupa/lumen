# VNNI 한 줄이 왜 안 빨라졌나 — Phase 7.M & 7.N 후기

> Lumen 빌드로그 #11. 측정-기반 의사결정의 두 번째 사례 모음.

## TL;DR

- Phase 7.M: **Q8 weights × Q8 activations** 정수 dot product (AVX2)
  를 구현. 텍스트북 trick. 측정 결과 **net-neutral**(-1.3% bench)이라
  default 비활성화.
- Phase 7.N: **vpdpbusd** (AVX-VNNI / AVX-512 VNNI)로 같은 dataflow를
  단축. 측정 결과 **-3.6% bench** on Zen 4. default 비활성화.
- 두 phase의 인프라(encoders, IR pattern, kernels, runtime dispatch)는
  유지. multi-acc VNNI 또는 AVX-512 ZMM fp32 같은 다음 layer가 들어오면
  즉시 활성화될 수 있게.

토큰은 두 phase 모두 naive Rust path와 bit-identical.

## 왜 시작했나

Phase 7.J 직후 ggml과 격차 1.5×. 그 격차의 원인 추정:
1. ggml가 `vpdpbusd` (정수 dot) 쓰는데 우리는 fp32 FMA로 dequant
2. ggml가 AVX-512 lane width (512-bit) 쓰는데 우리는 AVX2 (256-bit)

7.M은 (1)을 AVX2-only로 해결, 7.N은 (1)을 VNNI로 해결 시도.

## Phase 7.M — AVX2 int dot

### 무엇을 만들었나

기존 Q8×F32 커널 (Phase 7.L):
```
load 32 i8 weights → vpmovsxbd → 8 i32 → vcvtdq2ps → 8 fp32
  → vmulps × d (broadcast f32) → vfmadd231ps with f32 activations
```

새 Q8×Q8 커널 (Phase 7.M):
```
load 32 i8 weights, 32 i8 activations
vpsignb |w|, vpsignb a*sign(w)
vpmaddubsw → 16 i16 pair-sums
vpmaddwd  → 8 i32 pair-sums
vcvtdq2ps → 8 fp32 partials
* (d_w × d_a) broadcast → vfmadd231ps into fp32 row acc
```

새 코드:
- 7개의 AVX2 정수 인코더 (`vpsignb`, `vpmaddubsw`, `vpmaddwd`, `vmovdqu`,
  `vpaddd`, `vpcmpeqd`, `vpsrlw_imm8`)
- `ones16` 상수를 함수 시작에서 한 번 forge (`vpcmpeqd self,self,self`
  → `vpsrlw 15` = 0x0001 per i16 lane)
- 2개의 Dequantize op을 갖는 새 IR 패턴 인식
- `MatmulJitCache::get_or_compile_q8q8` 메서드
- `model.rs` 통합: RMSNorm 출력을 Q8으로 양자화 후 q/k/v 매트멀이
  공유 (1회 양자화, 3회 사용)

### 측정

| Step (32-token sum) | 7.L Q8×F32 | 7.M Q8×Q8 | Δ |
|---|---:|---:|---:|
| gate_up_matmul | 229 ms | 194 ms | **-15%** ✓ |
| lm_head_matmul | 99 ms | 90 ms | -9% ✓ |
| qkv_matmul | 53 ms | 41 ms | -22% ✓ |
| wo_matmul | 35 ms | 32 ms | -9% ✓ |
| **down_matmul** | **112 ms** | **141 ms** | **+26%** ✗ |
| Total decode | 562 ms | 530 ms | -6% |
| **Bench tg32 mean** | **62 tok/s** | **58 tok/s** | **-1.3% (noise)** |

대부분 매트멀이 좋아졌는데 **down_matmul만 +26% 회귀**. down은
K_blocks=152 (다른 매트멀은 28). 블록당 dependency chain이 길어서
누적.

Bench 결과는 거의 노이즈 (-1.3%). Profile의 -6%와 bench의 -1.3% 차이는
Instant::now() 호출 overhead가 profile 분포를 살짝 흔든 결과로 보임.

### 결론

AVX2 only의 `vpsignb + vpmaddubsw + vpmaddwd` 체인:
- Critical path ~20 cycles per block (단일 직선 chain)
- 기존 4-acc fp32 FMA path: 4 independent chains @ throughput-limited
  ~5 cycles per block

**ILP-rich한 fp32 path를 latency-bound int chain이 못 이김.**
ggml이 실제로 쓰는 건 단일 instruction `vpdpbusd` — 그게 다음 phase.

## Phase 7.N — VNNI로 단축

### 무엇을 만들었나

`vpdpbusd`: `acc_i32[i] += sum_4_pairs(u8 × i8)`. 한 instruction이
7.M의 `vpmaddubsw + vpmaddwd` 두 instructions + 그 사이 종속성을
대체.

새 코드:
- VEX-256 `vpdpbusd_vex_reg` 인코더 (Intel AVX-VNNI: Tiger Lake / Alder
  Lake / Raptor Lake)
- **EVEX-256 `vpdpbusd_evex_reg` 인코더** (AVX-512 VNNI: Skylake-X /
  Sapphire Rapids / Zen 4) — 4-byte EVEX prefix 직접 인코딩
- `VnniForm` enum (`Vex256` / `Evex256`)
- `CodegenOpts.vnni`로 codegen에 dispatch
- 새 커널 `emit_quant_matmul_q8q8_n1_body_vnni`
- `MatmulJitCache::get_or_compile_q8q8`에서 런타임 CPU 감지:
  `avx512vnni` > `avxvnni` > AVX2 fallback

EVEX 인코딩 직접 작성한 흥미로운 부분:

```rust
// EVEX prefix for vpdpbusd ymm0, ymm0, ymm0 → 6 bytes:
//   0x62 0xF2 0x7D 0x28 0x50 0xC0
em.u8(0x62);                    // EVEX marker
// P0: R̄ X̄ B̄ R̄' 0 0 mmm    (R/X/B/R' inverted; mmm=010 for 0F38)
em.u8(r_bar | x_bar | b_bar | r_prime_bar | 0b010);
// P1: W vvvv 1 pp           (W=0; vvvv inverted; pp=01 for 66 prefix)
em.u8(vvvv_inv | (1 << 2) | 0b01);
// P2: z L'L b V̄' aaa         (L'L=01 for 256-bit, V̄'=1, no mask)
em.u8((0b01 << 5) | (1 << 3));
em.u8(0x50);                    // opcode
em.u8(modrm);
```

이 box (AMD Zen 4): `avxvnni=false` but `avx512vnni=true`. EVEX-256
form은 실제 동작 ✓ — 7개 shape unit-test 모두 통과 (M up to 4864, K up
to 4864).

### A/B 측정 (Zen 4, 8 runs each)

```
has_vnni() = false (Q8×F32 4-acc):     mean 60.9 tok/s
has_vnni() = true  (Q8×Q8 EVEX vpdpbusd): mean 58.7 tok/s
                                          → -3.6% slower
```

### 왜 회귀

VNNI 커널 critical path per block:
```
vpdpbusd → vcvtdq2ps → vfmadd231ps    (~10 cycles, ONE chain into ymm0)
```

기존 7.L Q8×F32 4-acc 커널 critical path per block:
```
4 independent vpmovsxbd→vcvtdq2ps→vmulps→vfmadd231ps chains    (~5 cycles, ILP)
```

**Single-accumulator VNNI가 4-acc fp32 ILP보다 throughput-limited 시점에
서 진다.** vpdpbusd 자체는 vpmaddubsw+vpmaddwd 체인의 두 배 빠르지만,
그 다음 vcvtdq2ps + vfmadd231ps 종속성이 ymm0 단일 누산기로 직렬화됨.

다음 단계가 명확:
1. **Multi-acc VNNI 커널** — 4 fp32 row sub-accumulators, 7.G에서 fp32에
   적용한 trick과 동일.
2. **AVX-512 ZMM fp32** — 2× lane width.

## 두 번 데인 후 배운 것

### 1. 텍스트북 trick이 자동 win은 아니다

"Q8 native int dot이 dequant→fp32 multiply보다 빠르다"는 통념. 측정해보니
그건 **단일 instruction에 fused됐을 때만 (vpdpbusd 한 줄)** 사실. 분리된
체인(`vpsignb + vpmaddubsw + vpmaddwd`)으로는 ILP-rich fp32 path를 못
이김. 그리고 vpdpbusd가 있어도 다음 단계(vcvtdq2ps + fma)가 직렬화되면
single-acc bottleneck.

### 2. 같은 K_blocks가 다른 결과를 만든다

down_matmul (K_blocks=152) vs gate_up (K_blocks=28). 모든 매트멀이 같은
커널을 쓰는데도 한 형상에서만 회귀(+26%)가 났음. 블록당 체인 길이가
누적되는 정도가 달라서. 측정 없이 "이 커널이 빠르다/느리다"는 평가가
shape-dependent임을 잊으면 안 됨.

### 3. 인프라는 살아 있다

7.M의 Q8×Q8 codegen, 7.N의 VNNI encoders + 3-way CPU dispatch, model.rs의
quantize_activation_q8 + q8q8_matmul_dispatch — 모두 `default-off`로
잠들어 있지만 한 줄 (`has_vnni() = false` → `true`)이면 즉시 활성화.

multi-acc VNNI 커널이 들어오면 이 인프라가 그대로 win 경로가 됨. AVX-512
ZMM fp32 커널 들어오면 또 다른 fast path. **결정을 미루지만 옵션은
열어둠.**

### 4. 측정이 결정을 했다, 두 번

Phase 7.M에서도 7.N에서도 의식적인 결정: "텍스트북 직관 vs 측정 데이터,
측정을 따른다." 둘 다 default-off로 두는 결정은 코드 작성의 가치를
부정하지 않음 — 측정 결과를 정직하게 반영하는 것뿐.

## 다음

두 갈래:
- **A) Multi-acc VNNI 커널** — 7.G의 fp32-인 4-acc trick을 VNNI에 적용.
  중간 크기 작업. 잠재 win: VNNI single-acc 한계 극복 → +15-25% 예상.
- **B) AVX-512 ZMM fp32 커널** — EVEX 인코더 확장 + zmm-FMA. 큰 작업.
  잠재 win: 2× lane width → ggml과 같은 자릿수.

작은 win 먼저 → 큰 win 패턴대로면 A → B.

코드: [github.com/redchupa/lumen](https://github.com/redchupa/lumen), 현재
태그 `v0.3.0`. 다음 일정량의 win이 쌓이면 v0.4.0.
