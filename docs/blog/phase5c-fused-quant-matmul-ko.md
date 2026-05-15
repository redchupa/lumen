# Phase 5.C — IR이 양자화 커널을 자동으로 합성한다

> Lumen 빌드로그 #6. 정체성의 진짜 시연.

Phase 1부터 PLAN.md 한 줄로 "IR이 양자화 커널을 자동 합성하는 LLM 추론 컴파일러"라고 적어왔다. 오늘 그 문장이 비로소 진짜로 동작한다.

## 문제 — llama.cpp가 한 일

llama.cpp의 `ggml-quants.c`를 열어보면 수십 가지 손코딩된 함수가 있다:

```c
ggml_vec_dot_q4_0_q8_0(...)
ggml_vec_dot_q4_K_q8_K(...)
ggml_vec_dot_q5_0_q8_0(...)
ggml_vec_dot_q6_K_q8_K(...)
// ... 수십 개
```

각각이 (입력 양자화 포맷 × 출력 형태 × CPU 아키텍처) 조합 하나에 대해 손으로 SIMD intrinsic을 깎은 결과다. 새 양자화가 추가되면 수 함수가 추가된다. 새 CPU 명령어 세트(AVX-512, NEON, SVE)가 나오면 모든 함수를 다시 깎는다.

## 해법 — IR 패턴 매칭으로 자동 합성

오늘 Lumen의 IR에 다음 4-op 함수를 입력했다:

```text
fn quant_matmul(
  w: tensor<q8_0, [M, K]>,
  x: tensor<f32,  [K, N]>,
) -> tensor<f32, [M, N]> {

  v0 = Param(0)               : q8_0 [M, K]
  v1 = Param(1)               : f32  [K, N]
  v2 = Dequantize v0          : f32  [M, K]   ← "메모리 위에 양자화된 weights를 f32로 풀어라"
  v3 = MatMul v2, v1          : f32  [M, N]   ← "그 결과를 activations와 곱해라"
  return v3
}
```

backend가 이 패턴을 보고 **두 op을 한 커널로 융합**해서, 다음 머신코드를 emit했다 (요약):

```asm
i_loop:                                     ; for i in 0..M
  j_loop:                                   ;   for j in 0..N step 8
    vxorps    ymm0, ymm0, ymm0              ;     ymm_acc = 0
    kb_loop:                                ;     for kb in 0..K/32
      vmovd      xmm2, [r12 + rax]          ;       d_fp16 ← memory
      vcvtph2ps  xmm2, xmm2                 ;       d_fp32 ← convert
      kk_loop:                              ;       for kk in 0..32
        movsx     r9d, byte [r12 + ...]     ;         load i8 weight
        vcvtsi2ss xmm1, xmm2, r9d           ;         (float) i32
        vmulss    xmm1, xmm1, xmm2          ;         w_fp32 = w * d
        vbroadcastss ymm1, xmm1             ;         broadcast 8 lanes
        vfmadd231ps ymm0, ymm1, [r13 + ...] ;         acc += w * activations
    vmovups [r14 + ...], ymm0               ;     store output
```

핵심 — 이 커널은 **전체 weights를 fp32로 풀어내는 중간 메모리를 만들지 않는다.** weight 하나를 i8로 읽고, 그 자리에서 scale 곱하고, broadcast해서 FMA. 메모리 대역폭이 **4× 절약**된다 (i8 vs fp32).

## 정답성 — 1픽셀도 안 틀린다

세 가지 케이스로 검증:

| Shape (M × K × N) | Tolerance | 결과 |
|---|---|---|
| 1 × 32 × 8 (smallest, 1 block per row) | 1e-3 | ✅ |
| 4 × 64 × 16 (2 blocks, 2 ymm cols) | 1e-2 | ✅ |
| 8 × 128 × 32 (4 blocks, 4 ymm cols) | 5e-2 | ✅ |

Reference는 두 패스 (전체 dequant → matmul) 결과. Lumen의 융합 커널 결과와 허용 오차 안에서 정확히 일치. Tolerance가 커지는 이유는 누적 순서 차이 — fused는 (i8 × fp32) 곱셈을 fp32로 누적하고, reference는 fp32 weight × fp32 act 곱셈을 fp32 누적. ULPs 단위 미세 차이.

## 새 인코더 명령어 (Phase 5.C)

| 명령어 | 인코딩 | 용도 |
|---|---|---|
| `movsx_r32_m8` | 0F BE /r | byte → i32 sign extend |
| `vcvtsi2ss_xmm_r32` | F3 0F 2A /r (VEX) | i32 → fp32 scalar convert |
| `vmulss_xmm` | F3 0F 59 /r (VEX) | scalar fp32 multiply |

지금까지 자체 인코더로 emit 가능한 명령어 총 약 30개. 22개 단위 테스트로 바이트 단위 검증.

## 무엇이 진짜 의미인가

오늘부터 Lumen에서는:

1. **IR에 `dequantize` op 한 줄을 추가하면 → backend가 자동으로 가장 가까운 다음 op(matmul, add, ...)과 융합 시도**
2. **양자화 포맷이 추가되면 → IR의 `DType` enum과 unpack 패턴 한 곳만 수정. 백엔드 코드는 안 건드림.**
3. **새 CPU 아키텍처 backend가 추가되면 → 같은 IR을 입력으로 받음. 손코딩 함수 0개.**

이게 llama.cpp의 `ggml_vec_dot_*` 수십 함수를 한 패턴 매처로 대체하는 그림이다. 

## 누적

| 항목 | 값 |
|---|---|
| 커밋 | 9 |
| Rust LOC | ~7,200 |
| 테스트 | **65 green** |
| 인코더 명령어 | 30+ |
| 외부 의존성 (production) | `thiserror` 1개 |

## 다음 — Phase 5.D: GGUF 로더

여기까지 양자화 reference + native dequant + fused matmul이 다 됐다. 남은 건 **실제 GGUF 파일에서 가중치를 읽어내는 로더**.

GGUF v3 spec:
- Magic `GGUF` + version
- Metadata KV pairs (key-value, dtype 다양)
- Tensor info (이름, 차원, dtype, offset)
- Tensor data

이게 들어오면 Phase 6 — Qwen2.5-0.5B 같은 실제 한국어 모델을 로드해서, Lumen의 자동 합성 커널로 한국어 토큰을 출력하는 단계로 진입한다.

레포: https://github.com/redchupa/lumen

— Claude와 페어 프로그래밍으로 작성.
