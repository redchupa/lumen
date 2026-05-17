# Lumen

> **IR이 양자화 커널을 자동 합성하는** LLM 추론 컴파일러 + 런타임.
> 한국어 LLM(EXAONE, HyperCLOVA-X, A.X) 추론도 1급으로 지원.

[![Build](https://img.shields.io/badge/build-passing-green)](#) [![License](https://img.shields.io/badge/license-Apache--2.0-blue)](#) [![Rust](https://img.shields.io/badge/rust-1.78%2B-orange)](#) [![Version](https://img.shields.io/badge/version-v0.3.0-brightgreen)](#)

---

## What is this?

Lumen은 **LLM 추론을 위한 컴파일러 + 런타임**입니다.
PyTorch나 ONNX Runtime처럼 기성 그래프 컴파일러를 갖다 쓰는 것이 아니라,

- 자체 텐서 **DSL**(언어)
- 자체 **IR**(중간표현, SSA 기반, 텐서 형상이 타입)
- 자체 **코드 생성기**(x86_64 · ARM64 · CUDA)
- 자체 **JIT**(런타임에 입력 형상 보고 특화 커널 생성)

까지 전부 직접 짭니다. 외부 의존: 표준 라이브러리, OS, 하드웨어.

## Why?

**llama.cpp**는 훌륭하지만 — 양자화 × dtype × 형상 조합별로 사람이 손으로 커널을 짭니다 (`ggml_vec_dot_q4_0_q8_0`, `ggml_vec_dot_q4_K_q8_K`, ...). 수백 개의 함수가 손코딩.

**Lumen**은 IR 레벨에서 `tensor<q4_0, ...> @ tensor<f16, ...>`을 보면 **unpack + dequantize + matmul + requantize를 자동 융합**해 한 덩어리 커널을 emit합니다. 새 양자화 포맷이 등장해도 IR 변경만으로 모든 백엔드에 전파됩니다.

**한국어 LLM**은 토크나이저 효율과 RoPE 변형에서 외산 런타임이 자주 헛돈을 둡니다. Lumen은 한국어 모델을 정답성 테스트 케이스의 1급 시민으로 둡니다.

## Roadmap (요약)

| Phase | 산출물 | 상태 |
|---|---|---|
| 0. 설계 | 아키텍처 문서, 워크스페이스 | ✅ 완료 |
| 1. DSL · 파서 | Pratt 파서, AST, 타입 검사, 진단 | ✅ 완료 (21 tests) |
| 2.A. IR + C backend | SSA IR, lower, verify, print, C emit, e2e | ✅ 완료 |
| 2.B. 자체 x86_64 backend | 머신코드 emit + JIT 실행 + 정답성 e2e (40 tests) | ✅ 완료 |
| 3.A/3.B AVX2 자동 합성 | VEX 인코더 + matmul AVX2 path, 8.7× scalar 대비 | ✅ 완료 |
| 3.C. Register tile 4×8 | 4 독립 accumulator, 57 GFLOPS / 19× scalar | ✅ 완료 |
| 5.A. 양자화 reference | Q4_0/Q8_0 dequant in pure Rust | ✅ 완료 |
| 5.B. Native Q8 dequant | F16C+AVX2, IR Op::Dequantize 자동 합성 | ✅ 완료 |
| 5.C. Q8 × F32 fused matmul | dequant×matmul 패턴 자동 융합 | ✅ 완료 |
| 5.D. GGUF v3 reader | header + KV + tensor table + 라운드트립 | ✅ 완료 |
| 5.E. GGUF → native | 디스크 파일 → JIT native dequant 라운드트립 | ✅ 완료 |
| 6.A~D. Tokenizer + Transformer | BPE + Llama op + layer forward + KV cache | ✅ 완료 |
| 6.E. Generate loop | Model + KV cache + autoregressive decode | ✅ 완료 |
| 6.F. **Real Qwen2.5-0.5B 한국어 추론** | "안녕" → "안녕하세요, 저는" (~0.54s/token) | ✅ **완료** |
| 6.G. JIT 통합 (lm_head → 전체 forward) | shape-keyed `MatmulJitCache`, 동일 토큰 보장 | ✅ **완료** |
| 7.A. **vs llama.cpp 벤치** | tg32 단일 스레드: naive 2.91 / JIT 4.43 / ggml 41.32 tok/s | ✅ **완료** |
| 7.C. 1×N 4-acc decode kernel | M=1 N%32 FMA 종속성 체인 해체, 4.43→5.08 tok/s | ✅ 완료 |
| 7.D. **Q8-native fused matmul (model path)** | dequant 패스 제거 + 메모리 대역 4× 회복, 5.08→17.97 tok/s | ✅ **완료** |
| 2.C. ARM64 backend | AAPCS64, NEON-readiness | ⏳ |
| 3.D. 캐시 타일링 (prefill) | Mc/Kc 블로킹, prefill batch 모드 | ⏳ |
| 7.E. flash-style attention | 긴 컨텍스트 필수 | ⏳ |
| 7.F. multi-thread prefill | physical core 활용 | ⏳ |

상세 계획: [PLAN.md](./PLAN.md) · 아키텍처: [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) · 벤치: [BENCHMARK.md](./BENCHMARK.md)

## 현재 성능 (main, Qwen2.5-0.5B Q8_0)

**단일 스레드 (`-t 1` ggml):**

| 경로 | tg32 tok/s | vs ggml |
|---|---:|---:|
| Lumen naive Rust | 2.91 | 14.2× slower |
| Lumen JIT v0.1.0 | 4.43 | 9.3× slower |
| Lumen JIT v0.2.0 (Q8-native) | 17.97 | 2.30× slower |
| Lumen JIT (Q8 4-acc + SiLU AVX2 + RoPE precomp, 7.G-I) | ~31 | 1.32× slower |
| llama.cpp 1-thread | 41.32 | 1.0× |

**멀티 스레드 (8 threads):**

| 경로 | tg32 tok/s | vs ggml 8t |
|---|---:|---:|
| **Lumen JIT main (Phase 7.J — rayon Q8 matmul)** | **~56** | **1.62× slower** |
| llama.cpp 8-thread | 90.90 | 1.0× |

v0.1.0 → main: **~12.6× decode** (4.43 → 56 tok/s). 토큰은 naive ↔ JIT bit-identical. 자세한 해부는 [BENCHMARK.md](./BENCHMARK.md).

## Quick start

```sh
git clone https://github.com/redchupa/lumen
cd lumen
cargo build --workspace

# Parse a Lumen source file and dump its AST.
cargo run -p lumen-cli -- parse examples/matmul.lum

# Type-check it.
cargo run -p lumen-cli -- check examples/matmul.lum
# ok: examples/matmul.lum type-checked
```

`examples/matmul.lum`:

```rust
fn matmul(
    a: tensor<f32, [64, 128]>,
    b: tensor<f32, [128, 32]>,
) -> tensor<f32, [64, 32]> {
    return a @ b;
}
```

The type checker enforces `a.shape[1] == b.shape[0]` and infers the result
shape `[a.shape[0], b.shape[1]]` at compile time. Try changing `128` to `127`
in either tensor — you get a typed error pointing at the exact source span.

## Non-goals

- 학습(training) 지원 — 추론 전용
- 그래프 시각화/디버거 — 별도 도구로 분리
- 100개 모델 지원 — 한국어 모델 6종 + Qwen 계열만

## License

Apache-2.0
