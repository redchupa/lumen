# Lumen

> **IR이 양자화 커널을 자동 합성하는** LLM 추론 컴파일러 + 런타임.
> 한국어 LLM(EXAONE, HyperCLOVA-X, A.X) 추론도 1급으로 지원.

[![Build](https://img.shields.io/badge/build-WIP-yellow)](#) [![License](https://img.shields.io/badge/license-Apache--2.0-blue)](#) [![Rust](https://img.shields.io/badge/rust-1.78%2B-orange)](#)

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
| 5.A. 양자화 reference | Q4_0/Q8_0 dequant in pure Rust (5 tests) | ✅ 완료 |
| 5.B. Native Q8 dequant | F16C+AVX2, IR Op::Dequantize 자동 합성 | ✅ 완료 |
| 5.C. **Q8 × F32 fused matmul** | dequant×matmul 패턴 자동 융합, 65 tests | ✅ 완료 |
| 2.C. ARM64 backend | AAPCS64, NEON-readiness | ⏳ |
| 3.D. 캐시 타일링 | 블록 매크로커널, 256³+ 큰 사이즈 유지 | ⏳ |
| 4. JIT 엔진 | 런타임 컴파일 | ⏳ |
| 5. 양자화 | INT8/INT4, GGUF | ⏳ |
| 6. LLM 추론 | 토크나이저, KV, sampling | ⏳ |
| 7. 벤치 · 블로그 | vs llama.cpp | ⏳ |

상세 계획: [PLAN.md](./PLAN.md) · 아키텍처: [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)

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
