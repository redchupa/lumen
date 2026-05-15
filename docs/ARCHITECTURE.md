# Lumen — 아키텍처

## 1. 전체 그림

```
   *.lum 파일                  GGUF 모델
       │                          │
       ▼                          ▼
  ┌─────────┐               ┌──────────┐
  │ lumen-  │               │ lumen-   │
  │  dsl    │               │  model   │
  │ (Lex,   │               │ (Loader, │
  │  Parse, │               │  Token)  │
  │  Type)  │               └────┬─────┘
  └────┬────┘                    │
       │ AST                     │
       ▼                         │
  ┌─────────┐                    │
  │ lumen-  │                    │
  │  ir     │ ← passes ←─────────┤
  │ (SSA)   │                    │
  └────┬────┘                    │
       │ IR                      │
       ▼                         │
  ┌──────────┐                   │
  │ lumen-   │                   │
  │ codegen  │                   │
  │ x86/ARM/ │                   │
  │  CUDA    │                   │
  └────┬─────┘                   │
       │ Machine code            │
       ▼                         │
  ┌─────────┐                    │
  │ lumen-  │ ◄──────────────────┘
  │  jit    │   ← runtime shape
  └────┬────┘                       feedback
       │
       ▼
  ┌─────────┐
  │ lumen-  │
  │ runtime │
  └─────────┘
```

## 2. crate 책임 분할

| Crate | 입력 | 출력 | 의존 |
|---|---|---|---|
| `lumen-dsl` | `*.lum` 텍스트 | 타입 검사 끝난 AST | 표준 lib |
| `lumen-ir` | AST | SSA IR + 패스 | `lumen-dsl` |
| `lumen-codegen` | IR + 타겟 | 머신 코드 바이트 | `lumen-ir` |
| `lumen-jit` | IR + 입력 형상 | 실행 가능한 함수 포인터 | `lumen-codegen` |
| `lumen-runtime` | 텐서 + 함수 포인터 | 결과 텐서 | `lumen-jit` |
| `lumen-model` | GGUF 파일 | 가중치 텐서 + 토크나이저 | `lumen-runtime` |
| `lumen-cli` | CLI 인자 | stdout | 위 전부 |

## 3. 핵심 설계 결정

### 3.1 IR은 SSA, 텐서 형상이 타입의 일부

```
%0: tensor<f16, [1, 64, 4096]> = load_weight @"blk.0.attn_q.weight"
%1: tensor<f16, [1, 64, 4096]> = matmul %hidden, %0
```

- 형상이 IR 타입에 박혀 있으면 코드 생성기가 정적 최적화를 매우 강하게 할 수 있음
- 동적 형상은 별도 `tensor<f16, [?, 64, 4096]>` 표기. JIT 단계에서 구체화됨

### 3.2 양자화는 IR의 1급 시민 (Lumen의 정체성)

- `tensor<q4_0, [4096, 4096]>` 같은 양자화 타입이 IR 레벨에 존재
- 코드 생성기가 양자화 타입을 보면 `unpack + gemm + (optional)requantize` 융합 커널을 자동 생성
- **llama.cpp 비교**: `ggml_vec_dot_q4_0_q8_0`, `ggml_vec_dot_q4_K_q8_K`, `ggml_vec_dot_q5_0_q8_0`, ... 수십~수백 함수가 손코딩. Lumen은 한 패턴 매칭이 모든 조합을 처리.
- **확장성**: 새 양자화 포맷 추가 시 IR `DType` enum + unpack 패턴 한 곳만 추가하면 모든 백엔드(x86/ARM/GPU)에 전파.

### 3.3 JIT은 "1차 컴파일은 AOT, 형상별 특화만 JIT"

- DSL → IR → 일반 코드 변환은 빌드 시점에 끝남
- 런타임에는 입력 형상 보고 **타일 크기, 언롤 횟수, 벡터 폭** 같은 파라미터만 채워서 코드 emit
- 첫 호출 < 50ms 컴파일 목표

### 3.4 백엔드 트레이트

```rust
trait Backend {
    fn name(&self) -> &str;
    fn capabilities(&self) -> Capabilities;
    fn lower(&self, ir: &IrModule, opts: &CodegenOpts) -> Result<MachineCode>;
}
```

x86_64, ARM64, CUDA 모두 이 트레이트 구현. CLI는 런타임에 적합한 백엔드 선택.

### 3.5 메모리 모델

- 텐서 = `Arc<TensorBuffer>` + 메타데이터(`shape`, `dtype`, `stride`)
- 버퍼 풀: 사이즈 클래스별 freelist. 매 토큰마다 할당 안 함.

**KV 캐시** (Phase 6, paged-attention 영감):
- 1 layer × kv_heads × head_dim × dtype = "page" 단위
- 시퀀스마다 page 리스트만 보유. 연속 메모리 강제 X
- 새 토큰 → 마지막 page에 append. 가득 차면 새 page allocate.
- 7B 모델 / 32 layers / 32 KV heads / 128 head_dim / fp16 / 4096 ctx 가정
  → page 크기 32KB, 시퀀스당 ~4096개 page, 합 ~1GB (사전 할당 안 함, 토큰 진행에 따라 증가)
- v1.1 continuous batching 도입 시 page 풀이 시퀀스 간 공유됨

### 3.6 비동기 / 병렬

- 단일 요청 추론은 동기. async 안 씀.
- 배치 추론에서 토큰별 병렬은 `rayon` 사용 검토 (Phase 6 이후 결정)
- 마이크로커널은 멀티스레드 X (간단함 우선)

## 4. 비결정 사항 (RFC 필요)

- [ ] CUDA 코드 생성: 자체 PTX vs cuBLAS fallback (v1.0에서 무엇?)
- [ ] 토크나이저: HuggingFace tokenizers crate 의존 vs 자체 구현
- [ ] 양자화 포맷: GGUF만 vs 자체 포맷도 정의
- [ ] DSL 문법: 함수형(Triton 스타일) vs 명령형(C 스타일)

각 결정은 `docs/rfc/NNN-제목.md` 형식의 RFC 작성 후 머지.

## 5. 보안 / 안전성

- `unsafe` 사용은 다음 모듈에서만 허용:
  - `lumen-jit/src/exec.rs` — mmap, mprotect
  - `lumen-runtime/src/simd.rs` — SIMD intrinsic
  - `lumen-codegen/src/emit.rs` — 바이트 버퍼 write
- 그 외 모든 모듈은 `#![forbid(unsafe_code)]`

## 6. 다이어그램 외 — 데이터 흐름 예시

`lumen run model.gguf "안녕"` 실행 시:

1. `lumen-cli` → 명령 파싱
2. `lumen-model` → `model.gguf` 로드, `tokens = [128, 4521]` 생성
3. `lumen-runtime` → 모델 그래프(이미 컴파일된 IR) 인스턴스화
4. `lumen-jit` → 입력 형상 `[1, 2]` 보고 prefill 커널 생성/캐시 조회
5. `lumen-runtime` → forward 실행, logits 반환
6. 샘플링 → 다음 토큰
7. KV 캐시 업데이트, 4~6 반복
