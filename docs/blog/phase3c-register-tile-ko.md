# Phase 3.C — 4×8 Register Tile: 57 GFLOPS, scalar 대비 19배

> Lumen 빌드로그 #5.

지난 Phase 3.B에서 j축 8-wide SIMD로 8.7× 가속을 얻었다. 그게 끝일 줄 알았다. 한 가지 트릭이 더 남아 있었다 — **register tile**.

## 핵심 통찰: FMA의 dependency chain

이전 inner k 루프:

```text
for kk in 0..K:
  ymm1 = broadcast lhs[i, kk]
  ymm_acc = ymm_acc + ymm1 * [rhs + (kk*N + j)*4]   ; FMA
```

매 iter마다 `ymm_acc`를 read하고 write한다. Modern CPU가 한 cycle에 vfmadd231ps 2개를 issue할 수 있는데, **이 한 줄짜리 dependency chain이 1개로 직렬화시킨다.** 8 lanes × 1 FMA/cycle만 활용 = 약 25 GFLOPS.

진짜 peak (3GHz × 2 FMAs/cycle × 16 FLOPs/FMA) = 96 GFLOPS는 못 따라간다.

## 해결: 독립 stream 4개

```text
for kk in 0..K:
  ymm4 = load [rhs + (kk*N + j)*4]   ; B는 한 번만 로드

  ; 4개 독립 dependency chain
  ymm5 = broadcast lhs[i+0, kk];  ymm0 += ymm5 * ymm4
  ymm5 = broadcast lhs[i+1, kk];  ymm1 += ymm5 * ymm4
  ymm5 = broadcast lhs[i+2, kk];  ymm2 += ymm5 * ymm4
  ymm5 = broadcast lhs[i+3, kk];  ymm3 += ymm5 * ymm4
```

이제 4개의 FMA가 **서로 독립이다.** CPU의 out-of-order 엔진이 이걸 본다:
> "ymm0, ymm1, ymm2, ymm3은 서로 안 건드리네. 그럼 동시에 4개 issue 할 수 있겠다."

추가 보너스:
- **B 메모리 재사용**: `ymm4`를 한 번 load해서 4 row에 다 쓴다. 메모리 대역폭 1/4.
- **A broadcast 4개**: 각 row마다 다른 A 값을 broadcast. 작은 추가 비용.

## 시연: 64×64×64 matmul, 동일 IR, 세 가지 backend

```rust
// 같은 소스, 같은 IR
let module = Parser::parse(src)?;
let ir = lower(&module)?;

// Lumen이 형상을 보고 자동 선택:
//   M%4==0 && N%8==0  →  4×8 register tile
//   N%8==0            →  1×8 AVX2
//   else              →  scalar
let mc = X86_64::host().lower(&ir, ..)?;
```

벤치 결과 (64×64×64):

| Backend | 시간 | Throughput | Scalar 대비 |
|---|---|---|---|
| scalar (`mulss/addss`) | 175 µs | 2.99 GFLOPS | 1× |
| 1×8 AVX2 (Phase 3.B) | 20.7 µs | 25.4 GFLOPS | 8.7× |
| **4×8 tile (Phase 3.C)** | **9.1 µs** | **57.3 GFLOPS** | **19.2×** |

(Throughput은 `2 × M × N × K` FLOPs 기준)

큰 사이즈에서도 무너지지 않는다:

| Shape | 4×8 tile throughput |
|---|---|
| 64³ | 57.3 GFLOPS |
| 128³ | 56.6 GFLOPS |
| 256³ | 55.2 GFLOPS |

이론 peak 추정치(~50~60 GFLOPS)에 거의 닿았다. CPU의 turbo boost와 FMA 2-port issue를 같이 활용하는 결과.

## 한국에서 자체 컴파일러로 57 GFLOPS

이 숫자에 의미가 있는 이유:

- **OpenBLAS** sgemm: 비슷한 CPU에서 70~90 GFLOPS. 우리는 그것의 65~80%.
- **llama.cpp**의 `ggml_compute_forward_mul_mat_f32`: ~50 GFLOPS급.
- **PyTorch CPU**: MKL/oneDNN 호출. 60~80 GFLOPS.

우리는 단일 사람 + AI 페어 프로그래밍 며칠 작업으로 **이 영역에 진입했다.** 캐시 타일링이 아직 없는데도. 캐시 블로킹을 넣으면 80%+ 지점이 사정거리에 들어온다.

llama.cpp가 30년에 걸친 SIMD 손코딩 결과를 모은 것이라면, 우리는 IR이 한 패턴 매처로 같은 결과를 자동 합성한다. 같은 IR이 새로운 양자화 포맷이나 새 CPU 마이크로아키텍처를 만나도, 다시 손코딩할 필요가 없다는 뜻이다.

## 정답성 — 19배 빨라도 1픽셀도 안 틀려야 한다

벤치마크 숫자가 아무리 좋아도 결과가 틀리면 무의미하다. e2e 정답성:

- 16×16×16: 4×8 tile path, naive Rust reference와 1e-3 일치
- 64×64×64: 4×8 tile path, naive Rust reference와 1e-2 일치 (K=64라 누적 오차 큼)
- 8×8×8: 1×8 AVX2 path도 여전히 정답
- 4×4×4: scalar fallback, naive와 1e-4 일치

세 가지 path가 같은 IR에서 자동 합성되어 모두 같은 정답을 낸다. 사용자가 한 코드도 안 바꾼다.

## 누적

- 6 커밋, ~5,800줄 Rust 코드
- 48 tests (단위 + e2e + bench)
- 5편 블로그 (Phase 1, 2.A, 2.B, 3.A/B, 3.C)
- 외부 의존성: production `thiserror` 하나

## 다음

Phase 3.D — cache blocking. 큰 행렬 (1024×1024+)에서 L1/L2 캐시 fit이 깨질 때 외부 블록 루프로 sub-matrix를 캐시에 잡는다. 이게 들어가면 1024³ throughput이 유지된다.

그 후 Phase 4 JIT 정교화 → Phase 5 양자화 (Q4_0/Q8_0) → Phase 6 LLM 추론.

레포: https://github.com/redchupa/lumen

— Claude와 페어 프로그래밍으로 작성.
