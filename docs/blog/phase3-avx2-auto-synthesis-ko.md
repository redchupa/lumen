# Phase 3.A/3.B — 자동 합성이 처음으로 의미를 갖는 날

> Lumen 빌드로그 #4.

오늘 Lumen은 두 번째 백엔드 경로를 가졌다. 같은 IR이 들어가서, 같은 정답이 나오는데, **8.7배 빠른** 머신코드가 emit 된다.

```text
matmul %0, %1 : tensor<f32, [64, 64]>
```

이 한 줄을 보고 컴파일러가:

| 조건 | 선택 | 결과 |
|---|---|---|
| `N % 8 != 0` | 스칼라: `mulss/addss` 1쌍 | 정확하지만 느림 |
| `N % 8 == 0` | AVX2: `vbroadcastss + vfmadd231ps` | **8개 column 동시 처리** |

선택은 자동이다. 사용자가 `--avx2` 플래그를 켜지 않았다. 코드도 안 바꿨다. **컴파일러가 형상을 보고 결정했다.**

이게 Lumen의 정체성이 처음으로 살아 움직인 순간이다. llama.cpp가 `ggml_vec_dot_q4_0_q8_0` 같은 함수를 손으로 짜는 동안, 우리는 IR 패턴 매처가 같은 일을 하게 만든다.

## VEX prefix — 30년 만에 다시 짠 인코딩

Phase 2.B의 REX prefix가 8086에서 64비트로 가는 다리였다면, VEX는 x86이 SIMD 세계로 들어가는 다리다.

```text
2-byte VEX: C5 [R̄ vvvv L pp]
3-byte VEX: C4 [R̄ X̄ B̄ mmmmm] [W vvvv L pp]
```

각 비트의 의미가 REX보다 많다 — `mmmmm`은 opcode map (0F vs 0F 38 vs 0F 3A), `pp`는 SSE prefix를 인코딩하는 2비트, `vvvv`는 추가 source register (4비트). 그리고 거의 모든 비트가 **반전돼서 저장된다.**

`vxorps ymm0, ymm0, ymm0`을 emit하면 `C5 FC 57 C0` 4바이트가 나와야 한다. 사람이 손으로 풀면:

```
C5     : 2-byte VEX (R=0, X=0, B=0, W=0, map=0F → 2-byte form 가능)
FC     : R̄=1, vvvv=1111 (ymm0의 반전), L=1 (256bit), pp=00 (no prefix) → 1111_1100
57     : opcode XORPS
C0     : ModR/M (mod=11, reg=000=ymm0, rm=000=ymm0)
```

이 4바이트를 단위 테스트로 ground truth와 한 글자씩 맞춰봤다. 한 비트라도 어긋나면 CPU가 illegal instruction 예외로 죽는다. 디버거가 알려주지 않는다 — 그냥 죽는다.

5개 인코더 단위 테스트 모두 ground truth와 일치. 그 다음 단계로 갔다.

## 자동 합성의 첫 시연 — 같은 IR, 두 가지 머신코드

```rust
// 둘 다 같은 IR
let module = Parser::parse(src)?;
let ir = lower(&module)?;

// 호스트 자동 선택 (N=8 → AVX2)
let mc_avx2 = X86_64::host().lower(&ir, &opts)?;

// 강제로 스칼라
let mc_scalar = X86_64::scalar_only().lower(&ir, &opts)?;
```

두 머신코드는 길이부터 다르다. 스칼라는 inner k-loop 안에 `mulss/addss` 명령어가 작은 단위로 들어가고, AVX2는 `vbroadcastss/vfmadd231ps` 한 묶음만 들어간다.

두 코드를 같은 입력으로 호출하면 같은 정답이 나온다. 32×32×32 matmul, 8×8×8 matmul, 4×5×3 rectangular — 어느 케이스든 둘 다 naive Rust reference와 일치한다.

## FMA의 마법 — `acc = acc + a * b`가 한 명령어

```text
vfmadd231ps ymm0, ymm1, [r13 + rdx*4]   ; ymm0 += ymm1 * memory
```

이 한 명령어가 8개 floats에 대해 곱하기 + 더하기를 한 번에 한다. **2 FLOP × 8 lane = 16 FLOPs per instruction.**

Haswell 이후 CPU는 매 사이클에 vfmadd231ps를 1~2개 issue할 수 있다. 3GHz × 16 FLOPs = 48 GFLOPS 이론 peak.

우리 실측 25 GFLOPS — 약 **50% peak**. 캐시 타일링 없이, 매크로커널 없이, 그냥 j축만 8-wide vectorize 한 결과. Phase 3.C에서 캐시 타일링을 넣으면 70%+ 까지 가능할 것이다.

## 벤치 결과

| Shape | 스칼라 | AVX2 | 가속비 |
|---|---|---|---|
| 64×64×64 | 179 µs (2.9 G op/s) | **20.7 µs (25.4 G op/s)** | **8.7×** |
| 128×128×128 | 1.39 ms (3.0 G op/s) | **201 µs (20.8 G op/s)** | **6.9×** |

64×64는 L1/L2 캐시에 잘 들어가서 거의 이론값을 따라간다 (8 lanes = 8× 이론, 실측 8.7× 는 FMA가 더해진 효과).

128×128은 L1을 넘어서기 시작 — 메모리 bandwidth가 병목이 되기 직전. 캐시 타일링이 들어오면 여기서 큰 개선이 있다.

## 의존성 누락 정책 유지

이번 작업에 추가한 외부 의존성:
- **dev-dep**: `criterion = "0.5"` (벤치마크 전용)

빌드 산출물(`lumen` 바이너리)에 들어가는 의존성은 **여전히 `thiserror` 단 하나**. CLAUDE.md의 의존성 정책이 깨지지 않는다. 자체 인코더, 자체 백엔드, 자체 JIT 페이지 관리 — 다 우리가 짰다.

## 숫자

- 0.5일 작업
- Rust 코드 +700줄 (avx2_enc 320, x86_64.rs AVX2 path 130, e2e 150, bench 80)
- 새 테스트 +6 (AVX2 encoder 5, AVX2 e2e 1 추가; 기존 e2e 1개 확장)
- **총 46 tests green**
- 새 벤치 4개 (스칼라 64/128, AVX2 64/128)

## 한계 — 그리고 Phase 3.C

지금 AVX2 path는 j축만 vectorize한다. M축(행) 방향은 여전히 스칼라 루프. 이 때문에 큰 행렬에서는 LHS broadcast가 캐시 미스를 자주 일으킨다.

다음 단계:
- **Register tile**: 한 inner 루프에서 4×8 (행×열) tile을 한꺼번에 계산 → ymm 8개 사용
- **Cache blocking**: 외부에 매크로커널 블록 (예: 64×64 sub-matrix) → L1 잡기
- **Loop unrolling**: k축 언롤로 instruction-level parallelism

목표: ggml의 80~95% throughput. Phase 3.C에서.

레포: https://github.com/redchupa/lumen — Phase 누적 4편 블로그. 별 부탁드립니다.

— Claude와 페어 프로그래밍으로 작성.
