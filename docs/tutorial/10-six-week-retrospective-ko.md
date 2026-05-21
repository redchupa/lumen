# 10장. 6주 회고 — v0.1에서 v0.5까지

> 마지막 장입니다. 이 책 전반에서 추상적으로 풀어둔 측정-주도 패턴이 실제 6주
> 사이드 프로젝트 안에서 어떻게 흘렀는지를 시간 순으로 정리합니다. v0.1.0에서
> v0.5.0까지 어떤 결정을 했고, 어떤 측정을 했고, 무엇이 부정됐는지의 일기.
> 추상적 개념이 구체적 일기를 만나면 "내 사이드 프로젝트도 같은 패턴으로 가능"이
> 보일 거예요.

## 10.1 출발점 — 0편 commit

6주 전 첫 commit은 단순했어요. Rust 프로젝트 골격, 빈 `Cargo.toml`, 그리고 한
줄짜리 README. "LLM 추론 컴파일러 만들겠다"는 의도만 적힘.

이때 시점의 결정 사항:

- **목표**: 한국어 LLM(Qwen2.5-0.5B)을 처음부터 짠 컴파일러로 추론
- **언어**: Rust (메모리 안전성, 성능, 개인 학습 욕구)
- **백엔드**: x86_64 우선, ARM64는 나중에
- **양자화**: Q8_0부터 (가장 흔한 포맷)
- **비교 대상**: llama.cpp (ggml). 같은 모델로 같은 측정으로 비교
- **호흡**: 1주에 commit 5~10개. 매일 1~2시간

이 결정들이 6주 끝까지 거의 안 바뀌었어요. 첫 결정이 단단해야 사이드 프로젝트가
지속됩니다.

## 10.2 v0.1.0 (1~2주차) — 첫 토큰

첫 2주는 인프라 짜는 시간이었어요. 결과물은 없고 코드 양만 늘어요. 이 시기 흐지부지
되기 쉬운데, 작은 milestone을 정해두는 게 도움이 됐어요.

**Phase 1 — DSL · 파서**: `examples/matmul.lum` 파일이 파싱되고 타입 체크 통과.
21개 단위 테스트.

```
fn matmul(a: tensor<f32, [4, 8]>, b: tensor<f32, [8, 2]>) -> tensor<f32, [4, 2]> {
    return a @ b;
}
```

이 한 줄이 처음 컴파일러를 통과한 순간 진짜 작은 감동. 코드는 단순한데 그 뒤에
있는 lexer/parser/type-checker가 다 작동하는 시그널.

**Phase 2.A — IR + C backend**: IR을 만들고 그걸 C 코드로 출력. naive matmul이 정답
인지 검증 (reference oracle). C로 컴파일해서 결과 비교.

**Phase 2.B — 자체 x86_64 backend**: 직접 머신 코드 emit. 매트멀 결과가 naive
Rust와 비트 단위 동일. 첫 JIT 실행이 여기서 일어남.

**Phase 3 — AVX2 자동 합성**: SIMD 명령 인코더, AVX2 매트멀 path. 스칼라 대비 8.7배
가속(작은 매트멀 micro-bench).

**Phase 3.C — Register tile 4×8**: 4-acc 패턴 + 4×8 tile. 스칼라 대비 19배 가속.
57 GFLOPS 도달.

**Phase 5.C — Fused quant matmul**: Q8 weight + F32 activation 매트멀 IR로 자동
합성. 이 시점에서 컴파일러 인프라가 거의 갖춰짐.

**Phase 6 — Qwen2.5-0.5B 한국어 추론**: "안녕" 입력 → "안녕하세요, 저는" 출력.
첫 진짜 LLM 답. 1주차에 짜기 시작한 코드가 6주차에 ChatGPT 비슷한 거 동작.

여기까지 v0.1.0. **단일 스레드 4.43 tok/s**. ggml 41 tok/s의 9.3배 느림.

처음 봤을 땐 9.3배 격차가 막막했어요. 그런데 v0.1.0 = "동작은 한다"가 목표였고
그건 달성했죠. 가속은 다음 cycle 일.

### 1~2주차 배움

- 첫 commit은 골격만. 처음부터 완벽하게 짜려 하지 말기.
- 작은 milestone 자주 정하기. "이번 주 안에 matmul 한 줄 파싱 통과" 같은 거.
- naive 구현부터. 처음부터 빠르게 짜려 하면 디버깅 지옥. naive로 정답 만들고
  나중에 가속.

## 10.3 v0.2.0~v0.3.0 (3~4주차) — Q8 native, 멀티스레드

이 시기가 가장 큰 perf 가속이 일어난 시기예요.

**Phase 7.D — Q8 native fused matmul**: Q8 weight를 dequantize하지 않고 그 자리에서
매트멀. Phase 5.C의 IR 자동 합성을 model path까지 연결. **4.43 → 17.97 tok/s
(+4배 가속)**.

이건 직관이 맞은 케이스. "dequantize 단계 없애면 메모리 부담 4배 줄어 빠를 것".
측정으로 확정. 이때 가설-측정 사이클이 제대로 작동한 첫 사례였어요.

**Phase 7.G — Q8 N=1 4-acc 커널**: 5장에서 본 4-accumulator 패턴. block당 4 개
독립 acc. **17.97 → 29.10 tok/s (약 2배 가속)**.

또 직관이 맞은 케이스. FMA dependency chain 끊으면 빠를 것 → 측정으로 확정.

**Phase 7.J — Multi-thread (rayon)**: 멀티스레딩 첫 시도. rayon 라이브러리 활용.
**29 → 60 tok/s @ 8 thread**. ggml과의 8t 격차가 약 1.5배까지 좁혀짐.

이 시기 약 2주 동안 v0.1 대비 13배 가속. 매주 win이 보이는 시기. 동기 부여 가장
높을 때.

### 3~4주차 배움

- 큰 가속은 보통 명확한 직관에서 옴. Q8 native, 4-acc, 멀티스레딩. 다 책의 상식.
- 직관 win 사이클은 짧음. 이 시기는 빨리 다음 단계로.
- naive 구현 만들어 둔 게 도움 됨. 매 가속 시도 후 naive와 결과 비트 단위 비교
  로 정답성 확인.

## 10.4 v0.3.0~v0.4.0 (5주차) — VNNI, shape-aware

이 시기에 첫 진짜 부정 사례들이 나옵니다.

**Phase 7.M — Q8 int dot**: 정수 dot product 가속 시도. 측정 결과 거의 변화 없음
(net-neutral). 인프라는 유지.

**Phase 7.N — VNNI single-acc**: VNNI 명령어(vpdpbusd) 활용. 단일 acc로 시도. -3.6%
회귀. 폐기.

**Phase 7.O — VNNI 4-acc**: VNNI에 4-acc 결합. 그래도 -2.7% 회귀. 다만 인프라 유지.

**Phase 7.P — Shape-aware dispatch**: 매트멀 형상별로 다른 커널 선택. 짧은 K는 VNNI,
긴 K는 fp32 4-acc. **여기서 7.O의 인프라가 살아남**. 짧은 K에서 VNNI win이 활성
화되니까. **60 → 65 tok/s @ 8t**.

이 사이클이 정확히 "default-off 인프라가 부활하는 패턴"이에요. 7.O 부정 후 그
코드 폐기했으면 7.P가 불가능했을 거. 코드 살린 결정이 미래 win의 근거가 됨.

v0.4.0 시점 standing:
- 단일 스레드: 41.85 tok/s (ggml 40.40보다 +4% 빠름)
- 8 thread: 65.5 tok/s (ggml 90.9의 약 1.39배 느림)

처음으로 단일 스레드에서 ggml과 비등해진 시점이에요. 8 thread는 여전히 격차.

### 5주차 배움

- 부정 사례가 처음 누적되기 시작. 자기 의심 시작되는 시기.
- 인프라 살리기의 진짜 가치를 처음 느낌. 7.O → 7.P 부활 패턴.
- "1주에 win 없어도 OK"라는 마음 가짐 필요. 인프라 누적이 진척.

## 10.5 v0.4.0~v0.5.0 (6주차) — 격차 진단과 prefill

이 시기가 측정-주도 패턴의 핵심 사이클이었어요.

**Phase 7.U — 격차 분석**: 1.39배 8t 격차의 원인 진단. step-별 시간 측정, thread
별 측정, memory bandwidth 측정. 결론: 메모리 활용도 차이(Lumen 55%, ggml 74%)가
격차의 정체. **여기서 의외의 발견: 단일 스레드에서 Lumen이 ggml보다 +13% 빠름.**

이게 v0.5 cycle의 가장 큰 발견이었어요. 격차 원인이 명확해진 시점.

**Phase 8.A — Atomic ThreadPool**: mutex 기반 → atomic counter 기반 재설계.
**1 thread +9%, 8 thread +3%**. 8 thread는 noise 영역이지만 1~2 thread에서 명확한
가속. merge.

**Phase 8.B — chunk L2-fit cap**: cache fit 시도. -3.5% 회귀. revert.

**Phase 8.C — Software prefetch**: prefetch 명령 추가. 1 thread 약간 win, 8 thread
**-49% 회귀**. revert. 인프라(prefetch encoders)는 유지.

**Phase 8.D — Prefill batching**: prefill 인프라 1주일 작업. 정답성 통과, 단위
테스트 다 통과. 측정 결과 **pp32 = 22.65 tok/s, tg32 65의 token당 2.9배 회귀**.
충격적 결과.

**Phase 8.D.3 commit message**: "attention이 원인일 것"이라 추측 적음. 측정 안 한
추측이었음.

**Phase 8.D.4 (다음 날)**: attention만 단독으로 측정. 24 layers × 389µs = 9.3ms.
pp32 전체의 0.66%만. 가설 부정. commit message에 추측 적은 거 부메랑.

**Phase 8.D.5**: 진짜 원인 찾기. 매트멀 raw kernel만 측정. **N=32 커널이 N=1 ×
32보다 2.5배 느림**. N>1 codegen 비효율 확정.

**Phase 8.E.1 — N=1 fan-out 회복**: 임시 fix. N>1 dispatcher가 N=1 커널을 N번
호출. **pp32 22.65 → 54.15 tok/s (+2.4배 회복)**. tg32까지는 못 따라가지만 회귀
탈출.

여기까지 v0.5.0. 이 시기에 부정 6번, win 2번, 진단 1번, 회복 1번. 1주에 9번 사이클.

### 6주차 배움

- 1주일 1주제(prefill batching) 시도하면 큰 가설이 부정될 수 있음. 그래도 인프라는
  부활 대기로 남김.
- commit message 추측이 다음 날 부메랑. "측정 안 한 추측 적지 말기" 규칙 강화.
- 50줄 단위 bench가 진짜 강력. 1주일 디버깅을 1시간으로 단축.

## 10.6 6주 결과 정리

### 수치

| 구분 | v0.1.0 | v0.5.0 |
|---|---:|---:|
| 단일 스레드 tg32 | 4.43 tok/s | 45.64 tok/s (10.3배 가속) |
| 8 스레드 tg32 | 미지원 | 67.4 tok/s |
| vs ggml 단일 | 9.3배 느림 | **+13% 더 빠름** |
| vs ggml 8t | 미지원 | 1.30배 느림 |
| 측정-주도 사이클 | 0회 | 11회 누적 |
| 회고 블로그 편수 | 0편 | 17편 |

15배 가속(전 cycle 총합)이 인상적이지만 그것보다 **단일 스레드에서 ggml +13%**이
가장 큰 자산.

### 부정과 win 비율

```
명확한 perf win:  3번 (Phase 7.D, 7.G, 7.P)
직관 win:         1번 (Phase 8.A — 1t/2t)
진단 확정:        1번 (Phase 8.D.5)
부분 회복:        1번 (Phase 8.E.1)
net-neutral:      1번 (Phase 7.M)
부정 (revert):    4번 (Phase 7.N, 8.B, 8.C, partial 7.O)
부정 (인프라 유지): 5번 (Phase 7.R/S/T 합쳐서 1, prefetch infra, ZMM infra, prefill plumbing, false retro)
```

정직히 부정 비율 50% 이상. 이게 사이드 프로젝트 6주 분량의 정직한 비율이에요.

### 자산

- v0.5.0 GitHub Release
- 17편 회고 블로그
- 11회 사이클 측정 데이터
- 정답성 보장 (모든 commit이 naive와 bit-identical)
- ROADMAP에 다음 cycle 명확

## 10.7 무엇을 배웠나

6주 동안 가장 크게 배운 것 5가지.

**1. 직관은 정직히 자주 틀린다.** ZMM, prefetch, batching 등 책의 상식이 마이크로
아키텍처 분야에서는 자주 부정. 측정 없이 갔으면 11번 중 절반 잘못된 방향이었을
거.

**2. 인프라 폐기는 미래 비용.** 7.O의 7.P 부활 패턴이 명확한 증거. default-off로
유지하는 게 진짜 자산. 미래 호스트/조건에서 활성화 가능.

**3. 단일 측정의 의미는 한정적.** thread별, shape별, host별로 다 측정해야 진짜
패턴이 보임. 한 차원만 보면 noise에 속음.

**4. 글로 풀면 메타 패턴 발견.** 17편 회고 블로그 쓰면서 "commit message에 추측
적지 말기" 같은 규칙이 자기 자신에게 보임. 글 안 썼으면 같은 함정 반복했을 거.

**5. 잘 끝낼 줄 알기.** v0.5.0 release tag 박고 maintenance mode 선언. 6주 분량
+ honest engineering 자산이 누적된 상태에서 멈춤. 작년 9개월 흐지부지된 다른
사이드보다 잘 끝남.

## 10.8 이 책을 마치며

10장 + 0장 시리즈로 LLM 추론 컴파일러를 처음부터 짜는 과정을 풀었어요. 결국 이
책에서 다룬 게 무엇인가 정리하면.

**기술적**:
- LLM 추론 = 매트멀 + 메모리 가져오기
- 컴파일러 = DSL → AST → IR → 어셈블리 4단계
- JIT = 실행 중 메모리에 코드 쓰고 호출
- SIMD = 한 명령으로 8개 동시 처리
- 양자화 = fp32를 8비트로 압축 (답 거의 같음)
- Transformer = 매트멀 5개 + attention + 부수 op
- 멀티스레딩 = 메모리 병목으로 8배 안 나옴

**철학적**:
- 측정 없이 직관만으로 가면 절반 잘못됨
- 부정도 자산. default-off로 유지
- 사이드 프로젝트는 끝낼 줄 아는 것도 능력

이 책의 끝에서 새 사이드 프로젝트 시작하시는 분이 첫 commit 다음에 "측정 코드"를
넣게 됐다면, 이 책이 의미 있게 도움 된 거예요. 책 100번 읽는 것보다 직접 한 번
짜보는 것의 가치가 그 한 줄 측정 코드에 들어 있습니다.

## 10.9 10장 핵심 정리

1. **v0.1 → v0.5 = 6주 작업**. 단일 스레드 10.3배 가속. 8 스레드 1.30배 격차.
   가장 큰 자산은 1t에서 ggml +13%.

2. **시기별 패턴 차이**: 1~2주차는 인프라(결과 없음), 3~4주차는 큰 가속(직관 win),
   5~6주차는 부정 사례 누적과 진단.

3. **부정 사례가 절반 이상**. 정직한 비율. 매번 인프라는 유지.

4. **commit message가 미래의 자기를 비웃을 수 있다**. 측정 안 한 추측 적지 말기.

5. **잘 끝낼 줄 알기**. release tag + maintenance mode. 9개월 흐지부지보다 6주
   결말 있는 게 더 자산.

## 시리즈 종료

여기서 본문 10편은 끝입니다. 이 책 전체 인덱스와 짧은 요약은 [0장 — 시리즈
인덱스](./00-index-ko.md)에서 확인할 수 있어요. 0장은 마지막에 정리한 책의 입구
역할이라 처음 책을 펴는 분에게는 0장부터 보시는 게 도움이 됩니다.

GitHub 코드: [github.com/redchupa/lumen](https://github.com/redchupa/lumen)
회고 블로그 시리즈 (이 책과 별도): [docs/blog/INDEX.md](../blog/INDEX.md)
ROADMAP (다음 cycle): [ROADMAP.md](../../ROADMAP.md)

새 사이드 프로젝트 시작하시면 화이팅. 첫 commit 다음에 측정 코드 넣는 거 잊지
마세요.
