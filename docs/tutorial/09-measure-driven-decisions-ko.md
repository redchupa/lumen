# 9장. 측정-주도 의사결정 — 직관 11번 중 8번이 틀렸던 이유

> 이 장에서는 이 책의 전반에서 자주 등장한 "측정으로 부정", "11번 중 8번 부정"이
> 정확히 어떤 사이클이고 왜 그 패턴이 사이드 프로젝트의 본질인지 다룹니다. Lumen
> v0.5 cycle의 11번 가설을 하나씩 살펴보고, 측정 인프라를 어떻게 짜야 그 사이클이
> 가능한지까지. 이 장이 끝나면 새 사이드 프로젝트를 시작할 때 첫 commit 다음에
> "측정 코드"가 들어가는 습관이 생길 거예요.

## 9.1 가설 → 측정 → 결정 — 한 사이클의 모양

측정-주도 의사결정은 한 줄로 풀면 이렇습니다.

> **새 시도를 하기 전에 가설을 명시하고, 측정으로 검증하고, 결과에 따라 결정한다.**

순서가 중요해요.

1. **가설 명시**: "X를 하면 Y가 일어날 것이다"라는 예측을 명확히 적어둠
2. **측정 인프라**: 가설을 검증할 수 있는 측정이 가능한 상태인가 확인
3. **변경 적용**: 코드 변경
4. **측정**: 같은 환경에서 변경 전/후 결과 비교
5. **결정**:
   - 가설 맞음 → 변경 채택, 다음 단계로
   - 가설 틀림 → 변경 revert, 또는 default-off로 유지, 다른 가설로

이게 한 사이클이에요. 한 사이클이 짧을수록 좋아요. Lumen에서는 보통 한 사이클이
1~3일.

### 왜 이 순서가 중요한가

가장 흔한 잘못은 **2번을 건너뛰는 것**. 측정 인프라가 없는 상태에서 변경부터
하면 결과를 확인할 방법이 없어요. "예전보다 빨라진 것 같다"는 직관에 의존하게
되고, 실제로는 느려졌어도 모르고 commit합니다.

Lumen v0.1.0의 첫 commit이 "안녕"이라는 토큰 1개 출력하기였는데, 두 번째 commit이
"토큰 생성 시간 측정 코드"였어요. 그 이후 모든 변경은 측정 가능한 상태에서 진행
했습니다. 이게 11번 사이클을 가능하게 한 토대예요.

## 9.2 부정된 8번의 가설

v0.5 cycle에서 측정으로 부정된 가설들을 시간 순으로 봅니다. 각 가설이 왜 합리적
이었는지, 측정 결과는 어땠는지, 그게 왜 흥미로운 발견인지.

### Phase 7.M — Q8 정수 dot product

**가설**: "Q8 weight를 dequantize하지 말고 정수 그대로 dot product 하면 fp32 매트
멀보다 빠를 것."

**근거**: 정수 곱셈이 fp32 곱셈보다 단순. CPU 명령어 1개로 i8 × i8 → i16 가능.
SIMD로 묶으면 더 빠를 것.

**측정**: 약간 빠르지만 net-neutral (약 0%). 변환 비용과 정수 dot의 작은 가속이
상쇄.

**왜 흥미로운가**: 직관(정수가 빠르다)이 맞긴 했는데 효과가 미미. 더 큰 win은
다른 곳에 있다는 시그널. 이게 후속 measure에서 4-acc 패턴(Phase 7.G) 발견의 토대가
됐어요.

### Phase 7.N — VNNI single-accumulator

**가설**: "Intel/AMD의 VNNI 명령어(vpdpbusd)는 32개 int8 곱셈+합산을 한 명령에
처리. 단일 accumulator만 써도 충분히 빠를 것."

**근거**: 한 명령에 더 많은 일을 하니까 적은 명령으로 같은 결과. SIMD 명령 카운트
줄이는 게 가속의 핵심.

**측정**: -3.6% 회귀.

**왜**: 한 명령이 무거운 만큼 latency가 길어요(7 사이클). 한 acc만 쓰면 dependency
chain이 길어져서 OoO window가 차버립니다. SIMD 명령 카운트만 보고 판단하면 안
된다는 교훈.

### Phase 7.O — VNNI 4-accumulator

**가설**: "VNNI에 4-acc 패턴을 결합하면 single-acc보다 빠르고 fp32 4-acc보다도
빠를 것."

**근거**: 4-acc로 dependency chain 끊고 + VNNI로 명령당 일 양 늘리면 둘의 장점
결합.

**측정**: fp32 4-acc 대비 -2.7% 회귀.

**왜**: VNNI 명령이 너무 무거워서 4-acc로 dependency 끊어도 OoO window에 다 못
들어감. fp32 4-acc는 명령이 가벼워서 OoO에 잘 fit. 같은 작업량이라도 명령
"크기"에 따라 다른 결과.

**다음**: 이걸 그냥 폐기 안 하고 인프라 유지. Phase 7.P에서 shape-aware dispatch
가 추가되면서 짧은 K 매트멀에 한정 활성화 → **win으로 부활**. 7.O의 인프라가
7.P에서 살아남.

### Phase 7.R/S/T — AVX-512 ZMM lane width

**가설**: "AVX-512 ZMM은 한 명령에 16개 float, AVX2 YMM은 8개. 단순 비례로 ZMM이
2배 빠를 것."

**근거**: lane width가 2배니까 명령 카운트 절반.

**측정**: -4.5% 회귀 (AMD Zen 4에서).

**왜**: AMD Zen 4는 AVX-512를 내부적으로 256비트 두 번으로 쪼개서 실행. 즉 ZMM
명령이 두 µop으로 분해돼서 사실상 YMM 두 번과 같음. 처리량은 같은데 인코딩
오버헤드와 dependency 체인이 길어져서 손해.

**중요**: 같은 코드가 Intel Sapphire Rapids 같은 native 512-bit silicon에서는 win
일 가능성 큼. 그래서 인프라 폐기 안 함. Lumen의 ZMM 코드는 default-off 상태로
main에 살아 있어요. 다른 호스트에서 측정해서 결정.

### Phase 8.B — chunk L2-fit cap

**가설**: "매트멀 chunk 크기를 CPU L2 캐시(1MB)에 맞게 cap하면 cache miss 줄어
빨라질 것."

**근거**: cache hit이 cache miss보다 100배 빠름.

**측정**: -3.5% 회귀.

**왜**: chunk가 작아지면 dispatch 횟수 늘고 inter-chunk 동기화 비용 늘어남. cache
hit 이득보다 dispatch 손해가 큼. 또 매트멀이 sequential하게 weight 읽으니까 L2
fit 여부보다 prefetch와 streaming load가 더 중요.

### Phase 8.C — Software prefetch

**가설**: "매트멀 inner loop에 prefetch 명령 추가하면 메모리 latency를 hide할
수 있을 것."

**근거**: 1장에서 본 메모리 병목. prefetch로 다음 데이터 미리 가져오면 stall 줄어.

**측정**: 8 thread -49% 회귀!

**왜**: 8 thread가 동시에 prefetch 명령 발급하면 메모리 컨트롤러 queue가 가득
차요. prefetch 자체가 memory transaction이라서 메인 load들의 latency를 오히려
증가시킴. 단일 thread에서는 약 win이지만 8 thread에서는 큰 loss.

이게 흥미로운 케이스. 같은 코드가 thread 수에 따라 정반대 효과. 측정 안 하면
절대 모를 결과.

### Phase 8.D.3 — Prefill batching

**가설**: "Decode가 매번 weight 전체를 다시 읽으니 느림. prefill로 batching하면
weight를 한 번만 읽으니 5~10배 빠를 것."

**근거**: ggml의 pp128/tg128 비율이 8.4배. 우리도 비슷한 이득 가능할 것.

**측정**: pp32가 22.65 tok/s. tg32(65 tok/s) 대비 token당 2.9배 더 느림. 회귀.

**왜**: plumbing(forward_prefill_jit 등)은 완벽한데 그 아래 Q8 N>1 매트멀 커널이
N=1 커널 × N보다 2.3~2.6배 느림. 4-acc 패턴 같은 마이크로 튜닝이 N>1 path에는
안 들어가 있었음. plumbing 잘 짰는데 base가 안 따라줘서 회귀.

### Phase 8.D.3 retro — Attention이 원인

**가설**: "Prefill 회귀의 원인은 attention의 O(seq²) 비용일 것."

**근거**: prefill에서 attention 비용이 32×32=1024로 늘어남. 큰 영향 가능성.

**측정 (50줄 단위 bench)**: attention 24 layer × 389µs = 9.3ms. pp32 전체 1410ms의
**0.66%**. 결정적 영향 없음.

**왜**: attention이 늘긴 했지만 절대 비중이 너무 작음. 매트멀 95%가 핵심.

**중요한 메타 패턴**: 이 가설을 **measurement 없이 commit message에 자신 있게 적었
어요**. 다음 날 50줄 측정으로 부정. "commit message에 추측 적지 말기" 규칙이 여기서
나옴.

## 9.3 확정된 3번의 결과

부정만 있었던 게 아니에요. 명확한 win이 2번, 확정된 진단 1번.

### Phase 8.A — Atomic ThreadPool

**가설**: "기존 ThreadPool은 mutex로 worker가 task 경쟁. atomic counter로 바꾸면
contention 줄어 가속."

**측정**:
- 1 thread: +9.1% 가속
- 2 thread: +14.7% 가속
- 4 thread: +4.3% 가속
- 8 thread: +2.9% (noise 영역)

**해석**: 가설 맞긴 한데 효과 크기가 thread 수에 따라 다름. 1~2 thread에서 큰
효과, 8 thread에서는 메모리 병목이 더 크므로 dispatch 개선 효과 작음.

**결정**: merge. 8 thread만 봤다면 noise라 회피했을 텐데, 측정을 thread 수별로
다 보니 다른 결정이 가능했어요. 측정의 다차원성이 중요.

### Phase 8.D.5 — N>1 codegen 비효율 확정

이건 가속이 아니라 **진단의 확정**입니다. Phase 8.D.3 prefill 회귀의 진짜 원인을
50줄 단위 bench로 정확히 찾아낸 사이클.

**측정**:
```
같은 work양에 대해
  N=32 커널 1회 호출:   2925µs (qkv shape 기준)
  N=1 커널 32회 호출:   1156µs

N>1 커널이 2.5배 더 느림.
```

**결정**: N>1 codegen을 rewrite하는 게 진짜 다음 단계. 다만 큰 작업(1~2주)이라
임시로 N=1 fan-out으로 우회(8.E.1). N>1 codegen rewrite는 v0.6.0 마일스톤.

### Phase 8.E.1 — N=1 fan-out 회복

**가설**: "N>1 커널이 느리니, prefill에서 N=1 커널을 N번 호출하면 N>1 커널 1번
보다 빠를 것."

**측정**:
- pp32: 22.65 → 54.15 tok/s (+2.39배)

회귀에서 회복. 임시 fix지만 진짜 prefill 가속(N>1 codegen rewrite)까지 다리.

## 9.4 패턴 정리 — 11번에서 무엇이 보이나

11번 사이클(8 부정 + 1 net-neutral + 2 확정 + 1 partial fix)에서 메타 패턴을 정리
하면 이렇습니다.

### 1. 직관은 자주 틀린다 — 특히 마이크로아키텍처에서

ZMM이 YMM보다 2배 빠르다, prefetch는 latency hide, batching은 weight reuse 살린
다. 다 책에 나오는 직관인데 우리 측정에서는 다 부정. 이유는 마이크로아키텍처 디
테일(double-pumped, memory queue, OoO window 등) 때문.

이 영역은 책으로 안 보입니다. 그 호스트에서 직접 측정해야 합니다.

### 2. 같은 코드가 호스트마다 다른 결과

ZMM 인프라는 Zen 4에서 회귀지만 Intel Sapphire Rapids에서는 win 가능성 큼. 같은
코드를 폐기하지 말고 default-off로 두는 이유. **측정은 호스트 의존적**이라는
사실.

### 3. 같은 코드가 thread 수마다 다른 결과

prefetch는 1 thread에서 win, 8 thread에서 -49% loss. 측정은 **thread 수 의존적**.

### 4. 가속과 회귀의 비율은 정직하게 8:3

6주 동안 11번 시도해서 win 2번, partial fix 1번. 부정 8번. 사이드 프로젝트로
같은 분야 고수(ggml)와 경쟁할 때 정직한 비율. **부정이 더 많아도 누적 진척은
가능**이라는 점이 핵심.

### 5. Commit message가 다음 날 자기 자신을 비웃을 수 있다

Phase 8.D.3에서 "attention이 원인"이라 적은 게 다음 날 8.D.4 측정으로 부정.
**측정 없는 추측은 commit message에 적지 말기**. 1주일 후 다른 사람(또는 future
me)이 그 추측을 진실로 받아들이고 잘못된 방향 갈 수 있음.

## 9.5 측정 인프라를 어떻게 짜는가

11번 사이클이 가능했던 토대가 측정 인프라입니다. 어떻게 짜야 하는지 정리.

### 1. 첫 commit 다음에 측정 코드

Lumen의 첫 commit이 "안녕" 토큰 출력. 두 번째 commit이 "토큰당 시간 측정". 이게
모든 후속 measurement-driven 사이클의 시작이었어요.

새 사이드 프로젝트 시작할 때 이 순서를 권합니다. 첫 commit 다음에 측정 코드.
무엇을 측정할지는 분야마다 다르지만 시간/throughput/메모리 사용량 중에 하나는
필수.

### 2. Step-별 시간 분해

전체 시간만 측정하면 어디가 병목인지 모릅니다. Lumen은 한 layer 통과를 step
단위로 분해해서 측정해요.

```
forward_layer_decode_jit_timed():
  step_timer["attn_rms"]    += rms_norm(...)
  step_timer["qkv_matmul"]  += qkv_matmul(...)
  step_timer["rope"]        += rope(...)
  step_timer["attention"]   += attention(...)
  ...
```

이 step 분해 덕에 "매트멀이 95%"라는 진단이 가능했어요. 매트멀만 봐도 5개 sub-step
이 다 측정 가능.

### 3. 같은 환경, 같은 입력 반복

매번 다른 입력으로 측정하면 비교 불가능. 같은 model + 같은 prompt + 같은 thread
수 + warmup 3회 후 측정 같은 패턴이 표준.

### 4. Run-to-run variance 알기

같은 측정 5번 돌리면 어느 정도 차이가 나는지 미리 알기. 보통 1~2%. 그 이상의
차이만 의미 있는 변화로 받아들임. 한 번만 측정해서 5% 변화 보고 결정하면 noise에
속을 수 있음.

### 5. Micro-bench 따로 두기

전체 forward를 measure하는 게 너무 무거우면 작은 함수만 격리해서 micro-bench.
Lumen은 매트멀 단독 timing, attention 단독 timing 등 50줄 단위 bench가 여러 개.
가설 검증에 빠르게 활용.

## 9.6 가설 부정 후 무엇을 하는가

가설이 부정됐을 때 흔한 반응이 "내가 잘못한 거구나" 자기 의심. 이걸 다르게 받아
들여야 사이드 프로젝트 지속 가능해요.

### 인프라 폐기 vs default-off

부정된 변경 두 가지 처리 방법.

**폐기 (revert)**: 코드 다 되돌리기. 흔적 없음. 다음에 같은 분야 다시 손대면
처음부터.

**default-off**: 코드는 main에 살리되 기본값을 off로. flag나 env var로 한 줄
변경하면 활성화 가능.

Lumen은 모든 부정 사례에서 **default-off** 선택. 이유:

- 미래 다른 호스트(Intel Sapphire Rapids 등)에서 win일 수 있음
- 미래 다른 조건(shape-aware dispatch 등 추가)에서 win일 수 있음 — 7.O의 7.P
  부활이 그 증거
- 인프라 다시 짜는 비용 vs main에 두는 비용 — main에 두는 쪽이 거의 무료

이렇게 측정-주도 부정이 **"이 코드는 잘못됐다"가 아니라 "이 조건에서는 동작 안
한다"**로 받아들여집니다.

### 자기 의심 다루는 법

11번 중 8번 부정이 처음엔 정신적으로 힘들어요. "내가 진짜 모르는 건가" 자기
의심.

이걸 다르게 받아들이는 패턴:
- "내 직관이 부정당했다" → "이 분야의 마이크로아키텍처 디테일이 책에 다 적혀
  있지 않다는 사실을 확인했다"
- "또 회귀구나" → "또 측정 데이터가 누적됐다 — 다음 phase 결정에 도움"
- "5번 째 부정이라 지친다" → "측정 인프라가 작동 중이라는 시그널"

이 받아들임이 사이드 프로젝트 5주 이상 지속의 핵심이에요. 매 부정을 손해로
받으면 6주째 흐지부지. 매 부정을 데이터로 받으면 다음 phase 진입이 자연스러움.

## 9.7 사이드 프로젝트 시작하시는 분께

이 책에서 사이드 프로젝트 한 사이클을 다 풀었어요. 같은 분야든 다른 분야든
시작하시려는 분께 정리.

**1. 첫 commit 다음이 측정.** "Hello World" 다음에 "Hello World 시간 측정". 측정
없이 5주 가면 흐지부지.

**2. 가설 명시 → 변경 → 측정 → 결정.** 변경 전에 가설을 commit message에 적고,
측정 후 결과를 또 적기.

**3. 1주 commit 0개도 OK.** 매주 진척 강제하면 호흡 무거워짐.

**4. 부정 사례 인프라 살리기.** default-off로 남기기. 7.O의 7.P 부활 패턴.

**5. 회고 글 한 phase에 한 편.** 메타 패턴 발견에 진짜 도움. 글로 풀면서 자기
사고가 정리됨.

**6. 잘 끝낼 줄 알기.** release tag 박고 잠시 쉬는 것도 능력. 사이드는 끝낼 줄
알아야 9개월 흐지부지 안 됨.

## 9.8 9장 핵심 정리

1. **측정-주도 의사결정의 사이클**: 가설 → 측정 인프라 확인 → 변경 → 측정 → 결정.
   한 사이클 1~3일.

2. **Lumen v0.5 cycle 결과**: 11번 시도, 8번 부정, 1번 net-neutral, 2번 win, 1번
   진단 확정, 1번 partial fix. 가속 비율 정직하게 8:3.

3. **직관은 마이크로아키텍처 분야에서 자주 틀린다**. ZMM lane width, prefetch,
   batching 등 책의 상식이 실제 호스트에서는 부정.

4. **인프라 폐기보다 default-off**. 미래 다른 호스트/조건에서 살아날 수 있음.
   7.O의 7.P 부활이 증거.

5. **측정 인프라 다섯 가지**: 첫 commit 다음에, step-별 분해, 같은 환경 반복,
   variance 알기, micro-bench 분리.

## 다음 장 미리보기

10장(마지막)은 **6주 회고**입니다. 이 책 전반에서 추상적으로 풀어둔 측정-주도
패턴이 실제로 6주 사이드 프로젝트 안에서 어떻게 흘렀는지, v0.1에서 v0.5까지의
실제 일기. 측정 데이터, 가설, 부정 사례, win, 그리고 무엇을 배웠는지를 시간
순으로 정리합니다. 이 장을 읽고 나면 "내 사이드 프로젝트도 같은 패턴으로 가능
하다"가 보일 거예요.
