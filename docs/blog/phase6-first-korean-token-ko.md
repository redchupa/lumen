# Phase 6 — 처음으로 한국어가 나왔다

> Lumen 빌드로그 #7.

## 출력

```
prompt: "안녕"
generated 3 tokens in 1.61초 (0.54s/tok)
  id  91145 → "하세요"
  id     11 → ","
  id 134561 → " 저는"

full text: "안녕하세요, 저는"
```

이게 진짜 출력입니다. Qwen2.5-0.5B-Instruct-Q8_0 가중치 파일에서 시작해서, Lumen이 자체로:

1. GGUF v3 파일을 파싱하고
2. 24 layer × 0.5B params를 Q8_0 → fp32로 dequant하고
3. GPT-2 byte-level BPE로 "안녕" 토큰화하고
4. RMSNorm + GQA(14Q/2KV) attention + RoPE(base=1M) + SiLU FFN을 24번 돌고
5. lm_head로 151,936-dim logit 계산하고
6. argmax로 다음 토큰 골라
7. KV cache 업데이트 후 반복하고
8. 결과 토큰 ID들을 byte mapping 역적용으로 한국어 복원

이걸 외부 의존성 1개(`thiserror`)로 합니다. PyTorch, candle, mlx, llama.cpp 모두 없음. cargo의 dep tree에 들어가는 모든 게 우리 코드.

## 18일에 걸친 21개 커밋의 의미

Phase 1을 시작했을 때 PLAN.md 한 줄로 정의한 게:

> IR이 양자화 커널을 자동 합성하는 LLM 추론 컴파일러

당시 IR도 없었고, 백엔드도 없었고, 양자화도 없었고, 토크나이저도 없었고, 모델 추론은커녕 matmul도 못 했음. 6주 안에 다 만들기는 미친 계획이었다.

오늘 그 한 줄이 진짜로 동작합니다. 단계별로 쌓아온 것들:

| Phase | 시점 | 의미 |
|---|---|---|
| 1 | 첫째 날 | DSL 파서 — 텐서 형상이 타입에 박힌 미니 언어 |
| 2.A | | IR + C 백엔드 reference oracle |
| 2.B | | 자체 x86_64 머신코드 + JIT W^X 페이지 |
| 3.A/B/C | | AVX2 자동 합성, 4×8 register tile, **57 GFLOPS** |
| 5.A~E | | 양자화 reference → native Q8 dequant → **Dequant×MatMul 융합** → GGUF v3 reader → 디스크-to-fp32 closed loop |
| 6.A/B | | Byte-level BPE + GGUF에서 vocab/merges 로드 |
| 6.C.1/2 | | Llama op reference + Native RMSNorm |
| 6.D | | Transformer layer forward (Hybrid-B 설계) |
| 6.E.1/2 | | KV cache + greedy decode + Model 구조체 |
| 6.F.1 | | Toy GGUF round-trip — pipeline 정합성 |
| 6.F.2.a | | F16/Q8/Q4 dequant in loader |
| 6.F.2.b | | **Real Qwen2.5-0.5B 가중치 로드** + attention biases |
| 6.F.2.c | | GPT-2 byte-to-unicode tokenizer mode |
| **6.F.2.e** | **오늘** | **한국어 토큰이 나왔다** |

## 의외의 발견들

### 1. naive Rust matmul로 0.54s/token이 나온다

이 결과를 보기 전엔 토큰당 10-30초를 각오했다. 0.54s/token은:
- 24 layers × (Q/K/V/O proj + gate/up/down + attention)
- lm_head 896 × 151,936 (per token!)

내 손코딩 `weight_matmul` (triple loop, 추가 인라인 없음)이 약 1 GFLOPS 수준일 텐데, 그래도 의미 있는 속도가 나오는 이유는:
1. 0.5B 모델은 작다 (대부분 cache fit)
2. release mode에서 LLVM이 inner loop를 잘 인식
3. 단일 토큰 forward는 메모리 대역폭 bound

Phase 3의 자체 AVX2 backend (57 GFLOPS)를 plug-in으로 바꾸면 **10-20× 빨라질 것**. Phase 6.G로 잡혀있음.

### 2. Hybrid-B 결정이 정확했다

Phase 6.C.3에서 SiLU/RoPE의 자체 합성을 포기하고 Rust reference + JIT matmul만 가는 결정을 했다. 그 결정 덕분에:
- Phase 6.D부터 6.F.2.e까지 3-4일 만에 완성
- 동작이 동작이지, "이 op은 안 만들어서 못 돌려"가 없음
- LLM 추론에서 SiLU/RoPE는 1% 시간. 자체 합성하면 정확도 손실 + 디버깅 큼

llama.cpp도 사실 같은 패턴 — sigmoid는 libm, matmul만 손코딩 SIMD.

### 3. GPT-2 byte mapping의 함정

첫 번째 "안녕" 시도에서 panic. 원인: BPE 초기 토큰을 raw byte 단위로 시작했는데, GPT-2 vocab은 256개 byte-mapped chars (2-byte UTF-8 포함)를 초기 토큰으로 가짐. 한 char = 1 token (initial), BPE merges가 그 위에 학습.

이걸 알게 되니 수정은 한 줄. 그런데 알기까지 디버깅이 1시간. **표준 알고리즘 = 한 줄이 잘못되면 panic + 디버깅 1시간**이라는 컴파일러 일의 본질.

### 4. Qwen2의 attention bias 존재

Llama2/3 가정으로만 짜놓은 `LayerWeights`에 Qwen2의 `attn_q.bias`/`attn_k.bias`/`attn_v.bias`가 없어서 처음엔 가설을 잘못 잡았다. Inspect 결과 보고 추가. 모델 패밀리마다 작은 추가 사항이 있다는 걸 알게 됨 — Phase 7에서 EXAONE, HyperCLOVA-X도 같은 식으로 진행할 때 참고.

## 시연 명령어

```sh
cargo test -p lumen-model --release --test e2e_qwen_load \
    qwen_generates_first_korean_tokens -- --ignored --nocapture
```

출력:
```
loading Qwen2.5-0.5B-Q8_0 ...
  loaded in 765ms
prompt: "안녕" → ids: [126246, 144370]
generated 3 tokens in 1.61s  (0.54s/tok)
new ids: [91145, 11, 134561]
  id  91145 → "하세요"
  id     11 → ","
  id 134561 → " 저는"
full text: "안녕하세요, 저는"
```

## 숫자

- 21 커밋, ~11,100 lines of Rust
- 124 단위/통합 테스트 + 1 ignored heavy integration (Qwen full forward)
- 외부 production dep: `thiserror`
- 외부 dev dep: `libloading`, `tempfile`, `criterion`
- 인코더 명령어: 30+ (REX, VEX 2/3-byte, F16C, AVX2, FMA-3)
- IR op 자동 합성: matmul (3 tiers) + Dequantize + Dequant×MatMul 융합 + RmsNorm

## 다음

Phase 7:
1. **JIT plug-in** — model.rs의 `weight_matmul`을 Phase 3.C의 4×8 tile JIT 커널로 swap. 토큰당 0.54초 → 0.1초 이하 목표.
2. **벤치마크** vs llama.cpp on 같은 모델/하드웨어. 우리가 어디서 우월하고 어디서 아직 느린지 솔직 측정.
3. **블로그 5편 합본** — Phase 1부터 여기까지의 여정. 한국 기술 커뮤니티 공유.

레포: https://github.com/redchupa/lumen

— Claude와 페어 프로그래밍으로 작성.
