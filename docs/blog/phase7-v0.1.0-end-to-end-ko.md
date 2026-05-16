# v0.1.0 — 처음부터 끝까지, 외부 추론 의존성 0개로

> Lumen 빌드로그 #8. Phase 1 ~ 7.A 합본.
> 한 마디로: **`thiserror` 하나만 가지고 Qwen2.5-0.5B-Q8_0를
> 한국어로 추론하는 Rust 추론 컴파일러 + 런타임이 완성됐다.**
> llama.cpp 대비 단일 스레드 tg32 기준 **9.3× 느림 (4.43 vs
> 41.32 tok/s).** 왜 그 격차가 나는지, 어디서 줄일지는 끝에.

## 한 화면 요약

```
$ cargo test -p lumen-model --release --test e2e_qwen_load \
    qwen_bench_tg32_jit -- --ignored --nocapture

Lumen JIT tg32: 32 tokens in 7.22s  =>  4.43 tok/s (single-thread)

$ ./tools/llama-cpp/llama-bench.exe \
    -m qwen2.5-0.5b-instruct-q8_0.gguf -p 128 -n 32 -t 1

| model                  |       size | threads |   test |     t/s |
| Qwen2 0.5B Q8_0        |  638.62 MB |       1 |  pp128 |  160.82 |
| Qwen2 0.5B Q8_0        |  638.62 MB |       1 |   tg32 |   41.32 |
```

같은 모델, 같은 프롬프트("안녕"), 같은 스레드 수. 토큰 시퀀스도
정확히 같음 (`qwen_generate_jit_matches_naive_and_speed` 가 단언).

## 7주, 25 커밋의 여정

날짜로 보면 작은 단위로 자주 커밋했지만, 큰 그림에서는 다섯 덩어리.

### 덩어리 1 — DSL + IR (Phase 1, 2.A)

매트랩 인덱싱처럼 보이지만 텐서 형상이 타입의 일부인 작은 언어.

```rust
fn matmul(
    a: tensor<f32, [64, 128]>,
    b: tensor<f32, [128, 32]>,
) -> tensor<f32, [64, 32]> {
    return a @ b;
}
```

`a.shape[1] == b.shape[0]` 검사는 컴파일 시점에 끝나고, 결과 형상도
컴파일 시점에 추론. Pratt 파서, AST, 타입 검사, SSA IR, lower 패스,
verify, print, C emit까지 — 21 테스트로 그린 첫 컬러 풀.

### 덩어리 2 — 자체 x86_64 백엔드 (Phase 2.B, 3.A~C)

여기서부터 외부 LLVM/Cranelift 안 씁니다. `Vec<u8>`에 직접 바이트를
쌓는 어셈블러부터 시작.

```rust
// crates/lumen-codegen/src/x86_64_enc.rs
pub fn mov_ri64(buf: &mut Vec<u8>, dst: Reg, imm: i64);
pub fn vfmadd231ps_mem(buf: &mut Vec<u8>, dst: Ymm, a: Ymm, base: Reg, disp: i32);
```

W^X 페이지(Windows: `VirtualAlloc` + `VirtualProtect`, Linux:
`mmap` + `mprotect`)에 memcpy하고 함수 포인터로 캐스팅해서 호출.

- **3.A** AVX2 도입: 1×8 lane → naive Rust 대비 8.7×
- **3.B** F16C: `vcvtph2ps`로 fp16 load
- **3.C** 4×8 register tile: 4 독립 accumulator + 8-lane FMA →
  단독 매트멀 벤치에서 **57 GFLOPS, scalar 대비 19×**

### 덩어리 3 — 양자화 + GGUF (Phase 5.A~E)

Q4_0(18B 블록) / Q8_0(34B 블록) 디퀀트 reference를 Rust로 짠 뒤,
**IR 레벨에서 `Dequantize → MatMul` 패턴을 매처가 인식해 융합 커널을
자동 합성**하도록 만들었습니다. 이게 Lumen의 시그니처 — 새 양자화
포맷이 와도 IR 추가만 하면 모든 백엔드에 전파.

같은 시기에 GGUF v3 reader도. `general.alignment`, KV 메타데이터,
텐서 디스크립터, `ggml_row_size` parity까지 다 살피느라 5.D는 작지
않은 커밋이 됐습니다.

### 덩어리 4 — Transformer + 첫 한국어 (Phase 6.A~F)

GPT-2 byte-level BPE 토크나이저, RMSNorm, GQA(14Q/2KV) attention,
RoPE split-half(base=1M), SiLU FFN, KV cache, greedy sampling,
autoregressive decode loop.

여기서 가장 오래 헤맸던 게 토크나이저였습니다. "안녕"이 GGUF vocab
에 있는데 encode가 "토큰을 찾을 수 없다"고 panic. 원인은:
**GPT-2 BPE는 raw bytes로 시작하지 않고, 256개 byte-to-unicode
매핑된 chars(`Ġ`, `Ā` 같은)로 시작한다.** vocab에는 이 매핑된 형태가
들어 있으니까. 한 줄짜리 함수가 모든 걸 푸는데 그걸 찾는 데 한
세션이 걸린 적도 있었습니다.

```
prompt: "안녕"
naive (1.63s / 3 tokens): ids [91145, 11, 134561] → "안녕하세요, 저는"
JIT   (1.12s / 3 tokens): ids [91145, 11, 134561] → "안녕하세요, 저는"
```

### 덩어리 5 — JIT 통합 + 벤치 (Phase 6.G, 7.A)

처음에는 코드 생성기 단독 테스트만 있고, 실제 forward 패스는 여전히
naive Rust 매트멀이었습니다. Phase 6.G가 이 둘을 잇는 작업.

**`MatmulJitCache`** — `(M, K, N)` 키 → `unsafe extern "C" fn(*const f32, *const f32, *mut f32)`.
같은 형상이 두 번째 들어오면 컴파일 안 함. 모델 forward 한 번이
24 layer × (Q/K/V/O + gate/up/down) = 168개 매트멀이지만, 형상이
반복되니까 캐시 엔트리는 손에 꼽음.

decode에서 가장 큰 매트멀이 lm_head (`[1, 896] @ [896, 151936]`).
여기 하나만 JIT 태우는 게 6.G.2, 24 layer 전부 태우는 게 6.G.3.

그리고 **정답성 단언이 들어옴**: `qwen_generate_jit_matches_naive_and_speed`가
naive 경로와 JIT 경로의 토큰 시퀀스를 정확히 비교해서 bit-identical
인지 확인. 빠른 코드가 틀린 답을 내면 의미 없으니까.

## 9.3× 격차의 정직한 해부

Phase 7.A에서 마침내 llama.cpp 공식 빌드(`b9174-bin-win-cpu-x64`,
AVX2)와 같은 모델로 같은 워크로드를 돌렸습니다.

| 경로 | tg32 tok/s | vs llama.cpp |
|---|---:|---:|
| Lumen naive Rust | 2.91 | 14.2× slower |
| **Lumen JIT** | **4.43** | **9.3× slower** |
| llama.cpp (ggml) | 41.32 | 1.0× |

**JIT는 naive 대비 1.52× 빨라졌습니다.** 더 못 가져온 이유:

1. **M = 1 decode 한계**. autoregressive decode는 매트멀이 `(1, K, N)`.
   `M % 4 != 0` 이라 4×8 register tile이 안 켜지고 1×8 AVX2로
   떨어집니다. prefill (≥4 토큰 배치)이면 tile이 켜지고 격차가
   훨씬 벌어질 텐데, 아직 prefill 배치 패스가 없음.
2. **Naive baseline이 이미 뜨거움**. release 모드 LLVM이 안쪽 루프
   를 적당히 auto-vectorize. cold-Rust → 1×8-AVX2 점프는 2× 정도.
3. **Non-matmul ops가 무게를 차지**. attention 내부 Q·K^T, softmax,
   attn·V, RoPE, RMSNorm 다 Rust reference. Hybrid-B 결정 — 컴파일
   타깃은 양자화 매트멀에 집중, 초월함수/normalization은 Rust로.

**ggml이 우리 JIT보다 9.3× 빠른 이유는 미스터리가 아닙니다:**

1. **Q8-native matmul.** ggml은 양자화 레이아웃 그대로 곱합니다.
   우리는 load 시점에 Q8_0 → fp32로 풀고 fp32×fp32 매트멀을 하니까
   메모리 대역폭을 4배 쓰고, `vpmaddubsw`/`vpdpbusd` 정수 SIMD도
   못 씁니다.
2. **캐시 블로킹.** ggml은 L1/L2/L3 타일을 따로 둠. 우리는 register
   tile(4×8)만 있고 outer cache-block 루프가 없어서 working set이
   L2를 넘어가면 떨어짐.
3. **flash-style attention.** ggml은 attention 패스를 직접 손튜닝.
   우리는 교과서 공식 그대로.
4. **성숙한 AVX2 사용.** prefetch, FMA 2/cycle 피크를 친 명령어
   스케줄링. 단독 매트멀(3.C)에서는 우리도 이론 peak의 ~50%까지
   가지만, 실제 forward에서는 한참 떨어짐. ggml은 forward에서도
   80%+를 침.

이게 v0.1 → v1.0 작업 리스트입니다. **이 중 첫 세 개만 들어와도
같은 한 자릿수 격차로 좁힐 수 있을 거라고 봅니다.**

## 무엇이 정말 새로운가

비교 가능한 프로젝트들과 솔직히 비교하면:

- **llama.cpp** — 인간이 양자화×dtype×형상 조합별로 손코딩한 수백
  개의 ggml 커널. 빠릅니다. 그래서 어떻게 빠르게 가는지 잘 알아요.
- **candle / mlx** — Rust/Swift 추론. 하지만 둘 다 외부 BLAS나
  네이티브 라이브러리를 결국 호출함.
- **tinygrad** — 그래프 → 커널 자동 합성. 영감의 출처지만 우리와
  스코프가 다름 (학습 포함).

Lumen이 다른 점:

1. **IR 레벨 양자화 패턴 매칭으로 dequant×matmul 융합 커널을 자동
   합성**합니다. 새 양자화 포맷이 와도 사람이 커널을 안 짭니다 —
   IR 패턴 매처와 codegen만 받쳐주면 됨.
2. **외부 추론 의존성 0개.** 프로덕션 경로의 의존성은 `thiserror`
   하나입니다. 다른 추론 엔진(ggml/ONNX/candle/mlx)을 임포트하지
   않습니다. cargo dep tree가 한 화면에 들어옴.
3. **그러면서 토큰 시퀀스가 bit-identical** — naive 경로와 JIT
   경로의 결과를 통합 테스트에서 비교 검증. 속도를 정답성으로 사지
   않습니다.

## v0.1.0 — 무엇을 약속하는가 / 안 하는가

**약속:**
- Qwen2.5-0.5B-Q8_0를 한국어 프롬프트로 추론합니다 (지금 이 모델은
  표준 벤치마크 모델이라 검증이 깔끔)
- 단일 스레드 4.43 tok/s decode (M1 macOS / Linux는 아직 측정 안
  했음 — Windows AVX2만 검증)
- 동일 시드/프롬프트에 대해 naive 경로와 JIT 경로의 토큰이 bit-identical

**약속하지 않음:**
- 멀티스레드 (decode parallel은 어렵고, prefill만이라도 다음 마일스톤)
- ARM64 (Phase 2.C에서)
- CUDA (Phase 4.5 — 일정 후순위)
- 모델 학습 (non-goal, 영원히 안 함)
- 1B+ 모델 (Q8 dequant 풀면 RAM이 부족할 수 있음 — Q8-native가
  들어와야 자연스러움)

## 다음 10×는 어디 있나

벤치 봤으니까 어디서 가져올지 명확합니다 — 이거 다섯 개:

1. **Cache blocking** — 큰 매트멀(lm_head 896×151,936, FFN
   gate/up/down 896×4864)을 outer 블록 루프로 감싸기. tg32 → ~10
   tok/s 예상.
2. **Q8-native fused matmul kernels** — Phase 5.C가 prototype.
   디퀀트 패스 자체를 건너뜀. memory-bound 형상에서 다시 ~2×.
3. **Specialized attention** — flash-style fused softmax. 이 모델
   사이즈에서는 보통, 긴 컨텍스트에서는 큰 효과.
4. **AVX-512** — 일부 소비자 CPU에만 있지만 있으면 1.5-2×.
5. **Multi-thread prefill** — physical core 수만큼 거의 선형.

처음 셋만 들어와도 단일 스레드 ggml과 같은 자릿수에 들어와야 합니다.

## 마무리

v0.1.0은 "**처음부터 끝까지 직접, 정답을 보장하면서, 격차는 정직하게**"의
스냅샷입니다. 빠른 코드를 가져왔다고 자랑하는 게 아니라, 매트멀 한
줄을 어떻게 짤지부터 시작해서 실제 LLM이 한국어를 출력하기까지의
전 경로가 우리 코드라는 게 자랑입니다. 격차는 격차고, 격차 안에는
어디로 가야 하는지가 또렷이 보입니다.

다음 마일스톤은 Phase 3.D + Phase 5.C 본격 통합. 같은 모델, 같은
명령어로 다시 재면 됩니다. 코드는 [github.com/redchupa/lumen](https://github.com/redchupa/lumen),
태그 `v0.1.0`.
