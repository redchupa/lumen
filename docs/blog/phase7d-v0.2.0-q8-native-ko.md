# v0.2.0 — Q8 weights를 끝까지 Q8로 들고 가니까 4배 빨라졌다

> Lumen 빌드로그 #9. Phase 7.C + 7.D 합본.

## 결론부터

```
v0.1.0 tg32 (single thread):  4.43 tok/s    [Phase 7.A 기준]
v0.2.0 tg32 (single thread): 17.97 tok/s    [+305%, 4.05×]

vs llama.cpp (41.32 tok/s):  9.3× slower → 2.30× slower
```

토큰은 그대로 bit-identical. `qwen_generate_jit_matches_naive_and_speed`
가 모든 커밋에서 그대로 통과합니다 — naive Rust 경로와 JIT 경로의
디코드 토큰 시퀀스가 정확히 같음을 단언.

## 두 개의 win — 한 개는 작고, 한 개는 컸다

### Phase 7.C — 1×N 4-accumulator decode kernel (+15%)

기존 1×8 AVX2 path:

```asm
.kk_loop:
  vbroadcastss ymm1, [r12 + rax*4]      ; A[kk] broadcast
  vfmadd231ps ymm0, ymm1, [r13 + rdx*4] ; ymm0 += ymm1 * B[kk, j..j+8]
  inc rcx
  jmp .kk_loop
```

문제: `ymm0` 단일 누산기. FMA latency 4 cycle, throughput 1 FMA/cycle.
종속성 체인이 매 cycle을 못 채우고 4 cycle마다 1 FMA. **이론 peak
대비 25%**.

해결: `M == 1, N % 32 == 0`인 decode 형상에 4 누산기 1×32 stride 커널을
새로 dispatch:

```asm
.kk_loop:
  vbroadcastss ymm4, [r12 + rcx*4]
  ; 4 independent FMA chains
  vfmadd231ps ymm0, ymm4, [r13 + rdx*4 +  0]
  vfmadd231ps ymm1, ymm4, [r13 + rdx*4 + 32]
  vfmadd231ps ymm2, ymm4, [r13 + rdx*4 + 64]
  vfmadd231ps ymm3, ymm4, [r13 + rdx*4 + 96]
  inc rcx
  jmp .kk_loop
```

이론상 4배 win. 실측 **+14.7%** (4.43 → 5.08 tok/s). 4배 못 나온 이유:
매트멀이 forward의 일부일 뿐이고, **dequant 패스가 메모리 대역폭을
이미 잡아먹고 있어서** FMA 처리량은 진짜 병목이 아니었음.

이 결과 자체가 다음 작업(7.D)의 정확한 ROI 근거가 됐습니다.

### Phase 7.D — Q8 weights를 native로 (+254%)

여기가 진짜 큰 거.

**이전 (v0.1.0):**

```
disk file (.gguf, Q8_0)
   ↓ at load
fp32 dequant — 640MB Q8 → ~2.4GB fp32 (in memory)
   ↓ at load
transpose_for_jit — 1.94 seconds spinning
   ↓ per-token forward (24 layers × 7 matmuls + lm_head)
fp32 × fp32 matmul through JIT
```

**이후 (v0.2.0):**

```
disk file (.gguf, Q8_0)
   ↓ at load
Q8 blocks held raw in Vec<BlockQ8_0> — natural [d_out, d_in] ggml layout
   ↓ at load
transpose_for_jit — 700 nanoseconds (no F32 weights to transpose)
   ↓ per-token forward
fused Q8 × F32 matmul kernel — never materializes fp32 weights
```

새 커널 (`emit_quant_matmul_q8_n1_body`):

```text
for i in 0..M (output rows):       # M = d_out
  ymm0 = 0                         # 8-wide accumulator
  for kb in 0..K/32:               # one Q8_0 block at a time
    ymm_d = broadcast(fp16→fp32(d[kb]))
    # 4× unrolled 8-wide chunks across the 32 weights in this block:
    for inner in [0, 8, 16, 24]:
      ymm_w = vpmovsxbd [block + 2 + inner]  ; 8 i8 → 8 i32
              vcvtdq2ps                       ; 8 fp32
              vmulps    × ymm_d              ; scaled weights
      ymm_a = vmovups [act + (kb*32+inner)*4]
      ymm0  += ymm_w * ymm_a                 ; vfmadd231ps
  # horizontal sum ymm0 → scalar → out[i]
  vhaddps ymm0, ymm0, ymm0
  vhaddps ymm0, ymm0, ymm0
  vextractf128 xmm1, ymm0, 1
  vaddss xmm0, xmm0, xmm1
  movss [out + i*4], xmm0
```

K 방향에서 SIMD를 잡고 마지막에 horizontal reduce. M=1인 decode와
딱 맞아떨어지는 형상.

**무엇이 사라졌나:**

1. **F32 dequant 메모리 패스 자체**: 24 layer × 7 weight × 약 800KB = ~135MB
   per token이 메모리에서 사라짐.
2. **메모리 대역 4×**: Q8는 byte당 ~1.0625, F32는 4. 작은 모델이지만
   매트멀이 메모리-바운드일 때 직접 4배.
3. **transpose_for_jit 1.94초**: F32 weights가 transposed 레이아웃을
   원했지만 Q8는 ggml native `[d_out, d_in]` 그대로 사용 가능 — transpose
   안 함.

**측정**: 5.08 → 17.97 tok/s (+3.54×). 원본 4.43 baseline 대비 +305%.

## 설계 메모

### WeightStorage enum — 두 갈래를 한 타입에

```rust
pub enum WeightStorage {
    F32(Vec<f32>),              // pre/post-transpose
    Q8(Vec<BlockQ8_0>),         // native [d_out, d_in], no transpose
}
```

로더 (`tensor_to_storage`)는 GGUF의 dtype을 보고 분기:
- `Q8_0` → 디스크에서 메모리로 블록 그대로 들고 옴 (dequant 없음)
- F32/F16 → 기존 경로

forward 디스패처:
- naive 경로: `WeightStorage::as_f32_native()` — Q8 케이스만 dequant
  (테스트용이라 느려도 OK)
- JIT 경로: `weight_matmul_jit_storage()` — F32는 기존 fp32 커널,
  Q8는 새 fused 커널

같은 forward 코드에서 두 경로가 공존하고, 정답성 단언은 한 군데
(`qwen_generate_jit_matches_naive_and_speed`)에서 강제됩니다.

### 5.C IR 패턴을 N=1로 확장

Phase 5.C에서 만들어 둔 `Op::Param(Q8) + Op::Dequantize + Op::MatMul +
Op::Return` 패턴은 N % 8 == 0만 받았습니다. 디코드는 N=1. 여기서 패턴
인식기는 그대로 두고, **codegen 안에서 N==1 분기를 추가**해서 형상별로
다른 body 함수를 emit:

```rust
if n == 1 {
    emit_quant_matmul_q8_n1_body(...)    // 새 K-direction SIMD 커널
} else {
    emit_quant_matmul_q8_body(...)       // 기존 N-direction 커널
}
```

같은 IR에서 두 specialization이 나오는 게 Lumen이 자랑하는 "IR에서
패턴 매처가 backend별 최적 커널을 자동 합성"의 작동 사례. IR을 안
건드리고 codegen만 늘렸습니다.

## 무엇이 남았나

벤치 격차는 2.30×. 어디서 좁힐지:

1. **Flash-style attention** — attention 내부 Q·K^T, softmax, attn·V는
   여전히 pure Rust. 0.5B는 효과 작지만 긴 컨텍스트에서는 큰 효과.
2. **Multi-thread prefill** — physical core 수만큼 거의 선형. decode는
   parallelism이 어려움.
3. **AVX-512 dispatch** — 갖춰진 CPU에 한해 1.5-2×.
4. **Token embeddings도 Q8** — lm_head는 weight-tied되어 있어서
   token_embeddings를 dequant하지 않아야 진짜 545MB → 144MB. 지금은
   embeddings만 fp32로 들고 있음.

남은 격차가 한 자릿수로 들어왔으니, 다음 마일스톤은 multi-thread
prefill + flash attention 한 쌍이 자연스러워 보입니다.

## 마무리

v0.1.0이 "처음부터 끝까지 직접 짰다"의 마일스톤이었다면, v0.2.0은
"정직하게 봤더니 어디가 느린지 보이더라"의 마일스톤입니다. dequant
패스 한 줄 빼는 데 한 commit이 들어갔지만, 그 한 줄이 forward 전체
속도의 절반 이상을 결정했습니다.

좋은 시스템은 처음부터 빠른 게 아니라 **어디가 느린지 즉시 측정
가능한 시스템**입니다. Phase 7.A의 BENCHMARK.md 덕분에 Phase 7.C/D의
ROI가 명확했고, 결과도 정직하게 보고할 수 있었습니다. 5.08 → 17.97은
숫자 자체보다 "왜 그게 일어났는지 추적 가능한 시스템이 만들어졌다"가
더 큰 자산입니다.

다음: Phase 7.E (flash attention) + 7.F (multi-thread prefill).

코드는 [github.com/redchupa/lumen](https://github.com/redchupa/lumen),
태그 `v0.2.0`.
