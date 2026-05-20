# Lumen Roadmap

v0.5.0 (2026-05-21) 시점의 다음 마일스톤 후보. 각 항목은 사이드 프로젝트 호흡으로
1~3주 분량. 큰 의미 단위는 별도 release tag로 묶어 발행한다.

## v0.5.0 standing

- 단일 스레드에서 llama.cpp ggml 대비 +13% 우위 (Zen 4)
- 8 스레드에서 ggml 대비 1.304× 격차 (이전 v0.4.0: 1.376×)
- 11회 측정-주도 의사결정 사이클 (성공 1·회복 1·부정 9)
- 인프라 (JIT cache, ThreadPool, prefill plumbing, prefetch encoders, ZMM kernels)
  는 default-off 상태로 모두 살아 있음

## 다음 마일스톤 후보 (우선순위 순)

### 1. Phase 8.E.2 — Q8 N>1 matmul kernel codegen rewrite

**의미:** Phase 8.D.5에서 측정으로 확정. 현재 N>1 emit path는 N=1 kernel ×N보다
2.3~2.6× 느리다. Phase 7.G의 4-accumulator YMM 패턴 (FMA dependency chain 분리
+ 8-wide weight load via `vpmovsxbd`)을 N>1 path로 포팅하면 prefill에서 진짜
가속 가능.

**작업:** `crates/lumen-codegen/src/x86_64.rs:740` 의 `emit_quant_matmul_q8_body`
재작성. K-방향 4 sub-accumulator + N=8 column tile + 8-wide weight load.

**예상:** 3~5일. 성공 시 pp32 → 80+ tok/s (현재 54 tok/s), 8t 격차 1.30× → 1.1× 추정.

**위험:** AVX2의 single-lane broadcast 부재로 inner loop 디테일이 까다로움.
또 한 번 측정-주도 부정 가능성 25%.

### 2. ARM64 backend (PLAN.md Phase 2.C)

**의미:** Apple Silicon / Raspberry Pi / AWS Graviton 지원. x86_64만 지원하는
LLM 컴파일러는 흔하지 않음. 포트폴리오 차별점.

**작업:** ARM64 명령어 인코더 처음부터 짜기 (24-bit branch imm, LDR/STR scaled
offset, NEON fp32/i32 ops). x86_64 emit한 양만큼 (~3000 LOC) 다시 작성.

**예상:** 2주. 인프라부터 처음. AVX2 인코더 작업한 분량과 비슷.

**위험:** 새 아키텍처 첫 진입이라 디버깅 시간 큼. NEON이 AVX2와 의미 다른 부분
(예: signed multiply, byte permute) 처리.

### 3. Q4 native matmul (Q4_0, Q4_K)

**의미:** 현재 Q4_0 weight는 load 시점에 F32 dequant (`loader.rs:81`). 메모리
footprint Q8(670MB)과 동일. native Q4 matmul kernel 추가하면 weight 4× 더 작아져
더 큰 모델 (Qwen2.5-1.5B Q4, 3B Q4) 실행 가능. memory bw bound도 완화.

**작업:** Q4 dequant + matmul 통합 kernel (현재 Q8 fused matmul과 동일 패턴).
4-bit weight pack 해체 (low/high nibble) + scaling. Q4_K_M은 super-block 구조라
별도 처리 필요.

**예상:** Q4_0만 2주, Q4_K 추가 +1주. GGUF loader 확장도 함께.

**위험:** Q4_K_M의 super-block은 처음 짤 때 디버깅 까다로움.

### 4. Phase 6.D — flash-style attention

**의미:** 현재 multi_head_attention은 textbook naive Rust. 긴 context에서 O(seq²)
영향 큼. flash-attention 스타일 (tiled Q·K^T + online softmax)로 짜면 attention 자체
가속.

**작업:** attention codegen 또는 SIMD-친화적 Rust 재작성. JIT으로는 안 가도 됨
(매트멀이 95% 비중이라 attention 절대 비중 작음).

**예상:** 1주. ROI는 작음 (pp32 영향 1% 미만, decode 영향 3%).

**위험:** ROI 대비 작업량.

### 5. 다른 갈래

- **multi-thread prefill** (PLAN.md Phase 7.F) — prefill에서 layer 병렬화
- **continuous batching** (v1.1) — 동시 다중 시퀀스
- **speculative decoding** (v1.1) — draft model 가속
- **다른 모델** — EXAONE-3.5-2.4B, A.X-3.1-Lite 정답성 검증

## 외부 노출 / 포트폴리오

코드 완성도와 별개로 진행 가능:

- GitHub stars / 외부 PR 유도 (PLAN.md 성공 기준 §9)
- HackerNews / Reddit 글 — 측정-주도 회고 시리즈가 좋은 소재
- 기술 컨퍼런스 발표 — 한국 컴파일러 / 시스템 모임

## 비-목표 (PLAN.md §4)

- 학습/파인튜닝
- 멀티노드 분산 추론
- ROCm, Vulkan, WebAssembly 백엔드
- GUI 디버거

---

상세 phase 정의는 [PLAN.md](./PLAN.md) · 회고 시리즈는 [docs/blog/INDEX.md](./docs/blog/INDEX.md) · 측정 데이터는 [BENCHMARK.md](./BENCHMARK.md).
