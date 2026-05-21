# Lumen 빌드로그

사이드 프로젝트로 LLM 추론 컴파일러를 처음부터 짜는 회고 시리즈. 매 Phase 종료
시 한 편씩 작성. 측정-주도 의사결정 패턴을 정직하게 기록 — 가설이 부정된 경우도
인프라는 살리고 default-off로 유지.

## v0.1.0 — first working tokens

- [phase1 — DSL · 파서](./phase1-parser-ko.md) — Pratt 파서, AST, 타입 검사
- [phase2a — IR + C backend](./phase2a-ir-and-c-backend-ko.md) — SSA IR, reference oracle
- [phase2b — 자체 x86_64 backend](./phase2b-native-x86_64-ko.md) — 머신코드 emit, JIT 실행
- [phase3 — AVX2 자동 합성](./phase3-avx2-auto-synthesis-ko.md) — VEX 인코더, 8.7× scalar
- [phase3c — Register tile 4×8](./phase3c-register-tile-ko.md) — 4 독립 acc, 19× scalar
- [phase5c — Fused quant matmul](./phase5c-fused-quant-matmul-ko.md) — Q8×F32 자동 융합
- [phase6 — 첫 한국어 토큰](./phase6-first-korean-token-ko.md) — "안녕" → "안녕하세요, 저는"
- [phase7 — v0.1.0 end-to-end](./phase7-v0.1.0-end-to-end-ko.md) — 4.43 tok/s, ggml 9.3× slower

## v0.2.0 ~ v0.4.0 — kernel tuning, multi-thread, shape-aware

- [phase7d — v0.2.0 Q8-native](./phase7d-v0.2.0-q8-native-ko.md) — 17.97 tok/s (+3.5×)
- [phase7j — v0.3.0 profile + parallel](./phase7j-v0.3.0-profile-and-parallel-ko.md) — 60 tok/s @ 8t
- [phase7m_n — int dot honest](./phase7m-7n-int-dot-honest-ko.md) — VNNI 첫 시도, 부정 후 인프라 유지
- [phase7p — v0.4.0 shape-aware dispatch](./phase7p-v0.4.0-shape-aware-ko.md) — 65 tok/s, 1.39× gap

## v0.5.0 — measure-driven 11회 사이클

- [phase7t — Zen 4에서 AVX-512가 안 빠른 이유](./phase7t-zmm-zen4-honest-ko.md) — ZMM lane width 가설 부정, double-pumped 256-bit
- [phase7u/8a — 1.4× 격차 진단 + ThreadPool 재설계](./phase7u-8a-thread-scaling-memory-bw-ko.md) — 1t에서 Lumen +13% 발견, 격차는 memory bw 활용 효율
- [phase8d — Prefill batching이 답이 아니었던 이유](./phase8d-prefill-batching-honest-ko.md) — pp32 -2.9× 회귀, attention 1%, N>1 kernel 2.5× 느림
- [phase8d-8e1 — Prefill codegen story (English)](./phase8d-8e1-prefill-codegen-en.md) — 위 두 phase 영어 합본
- [v0.5 cycle 마무리](./v0.5-cycle-closing-ko.md) — 11회 측정-주도 사이클 전체 결산, maintenance mode 진입

## 패턴 정리

11회 사이클 누적 결과:

| Phase | 가설 | 결과 |
|---|---|---|
| 7.M | Q8 native int dot | net-neutral |
| 7.N | VNNI single-acc | -3.6% |
| 7.O | VNNI 4-acc | -2.7% (7.P 부활) |
| 7.R/S/T | AVX-512 ZMM lane width | -4.5% |
| 8.A | atomic ThreadPool | **+9% 1t, +3% 8t ✓** |
| 8.B | chunk L2-fit cap | -3.5% (revert) |
| 8.C | software prefetch | 8t -49% (revert) |
| 8.D.3 | prefill batching | -2.9× pp32 |
| 8.D.3 retro | attention is the cause | 1%만 (부정) |
| 8.D.5 | N>1 codegen은 느리다 | **확정 ✓** |
| 8.E.1 | N=1 fan-out 회복 | **+2.39× pp32 ✓** |

**positive**: 3 (8.A, 8.D.5 확정, 8.E.1) · **negative**: 7 · **net win**: 1.376× → 1.304×.

매번 인프라는 유지. 다른 호스트나 다른 phase에서 살아날 가능성 (실제로 7.O의
4-acc 인프라는 7.P에서 부활). 측정 없이 직관만으로 가면 11번 사이클 중 절반은
잘못된 방향이었을 것.

## 메타 패턴

- **commit message에 측정 없는 추측 적지 말기.** 다음 phase에서 따라잡힘 (8.D.3 → 8.D.4).
- **인프라 폐기보다 default-off**. 다른 호스트/조건에서 살아날 수 있음.
- **50줄 단위 측정 먼저**. 1주일 잘못된 phase보다 하루 micro-bench가 효율적.
- **회고 블로그**. 글로 풀면서 메타 패턴 발견. 다음 phase 결정 명확화.

---

전체 코드: [github.com/redchupa/lumen](https://github.com/redchupa/lumen) · 상세 phase: [PLAN.md](../../PLAN.md) · 다음 갈래: [ROADMAP.md](../../ROADMAP.md)
