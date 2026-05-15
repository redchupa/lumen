# Lumen — 프로젝트 계획

## 1. 한 줄 정의

**IR이 양자화 커널을 자동 합성하는** LLM 추론 컴파일러 + 런타임. 한국어 LLM 정답성 1급 보장.

## 2. 목적 (왜 만드는가)

1. **시장가치**: AI 추론 엔진 + 컴파일러 교집합 인력은 한국에 100명 미만. 평균 연봉 1.5억+.
2. **기술적 차별점**: llama.cpp는 양자화×dtype×형상 조합별로 손으로 짠 커널 컬렉션(수백 함수). Lumen은 IR 패턴 매칭 + 자동 융합으로 코드 합성.
3. **포트폴리오**: "컴파일러 만들었어요" + "LLM 런타임 만들었어요" + "양자화 자동 합성" 세 문장이 한 레포에 박힘.

## 3. 표적 사용자

- 한국어 LLM을 로컬에서 굴리려는 개발자
- llama.cpp의 코드 생성을 자동화하고 싶은 연구자
- 양자화·커널 디스패치 학습용 레퍼런스 코드를 찾는 학생

## 4. 범위

### IN (v1.0) — CPU only

- DSL `*.lum` 파일 → IR → 네이티브 실행
- 표적 아키: **x86_64 (AVX2/AVX-512)**, **ARM64 (NEON)**
- 양자화: FP16, INT8, INT4 (GGUF Q4_0/Q4_K/Q8_0 호환)
- 모델: Qwen2.5-0.5B/1.5B/3B, EXAONE-3.5-2.4B, A.X-3.1-Lite
- CLI: `lumen run model.gguf "prompt"`
- 벤치: vs llama.cpp on M2/M3, x86 Linux

### v1.1 (CPU 완성도 + 시그니처 기능)

- Speculative decoding (draft model 2~3× 가속)
- AVX-512 BF16 instructions
- Continuous batching (단일 프로세스 다중 시퀀스)
- Q4_K, Q5_K, Q6_K 추가

### v1.5 (GPU)

- CUDA 백엔드 (자체 PTX emit)
- Metal 백엔드 (M 시리즈 GPU)

### OUT (v1.x 전체)

- 학습/파인튜닝
- 멀티노드 분산 추론
- ROCm, Vulkan 백엔드
- WebAssembly 백엔드

## 5. 마일스톤

### Phase 0 — 설계 (1주)

- [x] 워크스페이스 + 디렉토리 골격
- [ ] `docs/ARCHITECTURE.md` 초안
- [ ] crate 분할 결정
- [ ] CI 골격 (cargo test, fmt, clippy)
- **검증**: `cargo build` 통과, 빈 워크스페이스 컴파일 성공

### Phase 1 — DSL · 파서 (3주)

- [ ] 렉서 (zero-copy, span 보존)
- [ ] Pratt 파서
- [ ] AST + visitor
- [ ] 타입 시스템 (텐서 형상이 타입의 일부)
- [ ] 진단(diagnostic) 출력 (codespan 스타일)
- **검증**: `examples/matmul.lum` 파싱 → AST 덤프 성공

### Phase 2 — IR · 코드 생성 (6주)

- [ ] SSA IR 정의 (Op, Value, Block)
- [ ] AST → IR 로어링
- [ ] 코드 생성기 백엔드 트레이트
- [ ] x86_64 (스칼라부터)
- [ ] ARM64 (스칼라부터)
- [ ] matmul 정답성 테스트 (vs ndarray)
- **검증**: 8x8x8 fp32 matmul 정답성, 단위 테스트 통과

### Phase 3 — SIMD 최적화 (4주)

- [ ] AVX2 intrinsic 출력 (Rust `std::arch`)
- [ ] NEON intrinsic 출력
- [ ] 캐시 타일링 (L1/L2/L3 기반)
- [ ] 매크로커널 자동 생성 (M × N × K 분해)
- **검증**:
  - GEMM FP32가 **llama.cpp의 ggml 대비 80~95%** throughput (현실적 목표)
  - 양자화 GEMM(Q4_0 × F16)이 ggml 대비 80%+ throughput
  - 참고: OpenBLAS 절대 비교는 v1.x 범위 밖. 자체 합성 vs 손코딩 비교만 한다.

### Phase 4 — JIT 엔진 (4주)

- [ ] 런타임 코드 생성 (mmap + W^X)
- [ ] 입력 형상 기반 특화
- [ ] 핫스팟 재컴파일 (counter 기반)
- [ ] 코드 캐시
- **검증**: 첫 호출 컴파일 < 50ms, 재호출 < 1μs 오버헤드

### Phase 5 — 양자화 (3주)

- [ ] GGUF 로더 (Q4_0, Q4_K, Q8_0)
- [ ] 가중치 → 내부 IR 텐서
- [ ] INT8 GEMM 커널
- [ ] INT4 unpack + GEMM
- **검증**: Qwen2.5-0.5B-Q4_0 가중치 로드 후 forward 1회 통과

### Phase 6 — LLM 추론 (6주)

- [ ] 토크나이저 (BPE, sentencepiece)
- [ ] RoPE, RMSNorm, SiLU, GQA
- [ ] KV 캐시 (paged-attention 영감)
- [ ] 샘플링 (top-k, top-p, temperature)
- [ ] CLI `lumen run`
- **검증**: Qwen2.5-0.5B로 "안녕하세요" 입력 → 문법적 한국어 출력

### Phase 7 — 벤치 · 글 (4주)

- [ ] vs llama.cpp throughput/latency (단일 배치, 배치 8, 16)
- [ ] vs llama.cpp 메모리 사용량
- [ ] M2/M3, x86 Linux 두 환경 측정
- [ ] 기술 블로그 5편 (DSL · IR · SIMD · JIT · 벤치)
- **검증**: 최소 한 가지 측면에서 llama.cpp와 동등 또는 우월

## 6. 비기능 요구사항

- 빌드: `cargo build --release` 3분 이내
- 의존성: Rust 표준 라이브러리 + `cc` (C 호출용)만. v1.5 GPU 단계에서 `cuda-sys` 검토.
- 라이선스: Apache-2.0
- 코드 스타일: `rustfmt` 기본, `clippy -D warnings`
- 테스트 커버리지: 핵심 IR/codegen 80%+

## 6.5. 테스트 전략

컴파일러 + LLM은 정답성이 전부. 다음 4계층:

### 6.5.1 단위 테스트 (`#[test]`)
- 렉서: 키워드/리터럴/스팬 정확성
- 파서: 골든 AST 비교 (`tests/snapshots/`)
- IR: 각 op의 타입 추론 정합성
- Codegen: 디스어셈블된 바이트 → 기대 명령어

### 6.5.2 디퍼런셜 정답성 테스트
- **Reference**: `ndarray` 또는 numpy 산출 결과를 `tests/fixtures/*.bin`에 저장
- 각 IR op 별로 random input + reference output 비교
- 허용 오차: FP32 1e-5, FP16 1e-3, Q4_0 1e-2 (logit 기준)

### 6.5.3 LLM end-to-end
- Qwen2.5-0.5B로 고정 프롬프트 + 시드 → 토큰 시퀀스 결정성
- llama.cpp의 같은 모델·시드·프롬프트 출력과 logit 비교
- 허용: top-5 token 집합 일치, top-1 logit 절대 차이 < 0.05

### 6.5.4 회귀 벤치
- `cargo bench` (criterion)
- GitHub Actions PR마다 자동 실행, baseline 대비 -10% 이상 회귀 시 fail

### 6.5.5 fixtures 디렉토리
```
tests/fixtures/
├── matmul_64x128x32_f32.bin   # input + reference output
├── rmsnorm_4096_f16.bin
├── rope_1x128x32x128.bin
└── qwen25_0.5b_logits_seed42.bin
```

## 6.6. 명명 규칙 (RFC 0001 예정)

- `Lumen`이라는 이름은 v0.1.0 릴리즈 전 GitHub 검색 충돌 확인 후 확정
- 백업 후보: `Iro`, `Sori`, `Karak` (한국어 어휘)

## 7. 리스크 & 완화

| 리스크 | 영향 | 완화 |
|---|---|---|
| GEMM 성능이 llama.cpp ggml 못 따라감 | 벤치에서 망함 | 1) 80% 도달 시 합격선, 2) 자동 합성이라는 메타 차별점으로 보완, 3) 양자화 변환 융합 같은 ggml이 안 하는 최적화로 역전 시도 |
| 자체 코드 생성이 LLVM 못 따라감 | 일반 코드 느림 | LLVM 의존을 거부하는 게 정체성. 그 대신 표적 범위(텐서 연산만)를 좁혀 비교 우위 확보 |
| 양자화 정확도 손실 | 출력 품질 저하 | llama.cpp 동일 양자화 결과와 logit 일치성 테스트 (§6.5.3) |
| 한국어 토크나이저 호환성 이슈 | LLM 출력 깨짐 | HuggingFace tokenizers crate를 dev-dep reference로 두고 정답성 비교 |
| KV 캐시 OOM | 큰 모델/긴 컨텍스트에서 죽음 | 페이지드 어텐션 영감 (§ARCHITECTURE 3.5), 명시적 max_tokens 강제 |
| 풀타임 40h+ 페이스 유지 실패 | 일정 1.5~2배 | 매주 일요일 PLAN.md 진척도 자가 점검. 2주 연속 미진척 시 범위 축소 |
| 외부 PR 받아본 적 없는 거 들킴 | 면접 약점 | Phase 2 종료 후 "good first issue" 라벨 적극 부여, 한국 Rust 커뮤니티에 공유 |

## 8. 작업 방식 (Claude 페어 프로그래밍)

- Claude가 짜고 사용자가 읽고 자기 손으로 다시 씀
- 매 Phase 종료 시 **기술 블로그 1편** 필수
- 매 마일스톤 종료 시 **README의 Roadmap 표** 업데이트
- 모든 커밋은 GPG 서명, "Co-Authored-By: Claude" 명시

## 9. 성공 기준

- [ ] Qwen2.5-0.5B를 Lumen으로 끝까지 추론 성공
- [ ] vs llama.cpp 어떤 차원이든 우월한 결과 1건 이상
- [ ] GitHub 스타 200+
- [ ] 기술 블로그 5편 누적 조회 1만+
- [ ] 외부 PR 또는 이슈 10건+ (관심 증명)
