# CLAUDE.md — Lumen

이 파일은 Claude Code(또는 다른 AI 에이전트)가 이 레포에서 작업할 때 따라야 할 규칙입니다.

## 0. 프로젝트 한 줄

자체 DSL · IR · JIT 컴파일러로 LLM 추론을 처음부터 짠다. 한국어 LLM 1급.

## 1. 코드 작성 원칙

### Rust 코드
- `rustfmt` 기본값
- `clippy -D warnings` 통과
- `unsafe`는 모듈 단위로 격리, 반드시 `// SAFETY:` 주석
- `unwrap()`/`expect()`는 테스트와 `examples/`에서만 허용. 프로덕션 경로는 `Result` 반환
- 의존성 추가 시 PLAN.md의 "비기능 요구사항" 위반 여부 먼저 확인

### C 코드 (FFI)
- C11
- 헤더는 `crates/*/include/`에 둠
- 빌드는 `cc` crate로 `build.rs`에서 처리
- `#define` 매크로 최소화, `static inline` 선호

## 2. 디렉토리 규칙

```
lumen/
├── crates/
│   ├── lumen-dsl/        # 렉서, 파서, AST, 타입
│   ├── lumen-ir/         # SSA IR, 패스 매니저
│   ├── lumen-codegen/    # 백엔드별 코드 생성
│   │   ├── src/x86_64.rs
│   │   ├── src/arm64.rs
│   │   └── src/cuda.rs
│   ├── lumen-jit/        # 런타임 컴파일, 코드 캐시
│   ├── lumen-runtime/    # 텐서, 메모리 풀, 디스패치
│   ├── lumen-model/      # GGUF 로더, 토크나이저
│   └── lumen-cli/        # `lumen` 바이너리
├── docs/                 # ARCHITECTURE.md, RFC들
├── benches/              # criterion 벤치
├── examples/             # *.lum 예제 + 사용 예
└── .github/workflows/    # CI
```

## 3. 작업 워크플로우

1. **새 기능 시작 전 PLAN.md 확인** — 어느 Phase에 해당하는지 확인하고 거기서 시작
2. **타입 먼저** — 함수 본문보다 시그니처와 타입을 먼저 확정
3. **테스트 먼저는 강요하지 않음** — 다만 IR/codegen 변경은 정답성 테스트가 함께 들어와야 함
4. **벤치는 Phase 3부터** — 그 전엔 정답성 우선

## 4. 커밋 규칙

- 한 커밋 = 한 논리 단위. 섞지 말 것
- 메시지 형식: `[crate] 한 줄 요약`
  - 예: `[lumen-ir] add SSA Value type`
- AI 협업 명시: 본문에 `Co-Authored-By: Claude <noreply@anthropic.com>`

## 5. 금지 사항

- **외부 LLM 추론 엔진 의존 금지** — ggml, ONNX Runtime, candle, mlx 등을 `lumen-runtime`에서 임포트하지 않는다. 비교 목적의 `benches/` 폴더에서만 허용.
- **거대 의존성 금지** — `tokio`, `serde`(필요 최소), `rayon` 정도까지만. 그 외는 PR에서 토론.
- **개인정보·시크릿 커밋 금지** — API 키, 가족 실명, 자녀 데이터, 보유 종목 등 평문 포함 금지. HuggingFace 토큰은 `.env`로만.
- **`.storage/` 류 자동생성 파일 직접 편집 금지** — 빌드 산출물은 항상 빌드로 재생성

## 6. 모델 / 데이터 정책

- 모델 파일은 레포에 커밋하지 않음. `examples/`에 다운로드 스크립트만 둠.
- 테스트용 미니 가중치(< 10MB)는 `tests/fixtures/`에 허용.
- 토큰화 정답성 비교에는 `tokenizers` crate를 dev-dependency로 둘 수 있음.

## 7. 글쓰기

- 매 Phase 종료 시 `docs/blog/phaseN-*.md` 작성
- 한국어 1편, 영어 1편을 한 세트로 (둘 다 같은 내용일 필요는 없음, 청중에 맞춰 톤 변경)
- 코드 스니펫은 모두 컴파일 가능해야 함 (`cargo test --doc` 통과)

## 8. 외부 사용자 응대 (오픈된 이후)

- 이슈/PR 응답은 24시간 이내
- "이거 어떻게 작동해요?" 질문에 답할 수 없는 코드는 머지하지 않음
- 컨트리뷰션 가이드: `CONTRIBUTING.md`(추후 작성)

## 9. 참고 자료

- Crafting Interpreters (Bob Nystrom)
- Engineering a Compiler (Cooper & Torczon)
- High Performance Computing (Severance, O'Reilly)
- llama.cpp 소스 (양자화·KV 캐시 참고)
- ggml 소스 (텐서 표현 참고)
- tinygrad (자동 합성 아이디어 참고)
- Triton 논문 (DSL 영감)
