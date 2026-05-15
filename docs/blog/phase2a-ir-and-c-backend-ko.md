# Phase 2.A — IR과 C 백엔드, 그리고 첫 정답성

> Lumen 빌드로그 #2. 자체 LLM 추론 컴파일러를 처음부터 만드는 과정.

이번 Phase는 "Phase 2를 어떻게 쪼갤 것인가"라는 메타 결정으로 시작했다.

원래 PLAN은 "IR + x86_64 + ARM64 + matmul 정답성 검증"을 6주 안에 끝내자였다. 그런데 막상 들어가니 보였다. **x86_64를 처음부터 짜려면 IR이 안정되어 있어야 한다.** IR이 흔들리는 동안 native backend를 같이 짜면 둘 다 망가진다.

그래서 Phase 2를 둘로 쪼갰다:

- **Phase 2.A** (오늘): IR + C 백엔드. C 백엔드는 reference oracle로 동작한다.
- **Phase 2.B** (다음): 자체 x86_64 / ARM64 backend. 정답성 = C 백엔드 결과와 비교.

이렇게 하면 IR이 바뀔 때마다 자체 backend의 정답성을 빠르게 검증할 수 있다.

## 텐서 형상이 타입에 들어 있다는 게 IR 단계에서 의미하는 것

Phase 1에서 만든 AST는 다음을 가지고 있었다:

```rust
fn matmul(a: tensor<f32, [64, 128]>, b: tensor<f32, [128, 32]>)
    -> tensor<f32, [64, 32]> { return a @ b; }
```

이걸 IR로 내리면:

```text
fn @matmul(%0: tensor<f32, [64, 128]>, %1: tensor<f32, [128, 32]>)
    -> tensor<f32, [64, 32]> {
  %2 = matmul %0, %1 : tensor<f32, [64, 32]>
  return %2
}
```

겉보기에 평범하다. 그런데 이 `tensor<f32, [64, 128]>`가 LLVM IR의 `i32`나 ONNX의 `tensor`와 다른 점은 — **두 형상이 다른 텐서는 다른 타입**이라는 거다. `tensor<f32, [64, 128]>`와 `tensor<f32, [64, 127]>`는 호환되지 않는다.

이게 코드 생성기에 어떤 자유를 주냐면:

- 루프 경계가 컴파일 시점에 상수다 → 완전 언롤 가능
- 메모리 액세스 패턴이 정적이다 → 프리페치 거리 결정 가능
- 양자화 블록 경계가 형상에 종속이다 → unpack 코드 자동 생성 가능

Phase 3의 SIMD 타일링과 Phase 5의 양자화 융합이 이 기반 위에 올라간다.

## C 백엔드는 백엔드이자 시험관이다

```c
void lumen_matmul(const float *p0, const float *p1, float *result) {
  for (size_t i = 0; i < 64; ++i) {
    for (size_t j = 0; j < 32; ++j) {
      float acc = (float)0;
      for (size_t kk = 0; kk < 128; ++kk) {
        acc += p0[i * 128 + kk] * p1[kk * 32 + j];
      }
      result[i * 32 + j] = acc;
    }
  }
}
```

이 C 코드를 처음 본 사람은 "이게 뭐가 차별점이야?"라고 할 수 있다. 맞는 말이다. 이 C 코드 자체는 단순한 naive triple-loop이고, OpenBLAS는커녕 손코딩 SIMD에도 못 미친다.

차별점은 **이 C 코드를 누가 짰는가**다. 사람이 짠 게 아니다. **IR의 패턴 매처가 짰다.** `matmul %0, %1 : tensor<f32, [64, 32]>` 한 줄을 보고, 자동으로 i / j / kk 루프를 emit했다.

이게 무엇을 약속하는가:

- Phase 3에서 동일 IR을 입력으로 **AVX2 intrinsic**을 emit하는 백엔드를 짤 수 있다.
- Phase 5에서 `matmul %0, %1 : tensor<f32, q4_0 × f16>`을 입력하면 **unpack + dequant + matmul + requantize**가 하나로 융합된 커널이 자동으로 나온다.
- 두 백엔드는 **동일한 reference C 결과와 비교**해서 정답성을 검증한다.

llama.cpp는 양자화 × dtype × 형상 조합마다 사람이 손으로 커널을 짠다. Lumen은 한 패턴 매처가 모든 조합을 합성한다. 이게 정체성이다.

## e2e 정답성 — 컴파일러를 신뢰하는 첫걸음

```rust
let module = Parser::parse(src).unwrap();
let ir = lower(&module).unwrap();
let c_src = emit_c(&ir).unwrap();

// gcc -shared -O2 -o libkernel.so kernel.c
let lib_path = build_shared(cc, &c_path, tmp.path(), "lumen_kernel")?;

// dlopen 후 함수 호출, Rust naive matmul과 비교
unsafe {
    let lib = libloading::Library::new(&lib_path)?;
    let func: libloading::Symbol<Fn3> = lib.get(b"lumen_matmul")?;
    func(a.as_ptr(), b.as_ptr(), out.as_mut_ptr());
}

for (got, want) in out.iter().zip(reference.iter()) {
    assert!((got - want).abs() < 1e-4);
}
```

이 한 줄을 통과시키는 데 들어간 게:

1. Lumen 소스 → AST
2. AST → SSA IR
3. IR 검증 패스
4. IR → C99 소스 emit
5. gcc로 dynamic library 빌드
6. libloading으로 dlopen
7. C ABI로 호출
8. Rust naive 결과와 1e-4 이내 일치 검증

8단계가 한 번의 `cargo test`로 모두 검증된다. Phase 2.B에서 자체 x86_64 backend를 짤 때, 이 e2e 테스트가 그대로 oracle로 쓰인다.

## 의존성 정책

Lumen은 외부 의존성을 좁게 유지한다. `Cargo.toml` 전체 의존성:

- `thiserror` — 에러 derive 매크로
- `libloading` (dev-only) — e2e 테스트에서 dlopen
- `tempfile` (dev-only) — e2e 테스트에서 임시 dir

빌드 산출물에 들어가는 의존성은 `thiserror` 하나다. `tokio`, `serde`, `nom`, `clap`, `cc` — 다 없다. 컴파일러는 가벼워야 한다. 빌드가 느리면 반복이 느리고, 반복이 느리면 컴파일러는 끝없는 무덤이 된다.

## 이번 Phase 숫자

- 1일 작업
- Rust 코드 +900줄
- 단위 테스트 +6 (printer 1, lower 1, verifier 1, C backend 1, e2e 2)
- 총 **29 tests green**, fmt/clippy clean
- 새 CLI 서브커맨드 2개: `lumen ir`, `lumen compile-c`

## 다음 — Phase 2.B: 자체 x86_64 backend

목표:
1. x86_64 머신 코드 직접 emit (System V Linux + Windows x64 ABI 둘 다)
2. ARM64 머신 코드 직접 emit (AAPCS64)
3. e2e: 자체 backend = C backend = Rust naive (1e-4)

가장 어려운 부분은 ABI다. 함수 인자를 register에 어떻게 매핑하나, stack alignment는 어떻게 맞추나, prologue/epilogue는 어떻게 짜나. AVX는 Phase 3에서 들어온다. 지금은 스칼라 `mulss`, `addss`만으로도 충분하다.

레포: https://github.com/redchupa/lumen — 별 받아주시면 다음 Phase 동기부여.

— Claude와 페어 프로그래밍으로 작성.
