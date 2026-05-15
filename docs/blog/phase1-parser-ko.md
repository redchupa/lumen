# Phase 1 — Lumen 파서: 텐서 형상이 타입이다

> Lumen 빌드로그 #1. 자체 LLM 추론 컴파일러를 처음부터 만드는 과정을 매 Phase마다 기록합니다.

이번 Phase의 목표는 단순했다.

```rust
fn matmul(
    a: tensor<f32, [64, 128]>,
    b: tensor<f32, [128, 32]>,
) -> tensor<f32, [64, 32]> {
    return a @ b;
}
```

이 코드를 **파싱하고**, **타입 검사**까지 통과시킨다. 끝나면 `lumen check` 한 줄로 "ok"가 떨어진다.

겉보기에 평범한 미니 언어지만, 한 가지 핵심 결정이 들어가 있다: **텐서 형상이 타입의 일부다.**

## 형상을 타입에 박으면 뭐가 좋은가

기존 그래프 컴파일러 대부분은 형상이 런타임 값이다. PyTorch에서 `x.shape`는 `torch.Size`고, ONNX의 형상도 추론 시점에 채워진다. 형상이 컴파일러에 정적으로 알려지지 않으면, 코드 생성기가 할 수 있는 최적화는 동적 디스패치 + 가드뿐이다.

Lumen은 정반대로 간다. `tensor<f32, [64, 128]>`는 `tensor<f32, [128, 32]>`와 다른 **타입**이다. 컴파일 시점에 `@` 연산자가 모양을 검사하고, 결과 타입을 합성한다. 이게 IR 단계에서 "Q4_0 × F16 matmul 4096×4096"을 보면 unpack + dequant + GEMM 융합 커널을 자동 emit하는 기반이 된다.

## 렉서: 한글 식별자가 공짜로 들어왔다

zero-copy 렉서를 만들었다. 토큰은 `Span(start, end)`만 들고 다닌다. 실제 텍스트는 원본 `&str`에서 슬라이스한다.

식별자 정책을 적을 때 잠깐 갈등했다 — ASCII만 받을까? `unicode_ident` crate를 쓸까? 결국 `char::is_alphabetic()`을 그대로 쓰기로 했다. 결과:

```rust
let toks = kinds("행렬곱");
assert_eq!(toks, vec![TokenKind::Ident, TokenKind::Eof]);
```

한글 식별자가 동작한다. 의도해서 만든 기능은 아니고 — 표준 라이브러리의 정의를 따랐을 뿐인데, 부수 효과로 따라왔다. Lumen이 "한국어 LLM 1급"을 표방한다는 점을 생각하면 우연 치고는 잘 맞다.

## Pratt 파서: 결합력의 미학

표현식 우선순위는 다음과 같이 정했다 (값이 클수록 강하게 묶임):

| 연산자 | left binding | right binding |
|---|---|---|
| 단항 `-` (prefix) | — | 40 |
| `@` (matmul) | 30 | 31 |
| `*`, `/` | 20 | 21 |
| `+`, `-` | 10 | 11 |

`right > left`로 만들면 좌결합이 된다. 즉 `a @ b @ c`는 `(a @ b) @ c`로 묶인다. 행렬곱은 결합법칙이 성립하지만, 캐시 관점에서 어느 쪽이 비용이 큰지가 형상에 달려 있어서 — 그 결정은 Phase 2의 IR pass에 넘긴다. 파서는 일관성만 보장하면 된다.

테스트 한 줄:

```rust
// `a + b @ c` should parse as `a + (b @ c)`.
```

이 한 줄을 통과시키려고 4년 묵은 정수 두 개를 굴려야 한다. 그게 Pratt 파서의 묘미다.

## 타입 검사: matmul 한 줄에 들어 있는 규칙들

`a @ b`에 대한 검사는 다음 다섯 가지를 본다:

1. `a`, `b` 둘 다 텐서인가
2. 둘 다 2-D인가
3. dtype이 같은가 (`f32 @ f16`은 거부)
4. `a.shape[-1] == b.shape[-2]`인가
5. 결과 형상을 `[a.shape[0], b.shape[1]]`로 합성

각 실패 케이스마다 별도의 에러 메시지를 만든다. 에러는 통과 못한 첫 검사 하나만 보고하는 게 아니다 — 함수 안의 모든 statement를 끝까지 돌면서 모은다. 한 번 컴파일에 가능한 모든 진단을 받는 게 개발 속도를 살린다.

## 진단 출력: rustc를 동경하며

```
error: matmul inner dims do not match: 128 vs 127
  --> examples/bad.lum:3:13
   |
 3 |     return a @ b;
   |             ^^^^^
```

rustc의 codespan 스타일을 흉내냈다. `Span::line_col()`이 소스 첫 글자부터 다시 세는 O(n) 함수인 점은 의도적이다. 에러 경로에서만 호출되고, 한 컴파일에 진단이 100개 넘어가는 일은 거의 없다. 빠른 경로(렉서, 파서)는 바이트 위치만 들고 다닌다.

## 숫자로 보는 결과물

- 1일 작업
- Rust 코드 ~1,100줄
- 단위 테스트 21개 (렉서 8 + 파서 7 + 타입 검사 5 + 진단 1)
- `cargo fmt`, `cargo clippy -D warnings` 통과
- CLI `lumen parse` / `lumen check` 동작
- 의존성: `thiserror` 하나만

## 다음

Phase 2 — IR과 코드 생성기. AST에서 SSA 형태의 IR로 내리고, x86_64와 ARM64에서 동작하는 첫 matmul 커널을 emit한다. 정답성은 ndarray reference와 비교한다. 성능은 Phase 3에서 본다.

레포는 https://github.com/redchupa/lumen 에서 공개되어 있다. PR과 이슈 환영.

— Claude와 페어 프로그래밍으로 작성.
