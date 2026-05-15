# Phase 2.B — 우리가 짠 x86_64 머신코드가 처음으로 곱셈을 해냈다

> Lumen 빌드로그 #3.

오늘 일자로, Lumen은 자기 자신이 만든 x86_64 머신 코드를 메모리에 올리고 그걸 함수로 호출해서 `4×4 matmul`을 계산했다. 정답은 naive Rust 구현과 1e-4 이내로 일치했다.

이게 무엇을 의미하는가:
- LLVM, cranelift, cc — **외부 컴파일러 일체를 안 거치고** matmul을 native 코드로 만들었다.
- 우리 IR → 우리 인코더 → mmap → mprotect → 함수 포인터 → 호출 → 결과.

오늘 짠 코드 어디에도 "어셈블리 텍스트를 outsource 한다" 같은 우회는 없다. 각 명령어가 어떤 바이트로 인코딩되는지 직접 결정했다.

## ModR/M, SIB, REX — 가장 짜증나는 것부터

x86_64 명령어 인코딩은 1980년대 8086에서 시작해 30년간 패치된 시스템이다. 한 명령어가 다음을 가진다:

```
[prefixes] [REX] [opcode] [ModR/M] [SIB] [displacement] [immediate]
```

각 필드가 옵셔널. 각 필드의 의미가 명령어마다 다름. 단순한 `mov rax, rcx`도:
- REX 프리픽스 `0x48` (W=1)
- opcode `0x89`
- ModR/M `0xC8` (mod=11, reg=001=RCX, rm=000=RAX)

→ `48 89 C8`. 3바이트.

SSE는 또 다르다. `movss xmm0, [rcx]`:
- 프리픽스 `0xF3` (SSE single-precision marker)
- opcode `0x0F 0x10`
- ModR/M `0x81` (mod=10/disp32, reg=000=xmm0, rm=001=rcx)
- displacement 4바이트 `00 00 00 00`

→ `F3 0F 10 81 00 00 00 00`. 8바이트.

R8~R15 같은 확장 레지스터를 쓰면 REX 프리픽스의 R/X/B 비트를 추가로 켜야 한다. `movss [r8+rax*4]`처럼 SIB까지 쓰면 base, index 둘 다 REX 확장이 들어갈 수 있다.

이걸 다 처리하는 인코더 헬퍼를 `lumen-codegen/x86_64_enc.rs`에 만들었다. 9개 단위 테스트로 한 바이트씩 검증했다 — `objdump` 결과와 비교한 ground-truth.

## ABI는 두 종류, 한 함수 안에서 둘 다 동시에

Windows x64 ABI: 정수/포인터 인자가 RCX, RDX, R8, R9.
System V x86_64 ABI: RDI, RSI, RDX, RCX.

같은 머신, 같은 명령어인데 **OS가 어떤 레지스터를 약속의 자리로 잡았는지가 다르다.** Linux 바이너리를 Windows에 가져가면 함수 호출이 깨진다 — 같은 ABI일 거라고 가정하면.

Lumen은 둘 다 지원한다. `Abi::WinX64`와 `Abi::SysV`를 한 backend가 들고 있고, 함수 매개변수를 어디서 받을지만 다르게 mapping한다. 본문은 동일하다 (자기 클로버 레지스터로 즉시 copy한 뒤 작업).

```rust
let p_lhs = abi.param_reg(0);   // RCX on Win64, RDI on SysV
let p_rhs = abi.param_reg(1);   // RDX on Win64, RSI on SysV
let p_out = abi.param_reg(2);   // R8  on Win64, RDX on SysV

// 그 다음은 ABI 무관:
mov_rr(em, Reg::R12, p_lhs);  // callee-saved에 보관
mov_rr(em, Reg::R13, p_rhs);
mov_rr(em, Reg::R14, p_out);
```

Win64에서는 함수 prologue에 32바이트 "shadow space"를 stack에 잡아야 한다 — System V에는 없는 개념. `if abi == Win64 { 40 } else { 8 }`로 해결.

## JIT 실행 페이지 — W^X 원칙

머신코드 바이트를 만들었다고 끝이 아니다. 그걸 **실행 가능한 페이지**에 올려야 한다.

1. `VirtualAlloc(MEM_COMMIT, PAGE_READWRITE)` — RW로 페이지 할당 (Win) / `mmap(PROT_READ|PROT_WRITE)` (Unix)
2. `memcpy` — 바이트 복사
3. `VirtualProtect(PAGE_EXECUTE_READ)` — RW를 R-X로 전환 (Win) / `mprotect(PROT_READ|PROT_EXEC)` (Unix)
4. `FlushInstructionCache` — Win에서는 필수 (Unix x86_64는 무시 가능, ARM은 `dc cvau`+`ic ivau`+`dsb ish` 필요)

W^X (Write XOR Execute): 한 순간에 같은 페이지가 쓸 수 있으면서 실행 가능하면 안 된다. 보안 + 안정성. 모든 modern OS가 강제하는 원칙.

이 4단계가 100줄 unsafe Rust로 끝난다. `region`이나 `memmap` 같은 crate 없이, libc/Win32 함수만 직접 선언해서 호출. CLAUDE.md의 의존성 최소화 정책이 여기서도 이긴다.

## 출력 결과

8×8 matmul로 emit된 머신코드 크기: 약 130바이트.

`objdump`로 풀어보면:

```
0:   41 54                push   r12
2:   41 55                push   r13
4:   41 56                push   r14
6:   41 57                push   r15
8:   53                   push   rbx
9:   48 81 EC 28 00 00 00 sub    rsp,0x28
10:  4D 89 CC             mov    r12,rcx
13:  4D 89 D5             mov    r13,rdx
16:  4D 89 C6             mov    r14,r8
19:  4D 31 FF             xor    r15,r15
...
```

전통 x86 어셈블리. JIT 결과물이라 디버거에 함수 이름이 안 보이지만, 명령어 시퀀스는 손코딩한 것과 구분이 안 된다.

## 정답성: 컴파일러를 신뢰하는 두 번째 단계

Phase 2.A에서 C 백엔드의 정답성을 검증했다 (Lumen → C → gcc → 실행 = naive Rust). 그게 첫 번째 oracle이었다.

Phase 2.B에서는 자체 x86_64가 같은 oracle을 통과한다:

```rust
let module = Parser::parse(src)?;
let ir = lower(&module)?;
let backend = X86_64::host();
let mc = backend.lower(&ir, &opts)?;
let region = ExecRegion::from_machine_code(&mc)?;

let f: unsafe extern "C" fn(*const f32, *const f32, *mut f32) = region.as_fn();
f(a.as_ptr(), b.as_ptr(), out.as_mut_ptr());

let reference = naive_matmul(&a, &b, m, k, n);
for (got, want) in out.iter().zip(reference.iter()) {
    assert!((got - want).abs() < 1e-4);
}
```

이게 통과한다. 우리 컴파일러가 **두 가지 독립적인 백엔드**에서 같은 정답을 낸다.

## 숫자

- 1일 작업
- Rust 코드 +1,200줄 (encoder 400, backend 280, exec.rs 200, e2e 130, 그 외)
- 새 테스트 +13 (encoder 단위 9, exec.rs 1, abi 1, e2e 3)
- **총 40 tests green** (Phase 1: 21 + Phase 2.A: 6 + Phase 2.B: 13)
- `cargo fmt`, `cargo clippy --workspace -- -D warnings` 통과
- 새 외부 의존성: 0 (Win32/libc 함수는 직접 declare)

## 한계와 다음

지금 인코더가 emit하는 명령어:
- `mov`, `xor`, `add`, `sub`, `imul`, `inc`, `cmp`
- `movss`, `addss`, `mulss`, `xorps`
- `push`, `pop`, `ret`, `jmp`, `jcc`

스칼라만. AVX2가 들어오면 `vmovups`, `vmulps`, `vfmadd231ps` 같은 4-byte 인코딩 (VEX 프리픽스)이 추가된다. 한 번 더 큰 일.

Phase 3에서 그 일을 한다. SIMD 타일링 + 매크로커널 + ggml의 80~95% throughput 목표.

그 전에 Phase 2.C로 ARM64를 한 번 더 깐다. M2/M3에서 동작해야 LLM 추론 시연이 가능하다.

레포: https://github.com/redchupa/lumen — 별 부탁드립니다.

— Claude와 페어 프로그래밍으로 작성.
