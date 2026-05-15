//! Throughput comparison: scalar vs AVX2 matmul kernels emitted by Lumen.
//!
//! Run: `cargo bench -p lumen-codegen --bench matmul_scalar_vs_avx2`
//!
//! Both kernels are JIT-emitted, so this measures **the kernel itself**, not
//! the compilation step.

#![cfg(target_arch = "x86_64")]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use lumen_codegen::backend::{Backend, CodegenOpts};
use lumen_codegen::x86_64::X86_64;
use lumen_dsl::Parser;
use lumen_ir::lower;
use lumen_jit::ExecRegion;

fn jit_compile(
    backend: &X86_64,
    src: &str,
) -> (
    ExecRegion,
    unsafe extern "C" fn(*const f32, *const f32, *mut f32),
) {
    let module = Parser::parse(src).unwrap();
    let ir = lower(&module).unwrap();
    let mc = backend.lower(&ir, &CodegenOpts::default()).unwrap();
    let region = ExecRegion::from_machine_code(&mc).unwrap();
    // SAFETY: signature matches the emitted function.
    let func: unsafe extern "C" fn(*const f32, *const f32, *mut f32) = unsafe { region.as_fn() };
    (region, func)
}

fn run_matmul(size: usize, backend: &X86_64, c: &mut Criterion, name: &str) {
    let src = format!(
        r#"
        fn matmul(
            a: tensor<f32, [{size}, {size}]>,
            b: tensor<f32, [{size}, {size}]>,
        ) -> tensor<f32, [{size}, {size}]> {{
            return a @ b;
        }}
        "#
    );
    let (_region, func) = jit_compile(backend, &src);

    let a: Vec<f32> = (0..size * size).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..size * size).map(|i| ((i + 7) as f32) * 0.001).collect();
    let mut out = vec![0.0f32; size * size];

    let flops = 2 * size * size * size; // M*N*K multiply-adds
    let mut group = c.benchmark_group(format!("matmul/{}", name));
    group.throughput(Throughput::Elements(flops as u64));
    group.bench_function(format!("{}x{}x{}", size, size, size), |bencher| {
        bencher.iter(|| unsafe {
            func(
                black_box(a.as_ptr()),
                black_box(b.as_ptr()),
                black_box(out.as_mut_ptr()),
            );
        });
    });
    group.finish();
}

// Build a backend that has AVX2 but never picks the 4x8 tile, by reshaping the
// IR so M % 4 != 0. We achieve this by adding a parameterized matmul function.
// Simpler: keep `host()` for the tile path; introduce a "1x8 only" config by
// adapting shapes for those benches.

fn bench_scalar_64(c: &mut Criterion) {
    run_matmul(64, &X86_64::scalar_only(), c, "scalar");
}

fn bench_tile_4x8_64(c: &mut Criterion) {
    // 64x64x64 → both M and N divisible → picks 4x8 tile.
    run_matmul(64, &X86_64::host(), c, "tile_4x8");
}

fn bench_scalar_128(c: &mut Criterion) {
    run_matmul(128, &X86_64::scalar_only(), c, "scalar");
}

fn bench_tile_4x8_128(c: &mut Criterion) {
    run_matmul(128, &X86_64::host(), c, "tile_4x8");
}

fn bench_tile_4x8_256(c: &mut Criterion) {
    run_matmul(256, &X86_64::host(), c, "tile_4x8");
}

criterion_group!(
    benches,
    bench_scalar_64,
    bench_tile_4x8_64,
    bench_scalar_128,
    bench_tile_4x8_128,
    bench_tile_4x8_256,
);
criterion_main!(benches);
