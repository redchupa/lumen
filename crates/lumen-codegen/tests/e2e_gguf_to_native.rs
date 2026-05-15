//! The end-to-end Phase-5 demo:
//!
//! ```text
//!   in-memory GGUF buffer
//!     -> lumen-model::GgufFile parses header + tensor info
//!     -> tensor_data() gives us raw Q8_0 block bytes
//!     -> reinterpret as &[BlockQ8_0]
//!     -> Lumen native backend dequantizes via JIT
//!     -> result equals the pure-Rust reference dequant
//! ```
//!
//! This is the closed loop from "model on disk in GGUF format" to "fp32
//! values in a buffer, computed by Lumen's own machine code."

#![cfg(target_arch = "x86_64")]

use lumen_codegen::backend::{Backend, CodegenOpts};
use lumen_codegen::x86_64::X86_64;
use lumen_ir::ty::{DType, Dim, Shape, TensorType};
use lumen_ir::{Function, IrModule, Op, Value};
use lumen_jit::ExecRegion;
use lumen_model::gguf::{GgmlType, GgufFile, KvType};
use lumen_runtime::quant::{dequantize_q8_0, quantize_q8_0, BlockQ8_0, QK};

fn write_string(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(bytes);
}

fn write_gguf(name: &str, dims: &[u64], blocks: &[BlockQ8_0]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());

    write_string(&mut buf, "general.alignment");
    buf.extend_from_slice(&(KvType::U32 as u32).to_le_bytes());
    buf.extend_from_slice(&32u32.to_le_bytes());

    write_string(&mut buf, name);
    buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for d in dims {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    buf.extend_from_slice(&(GgmlType::Q8_0 as u32).to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());

    let pad = (32 - (buf.len() % 32)) % 32;
    buf.resize(buf.len() + pad, 0);

    let raw: &[u8] =
        unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, blocks.len() * 34) };
    buf.extend_from_slice(raw);
    buf
}

#[test]
fn gguf_q8_tensor_dequant_via_native_backend() {
    // 256 elements = 8 blocks. Shape [8, 32].
    let n = 256;
    let src: Vec<f32> = (0..n).map(|i| ((i % 23) as f32) * 0.013 - 0.15).collect();
    let mut blocks = vec![BlockQ8_0 { d: 0, qs: [0; 32] }; n / QK];
    quantize_q8_0(&src, &mut blocks);

    let gguf = write_gguf("blk.0.weight", &[8, 32], &blocks);
    let file = GgufFile::from_bytes(gguf).unwrap();
    let info = file.tensor("blk.0.weight").unwrap();
    assert_eq!(info.dtype, GgmlType::Q8_0);

    let raw = file.tensor_data("blk.0.weight").unwrap();
    let parsed_blocks: &[BlockQ8_0] =
        unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const BlockQ8_0, raw.len() / 34) };
    assert_eq!(parsed_blocks.len(), n / QK);

    // Build IR for dequant.
    let in_ty = TensorType {
        dtype: DType::Q8_0,
        shape: Shape(vec![Dim::Static(n as u32)]),
    };
    let out_ty = TensorType {
        dtype: DType::F32,
        shape: Shape(vec![Dim::Static(n as u32)]),
    };
    let mut f = Function::new("dq", vec![in_ty], out_ty.clone());
    let v_in = f.param_values[0];
    let v_out = f.push(Value {
        op: Op::Dequantize { x: v_in },
        ty: out_ty,
    });
    let placeholder = TensorType {
        dtype: DType::F32,
        shape: Shape(vec![]),
    };
    f.push(Value {
        op: Op::Return { value: v_out },
        ty: placeholder,
    });
    let ir = IrModule { functions: vec![f] };

    // Native dequant.
    let backend = X86_64::host();
    let mc = backend.lower(&ir, &CodegenOpts::default()).unwrap();
    let region = ExecRegion::from_machine_code(&mc).unwrap();
    let mut native_out = vec![0.0f32; n];
    unsafe {
        let f: unsafe extern "C" fn(*const BlockQ8_0, *mut f32) = region.as_fn();
        f(parsed_blocks.as_ptr(), native_out.as_mut_ptr());
    }

    // Reference dequant (pure Rust) for comparison.
    let mut want = vec![0.0f32; n];
    dequantize_q8_0(parsed_blocks, &mut want);

    for (i, (g, w)) in native_out.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-5,
            "idx {}: native {} reference {}",
            i,
            g,
            w
        );
    }
}
