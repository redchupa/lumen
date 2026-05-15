//! Text printer. Inspired by LLVM IR / MLIR.
//!
//! Example output:
//!
//! ```text
//! fn @matmul(%0: tensor<f32, [64, 128]>, %1: tensor<f32, [128, 32]>)
//!     -> tensor<f32, [64, 32]> {
//!   %2 = matmul %0, %1 : tensor<f32, [64, 32]>
//!   return %2
//! }
//! ```

use std::fmt::Write;

use crate::module::{Function, IrModule, ValueId};
use crate::op::Op;
use crate::ty::{DType, Dim, TensorType};

pub fn print_module(m: &IrModule) -> String {
    let mut out = String::new();
    for (i, f) in m.functions.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        print_function(&mut out, f);
    }
    out
}

pub fn print_function(out: &mut String, f: &Function) {
    write!(out, "fn @{}(", f.name).unwrap();
    for (i, pv) in f.param_values.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        write!(
            out,
            "{}: {}",
            val(*pv),
            format_type(&f.values[pv.0 as usize].ty)
        )
        .unwrap();
    }
    writeln!(out, ") -> {} {{", format_type(&f.ret)).unwrap();

    for block in &f.blocks {
        for vid in &block.values {
            let v = &f.values[vid.0 as usize];
            if matches!(v.op, Op::Param { .. }) {
                continue; // params already shown in signature
            }
            out.push_str("  ");
            print_value(out, *vid, v.ty.clone(), &v.op);
            out.push('\n');
        }
    }
    out.push_str("}\n");
}

fn print_value(out: &mut String, id: ValueId, ty: TensorType, op: &Op) {
    match op {
        Op::Return { value } => {
            write!(out, "return {}", val(*value)).unwrap();
        }
        Op::Param { index } => {
            write!(out, "{} = param {} : {}", val(id), index, format_type(&ty)).unwrap();
        }
        Op::LoadWeight { name } => {
            write!(
                out,
                "{} = load_weight @\"{}\" : {}",
                val(id),
                name,
                format_type(&ty)
            )
            .unwrap();
        }
        Op::LoadInput { index } => {
            write!(
                out,
                "{} = load_input {} : {}",
                val(id),
                index,
                format_type(&ty)
            )
            .unwrap();
        }
        Op::Constant { bits } => {
            write!(
                out,
                "{} = const 0x{:x} : {}",
                val(id),
                bits,
                format_type(&ty)
            )
            .unwrap();
        }
        Op::MatMul { lhs, rhs } => {
            write!(
                out,
                "{} = matmul {}, {} : {}",
                val(id),
                val(*lhs),
                val(*rhs),
                format_type(&ty)
            )
            .unwrap();
        }
        Op::Add { lhs, rhs } => {
            write!(
                out,
                "{} = add {}, {} : {}",
                val(id),
                val(*lhs),
                val(*rhs),
                format_type(&ty)
            )
            .unwrap();
        }
        Op::Mul { lhs, rhs } => {
            write!(
                out,
                "{} = mul {}, {} : {}",
                val(id),
                val(*lhs),
                val(*rhs),
                format_type(&ty)
            )
            .unwrap();
        }
        Op::Silu { x } => write!(out, "{} = silu {}", val(id), val(*x)).unwrap(),
        Op::Gelu { x } => write!(out, "{} = gelu {}", val(id), val(*x)).unwrap(),
        Op::Softmax { x, axis } => {
            write!(out, "{} = softmax {}, axis={}", val(id), val(*x), axis).unwrap();
        }
        Op::RmsNorm { x, weight, eps } => {
            write!(
                out,
                "{} = rms_norm {}, {}, eps={}",
                val(id),
                val(*x),
                val(*weight),
                eps
            )
            .unwrap();
        }
        Op::Rope { x, positions, base } => {
            write!(
                out,
                "{} = rope {}, {}, base={}",
                val(id),
                val(*x),
                val(*positions),
                base
            )
            .unwrap();
        }
        Op::Reshape { x, new_shape } => {
            write!(out, "{} = reshape {} to {:?}", val(id), val(*x), new_shape).unwrap();
        }
        Op::Transpose { x, perm } => {
            write!(out, "{} = transpose {} perm={:?}", val(id), val(*x), perm).unwrap();
        }
        Op::Dequantize { x } => {
            write!(
                out,
                "{} = dequantize {} : {}",
                val(id),
                val(*x),
                format_type(&ty)
            )
            .unwrap();
        }
        Op::Quantize { x, target } => {
            write!(
                out,
                "{} = quantize {} to {} : {}",
                val(id),
                val(*x),
                dtype_str(*target),
                format_type(&ty)
            )
            .unwrap();
        }
    }
}

fn val(id: ValueId) -> String {
    format!("%{}", id.0)
}

pub fn format_type(t: &TensorType) -> String {
    let dims = t
        .shape
        .0
        .iter()
        .map(|d| match d {
            Dim::Static(v) => v.to_string(),
            Dim::Dynamic(id) => format!("?{}", id),
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("tensor<{}, [{}]>", dtype_str(t.dtype), dims)
}

pub fn dtype_str(d: DType) -> &'static str {
    match d {
        DType::F16 => "f16",
        DType::F32 => "f32",
        DType::I32 => "i32",
        DType::Q4_0 => "q4_0",
        DType::Q8_0 => "q8_0",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::{Function, Value};
    use crate::ty::{Shape, TensorType};

    fn t(d: DType, dims: &[u32]) -> TensorType {
        TensorType {
            dtype: d,
            shape: Shape(dims.iter().map(|&v| Dim::Static(v)).collect()),
        }
    }

    #[test]
    fn prints_matmul_function() {
        let a_ty = t(DType::F32, &[64, 128]);
        let b_ty = t(DType::F32, &[128, 32]);
        let c_ty = t(DType::F32, &[64, 32]);
        let mut f = Function::new("matmul", vec![a_ty, b_ty], c_ty.clone());
        let a = f.param_values[0];
        let b = f.param_values[1];
        let c = f.push(Value {
            op: Op::MatMul { lhs: a, rhs: b },
            ty: c_ty,
        });
        f.push(Value {
            op: Op::Return { value: c },
            ty: TensorType {
                dtype: DType::F32,
                shape: Shape(vec![]),
            },
        });

        let mut out = String::new();
        print_function(&mut out, &f);
        assert!(out.contains("fn @matmul"));
        assert!(out.contains("matmul %0, %1"));
        assert!(out.contains("return %2"));
    }
}
