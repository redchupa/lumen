//! IR verifier. Catches malformed IR before we hand it to the backend.
//!
//! Checks (Phase 2):
//!   - Every operand `ValueId` refers to a defined value.
//!   - Every operand is defined before use within the same block (SSA).
//!   - Function bodies end with exactly one `Return`.
//!   - Tensor dtype consistency for binary ops.
//!   - MatMul shape rule: rank-2, dtype match, inner dim match.

use crate::module::{Function, IrModule, ValueId};
use crate::op::Op;
use crate::ty::{Dim, TensorType};

#[derive(thiserror::Error, Debug)]
pub enum VerifyError {
    #[error("function `{0}`: undefined value %{1}")]
    UndefinedValue(String, u32),
    #[error("function `{0}`: value %{1} used before defined")]
    UseBeforeDef(String, u32),
    #[error("function `{0}` has no terminator (return)")]
    NoReturn(String),
    #[error("function `{0}` has multiple terminators")]
    MultipleReturns(String),
    #[error("function `{0}`: matmul %{1} requires rank-2 tensors")]
    MatMulRank(String, u32),
    #[error("function `{0}`: matmul %{1} dtype mismatch ({2:?} vs {3:?})")]
    MatMulDType(String, u32, crate::ty::DType, crate::ty::DType),
    #[error("function `{0}`: matmul %{1} inner dim mismatch ({2} vs {3})")]
    MatMulInner(String, u32, u32, u32),
    #[error("function `{0}`: binary op %{1} type mismatch ({2:?} vs {3:?})")]
    BinTypeMismatch(String, u32, TensorType, TensorType),
    #[error("function `{0}`: declared return type {1:?} but returns {2:?}")]
    ReturnTypeMismatch(String, TensorType, TensorType),
}

pub fn verify_module(m: &IrModule) -> Result<(), Vec<VerifyError>> {
    let mut errors = Vec::new();
    for f in &m.functions {
        verify_function(f, &mut errors);
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn verify_function(f: &Function, errors: &mut Vec<VerifyError>) {
    let mut return_count = 0usize;
    let mut last_return_value: Option<ValueId> = None;

    // Order in which values are defined (insertion order).
    let mut defined = vec![false; f.values.len()];

    for block in &f.blocks {
        for &vid in &block.values {
            let v = &f.values[vid.0 as usize];
            // Operands must be defined-before-use.
            for operand in operands(&v.op) {
                let idx = operand.0 as usize;
                if idx >= f.values.len() {
                    errors.push(VerifyError::UndefinedValue(f.name.clone(), operand.0));
                    continue;
                }
                if !defined[idx] {
                    errors.push(VerifyError::UseBeforeDef(f.name.clone(), operand.0));
                }
            }

            // Op-specific checks.
            match &v.op {
                Op::MatMul { lhs, rhs } => {
                    let l_ty = &f.values[lhs.0 as usize].ty;
                    let r_ty = &f.values[rhs.0 as usize].ty;
                    if l_ty.shape.0.len() != 2 || r_ty.shape.0.len() != 2 {
                        errors.push(VerifyError::MatMulRank(f.name.clone(), vid.0));
                    } else {
                        if l_ty.dtype != r_ty.dtype {
                            errors.push(VerifyError::MatMulDType(
                                f.name.clone(),
                                vid.0,
                                l_ty.dtype,
                                r_ty.dtype,
                            ));
                        }
                        if let (Dim::Static(lk), Dim::Static(rk)) =
                            (&l_ty.shape.0[1], &r_ty.shape.0[0])
                        {
                            if lk != rk {
                                errors.push(VerifyError::MatMulInner(
                                    f.name.clone(),
                                    vid.0,
                                    *lk,
                                    *rk,
                                ));
                            }
                        }
                    }
                }
                Op::Add { lhs, rhs } | Op::Mul { lhs, rhs } => {
                    let l_ty = &f.values[lhs.0 as usize].ty;
                    let r_ty = &f.values[rhs.0 as usize].ty;
                    if l_ty != r_ty {
                        errors.push(VerifyError::BinTypeMismatch(
                            f.name.clone(),
                            vid.0,
                            l_ty.clone(),
                            r_ty.clone(),
                        ));
                    }
                }
                Op::Return { value } => {
                    return_count += 1;
                    last_return_value = Some(*value);
                }
                _ => {}
            }

            defined[vid.0 as usize] = true;
        }
    }

    match return_count {
        0 => errors.push(VerifyError::NoReturn(f.name.clone())),
        1 => {
            if let Some(v) = last_return_value {
                let actual = f.values[v.0 as usize].ty.clone();
                if actual != f.ret {
                    errors.push(VerifyError::ReturnTypeMismatch(
                        f.name.clone(),
                        f.ret.clone(),
                        actual,
                    ));
                }
            }
        }
        _ => errors.push(VerifyError::MultipleReturns(f.name.clone())),
    }
}

fn operands(op: &Op) -> Vec<ValueId> {
    match op {
        Op::Param { .. } | Op::LoadWeight { .. } | Op::LoadInput { .. } | Op::Constant { .. } => {
            vec![]
        }
        Op::MatMul { lhs, rhs } | Op::Add { lhs, rhs } | Op::Mul { lhs, rhs } => {
            vec![*lhs, *rhs]
        }
        Op::Silu { x }
        | Op::Gelu { x }
        | Op::Softmax { x, .. }
        | Op::Reshape { x, .. }
        | Op::Transpose { x, .. }
        | Op::Dequantize { x }
        | Op::Quantize { x, .. } => vec![*x],
        Op::RmsNorm { x, weight, .. } => vec![*x, *weight],
        Op::Rope { x, positions, .. } => vec![*x, *positions],
        Op::Return { value } => vec![*value],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::lower;
    use lumen_dsl::Parser;

    #[test]
    fn verifies_matmul() {
        let src = r#"
            fn matmul(
                a: tensor<f32, [64, 128]>,
                b: tensor<f32, [128, 32]>,
            ) -> tensor<f32, [64, 32]> {
                return a @ b;
            }
        "#;
        let module = Parser::parse(src).unwrap();
        let ir = lower(&module).unwrap();
        verify_module(&ir).expect("verify");
    }
}
