//! AST → IR lowering. Phase 2.
//!
//! Maps each typed AST function into an IR function. Each AST identifier (a
//! parameter or a `let` binding) becomes a [`ValueId`] in an environment
//! threaded through expression lowering.

use std::collections::HashMap;

use lumen_dsl::ast::{
    self, BinOp as AstBinOp, Block as AstBlock, Expr, ExprKind, Function as AstFn, Item, Module,
    ScalarType, Stmt, StmtKind, Type as AstType, TypeKind as AstTypeKind, UnaryOp,
};

use crate::module::{Function, IrModule, Value, ValueId};
use crate::op::Op;
use crate::ty::{DType, Dim, Shape, TensorType};

#[derive(thiserror::Error, Debug)]
pub enum LowerError {
    #[error(
        "internal: unresolved identifier `{0}` in lowering (should have been caught by typeck)"
    )]
    UnresolvedIdent(String),
    #[error("not yet supported in lowering: {0}")]
    Unsupported(&'static str),
}

pub fn lower(ast: &Module) -> Result<IrModule, LowerError> {
    let mut m = IrModule::default();
    for item in &ast.items {
        match item {
            Item::Function(f) => m.functions.push(lower_function(f)?),
        }
    }
    Ok(m)
}

fn lower_function(f: &AstFn) -> Result<Function, LowerError> {
    let param_tys: Vec<TensorType> = f.params.iter().map(|p| lower_type(&p.ty)).collect();
    let ret_ty = lower_type(&f.ret);
    let mut ir = Function::new(&f.name.name, param_tys, ret_ty);

    let mut env: HashMap<String, ValueId> = HashMap::new();
    for (p, &vid) in f.params.iter().zip(ir.param_values.iter()) {
        env.insert(p.name.name.clone(), vid);
    }

    lower_block(&mut ir, &mut env, &f.body)?;
    Ok(ir)
}

fn lower_block(
    ir: &mut Function,
    env: &mut HashMap<String, ValueId>,
    block: &AstBlock,
) -> Result<(), LowerError> {
    for stmt in &block.stmts {
        lower_stmt(ir, env, stmt)?;
    }
    Ok(())
}

fn lower_stmt(
    ir: &mut Function,
    env: &mut HashMap<String, ValueId>,
    stmt: &Stmt,
) -> Result<(), LowerError> {
    match &stmt.kind {
        StmtKind::Let { name, value } => {
            let v = lower_expr(ir, env, value)?;
            env.insert(name.name.clone(), v);
            Ok(())
        }
        StmtKind::Return(e) => {
            let v = lower_expr(ir, env, e)?;
            // The Return op itself doesn't produce a useful value; use rank-0
            // f32 as a placeholder type so the IR stays well-typed.
            let placeholder = TensorType {
                dtype: DType::F32,
                shape: Shape(vec![]),
            };
            ir.push(Value {
                op: Op::Return { value: v },
                ty: placeholder,
            });
            Ok(())
        }
        StmtKind::Expr(e) => {
            let _ = lower_expr(ir, env, e)?;
            Ok(())
        }
    }
}

fn lower_expr(
    ir: &mut Function,
    env: &mut HashMap<String, ValueId>,
    e: &Expr,
) -> Result<ValueId, LowerError> {
    match &e.kind {
        ExprKind::Ident(name) => env
            .get(name)
            .copied()
            .ok_or_else(|| LowerError::UnresolvedIdent(name.clone())),
        ExprKind::IntLit(v) => {
            let ty = TensorType {
                dtype: DType::I32,
                shape: Shape(vec![]),
            };
            Ok(ir.push(Value {
                op: Op::Constant { bits: *v as u128 },
                ty,
            }))
        }
        ExprKind::FloatLit(v) => {
            let bits = (*v as f32).to_bits() as u128;
            let ty = TensorType {
                dtype: DType::F32,
                shape: Shape(vec![]),
            };
            Ok(ir.push(Value {
                op: Op::Constant { bits },
                ty,
            }))
        }
        ExprKind::Unary {
            op: UnaryOp::Neg,
            expr: inner,
        } => {
            let v = lower_expr(ir, env, inner)?;
            let ty = ir.type_of(v).clone();
            // -x = 0 - x.  Emit Constant(0) and then a Mul by -1 is more work;
            // simpler: lower as `0 - x` once we have Sub. For now Phase 2 only
            // needs matmul, so reject other unary uses.
            Err(LowerError::Unsupported(
                if matches!(ty.dtype, DType::F32 | DType::I32) {
                    "unary `-` (will be added with Sub op)"
                } else {
                    "unary `-` on non-numeric type"
                },
            ))
        }
        ExprKind::BinOp { op, lhs, rhs } => {
            let l = lower_expr(ir, env, lhs)?;
            let r = lower_expr(ir, env, rhs)?;
            lower_binop(ir, *op, l, r)
        }
        ExprKind::Call { .. } => Err(LowerError::Unsupported("free function calls (Phase 6+)")),
    }
}

fn lower_binop(
    ir: &mut Function,
    op: AstBinOp,
    lhs: ValueId,
    rhs: ValueId,
) -> Result<ValueId, LowerError> {
    let l_ty = ir.type_of(lhs).clone();
    let r_ty = ir.type_of(rhs).clone();
    match op {
        AstBinOp::MatMul => {
            let result_ty = matmul_result_ty(&l_ty, &r_ty);
            Ok(ir.push(Value {
                op: Op::MatMul { lhs, rhs },
                ty: result_ty,
            }))
        }
        AstBinOp::Add => {
            assert_eq!(l_ty, r_ty, "typeck should have caught this");
            Ok(ir.push(Value {
                op: Op::Add { lhs, rhs },
                ty: l_ty,
            }))
        }
        AstBinOp::Mul => {
            assert_eq!(l_ty, r_ty, "typeck should have caught this");
            Ok(ir.push(Value {
                op: Op::Mul { lhs, rhs },
                ty: l_ty,
            }))
        }
        AstBinOp::Sub | AstBinOp::Div => Err(LowerError::Unsupported(
            "`-` and `/` ops (will be added in Phase 2.x)",
        )),
    }
}

fn matmul_result_ty(l: &TensorType, r: &TensorType) -> TensorType {
    // Typeck already validated rank-2 + inner dim match.
    let (m, _k) = static_2d_shape(l);
    let (_, n) = static_2d_shape(r);
    TensorType {
        dtype: l.dtype,
        shape: Shape(vec![Dim::Static(m), Dim::Static(n)]),
    }
}

fn static_2d_shape(t: &TensorType) -> (u32, u32) {
    debug_assert_eq!(t.shape.0.len(), 2);
    match (&t.shape.0[0], &t.shape.0[1]) {
        (Dim::Static(a), Dim::Static(b)) => (*a, *b),
        _ => panic!("dynamic dims not supported in Phase 2 lowering"),
    }
}

pub fn lower_type(t: &AstType) -> TensorType {
    match &t.kind {
        AstTypeKind::Scalar(s) => TensorType {
            dtype: scalar_to_dtype(*s),
            shape: Shape(vec![]),
        },
        AstTypeKind::Tensor { elem, shape } => TensorType {
            dtype: scalar_to_dtype(*elem),
            shape: Shape(
                shape
                    .dims
                    .iter()
                    .map(|d| match d.kind {
                        ast::DimKind::Static(v) => Dim::Static(v as u32),
                        ast::DimKind::Dynamic => Dim::Dynamic(0),
                    })
                    .collect(),
            ),
        },
    }
}

fn scalar_to_dtype(s: ScalarType) -> DType {
    match s {
        ScalarType::F16 => DType::F16,
        ScalarType::F32 => DType::F32,
        ScalarType::I32 => DType::I32,
        ScalarType::Q4_0 => DType::Q4_0,
        ScalarType::Q8_0 => DType::Q8_0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_dsl::Parser;

    #[test]
    fn lowers_matmul_function() {
        let src = r#"
            fn matmul(
                a: tensor<f32, [64, 128]>,
                b: tensor<f32, [128, 32]>,
            ) -> tensor<f32, [64, 32]> {
                return a @ b;
            }
        "#;
        let module = Parser::parse(src).unwrap();
        let ir = lower(&module).expect("lower");
        assert_eq!(ir.functions.len(), 1);
        let f = &ir.functions[0];
        assert_eq!(f.name, "matmul");
        assert_eq!(f.params.len(), 2);
        assert_eq!(f.param_values.len(), 2);
        // values: %0=Param(0), %1=Param(1), %2=MatMul, %3=Return
        assert_eq!(f.values.len(), 4);
        assert!(matches!(f.values[2].op, Op::MatMul { .. }));
        assert!(matches!(f.values[3].op, Op::Return { .. }));
    }
}
