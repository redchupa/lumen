//! Type checker.
//!
//! Inference rules (Phase 1):
//!   - integer literal → `i32`
//!   - float literal   → `f32`
//!   - `a + b`, `a - b`, `a * b`, `a / b`:  types must be equal; result = lhs
//!     (broadcasting deferred to Phase 2+).
//!   - `a @ b` (matmul): both must be 2D tensors, `a.shape[1] == b.shape[0]`,
//!     element types must match, result is `tensor<elem, [a.shape[0], b.shape[1]]>`.
//!   - Function return type must match the actual return expression's type.

use std::collections::HashMap;

use crate::ast::*;
use crate::diagnostic::Diagnostic;
use crate::span::Span;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckedType {
    Scalar(ScalarType),
    Tensor { elem: ScalarType, shape: Vec<u64> },
}

impl CheckedType {
    fn display(&self) -> String {
        match self {
            CheckedType::Scalar(s) => scalar_name(*s).to_string(),
            CheckedType::Tensor { elem, shape } => {
                let dims = shape
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("tensor<{}, [{}]>", scalar_name(*elem), dims)
            }
        }
    }
}

fn scalar_name(s: ScalarType) -> &'static str {
    match s {
        ScalarType::F16 => "f16",
        ScalarType::F32 => "f32",
        ScalarType::I32 => "i32",
        ScalarType::Q4_0 => "q4_0",
        ScalarType::Q8_0 => "q8_0",
    }
}

pub struct TypeChecker {
    errors: Vec<Diagnostic>,
}

impl TypeChecker {
    pub fn new() -> Self {
        Self { errors: Vec::new() }
    }

    pub fn check(module: &Module) -> Result<(), Vec<Diagnostic>> {
        let mut tc = TypeChecker::new();
        for item in &module.items {
            match item {
                Item::Function(f) => tc.check_function(f),
            }
        }
        if tc.errors.is_empty() {
            Ok(())
        } else {
            Err(tc.errors)
        }
    }

    fn check_function(&mut self, f: &Function) {
        let mut env: HashMap<String, CheckedType> = HashMap::new();
        for p in &f.params {
            let ty = lower_type(&p.ty);
            env.insert(p.name.name.clone(), ty);
        }
        let expected_ret = lower_type(&f.ret);

        for (i, stmt) in f.body.stmts.iter().enumerate() {
            let is_last = i + 1 == f.body.stmts.len();
            self.check_stmt(stmt, &mut env, &expected_ret, is_last, &f.name.name);
        }
    }

    fn check_stmt(
        &mut self,
        stmt: &Stmt,
        env: &mut HashMap<String, CheckedType>,
        expected_ret: &CheckedType,
        _is_last: bool,
        fn_name: &str,
    ) {
        match &stmt.kind {
            StmtKind::Let { name, value } => {
                if let Some(ty) = self.check_expr(value, env) {
                    env.insert(name.name.clone(), ty);
                }
            }
            StmtKind::Return(e) => {
                if let Some(ty) = self.check_expr(e, env) {
                    if &ty != expected_ret {
                        self.errors.push(Diagnostic::error(
                            format!(
                                "function `{}` declared return type {} but returns {}",
                                fn_name,
                                expected_ret.display(),
                                ty.display()
                            ),
                            e.span,
                        ));
                    }
                }
            }
            StmtKind::Expr(e) => {
                let _ = self.check_expr(e, env);
            }
        }
    }

    fn check_expr(
        &mut self,
        expr: &Expr,
        env: &HashMap<String, CheckedType>,
    ) -> Option<CheckedType> {
        match &expr.kind {
            ExprKind::IntLit(_) => Some(CheckedType::Scalar(ScalarType::I32)),
            ExprKind::FloatLit(_) => Some(CheckedType::Scalar(ScalarType::F32)),
            ExprKind::Ident(name) => match env.get(name) {
                Some(t) => Some(t.clone()),
                None => {
                    self.errors.push(Diagnostic::error(
                        format!("unknown identifier `{}`", name),
                        expr.span,
                    ));
                    None
                }
            },
            ExprKind::Unary {
                op: UnaryOp::Neg,
                expr: inner,
            } => self.check_expr(inner, env),
            ExprKind::BinOp { op, lhs, rhs } => {
                let l = self.check_expr(lhs, env)?;
                let r = self.check_expr(rhs, env)?;
                self.check_binop(*op, &l, &r, lhs.span.merge(rhs.span))
            }
            ExprKind::Call { callee, args } => {
                for a in args {
                    let _ = self.check_expr(a, env);
                }
                self.errors.push(Diagnostic::error(
                    format!(
                        "call to `{}`: free function calls not implemented yet",
                        callee.name
                    ),
                    expr.span,
                ));
                None
            }
        }
    }

    fn check_binop(
        &mut self,
        op: BinOp,
        l: &CheckedType,
        r: &CheckedType,
        span: Span,
    ) -> Option<CheckedType> {
        match op {
            BinOp::MatMul => self.check_matmul(l, r, span),
            _ => {
                if l == r {
                    Some(l.clone())
                } else {
                    self.errors.push(Diagnostic::error(
                        format!(
                            "operator `{}` requires matching types, got {} and {}",
                            op.symbol(),
                            l.display(),
                            r.display()
                        ),
                        span,
                    ));
                    None
                }
            }
        }
    }

    fn check_matmul(
        &mut self,
        l: &CheckedType,
        r: &CheckedType,
        span: Span,
    ) -> Option<CheckedType> {
        let (le, ls) = match l {
            CheckedType::Tensor { elem, shape } => (*elem, shape),
            _ => {
                self.errors.push(Diagnostic::error(
                    format!("matmul LHS must be a tensor, got {}", l.display()),
                    span,
                ));
                return None;
            }
        };
        let (re, rs) = match r {
            CheckedType::Tensor { elem, shape } => (*elem, shape),
            _ => {
                self.errors.push(Diagnostic::error(
                    format!("matmul RHS must be a tensor, got {}", r.display()),
                    span,
                ));
                return None;
            }
        };
        if le != re {
            self.errors.push(Diagnostic::error(
                format!(
                    "matmul element types differ: {} vs {}",
                    scalar_name(le),
                    scalar_name(re)
                ),
                span,
            ));
            return None;
        }
        if ls.len() != 2 || rs.len() != 2 {
            self.errors.push(Diagnostic::error(
                format!(
                    "matmul requires 2-D tensors; got rank {} and rank {}",
                    ls.len(),
                    rs.len()
                ),
                span,
            ));
            return None;
        }
        if ls[1] != rs[0] {
            self.errors.push(Diagnostic::error(
                format!("matmul inner dims do not match: {} vs {}", ls[1], rs[0]),
                span,
            ));
            return None;
        }
        Some(CheckedType::Tensor {
            elem: le,
            shape: vec![ls[0], rs[1]],
        })
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

fn lower_type(t: &Type) -> CheckedType {
    match &t.kind {
        TypeKind::Scalar(s) => CheckedType::Scalar(*s),
        TypeKind::Tensor { elem, shape } => CheckedType::Tensor {
            elem: *elem,
            shape: shape
                .dims
                .iter()
                .map(|d| match d.kind {
                    DimKind::Static(v) => v,
                    DimKind::Dynamic => 0, // sentinel; Phase 1 has no dynamic dims in source.
                })
                .collect(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    fn check(src: &str) -> Result<(), Vec<Diagnostic>> {
        let m = Parser::parse(src).expect("parse");
        TypeChecker::check(&m)
    }

    #[test]
    fn matmul_ok() {
        let src = r#"
            fn matmul(
                a: tensor<f32, [64, 128]>,
                b: tensor<f32, [128, 32]>,
            ) -> tensor<f32, [64, 32]> {
                return a @ b;
            }
        "#;
        check(src).expect("typecheck");
    }

    #[test]
    fn matmul_inner_mismatch() {
        let src = r#"
            fn m(
                a: tensor<f32, [64, 128]>,
                b: tensor<f32, [64, 32]>,
            ) -> tensor<f32, [64, 32]> {
                return a @ b;
            }
        "#;
        let errs = check(src).unwrap_err();
        assert!(errs.iter().any(|e| e.message.contains("inner dims")));
    }

    #[test]
    fn matmul_dtype_mismatch() {
        let src = r#"
            fn m(
                a: tensor<f32, [4, 4]>,
                b: tensor<f16, [4, 4]>,
            ) -> tensor<f32, [4, 4]> {
                return a @ b;
            }
        "#;
        let errs = check(src).unwrap_err();
        assert!(errs.iter().any(|e| e.message.contains("element types")));
    }

    #[test]
    fn return_type_mismatch() {
        let src = r#"
            fn m(a: tensor<f32, [4, 4]>) -> tensor<f32, [8, 8]> {
                return a;
            }
        "#;
        let errs = check(src).unwrap_err();
        assert!(errs.iter().any(|e| e.message.contains("return type")));
    }

    #[test]
    fn unknown_identifier() {
        let src = r#"
            fn m(a: f32) -> f32 {
                return zz;
            }
        "#;
        let errs = check(src).unwrap_err();
        assert!(errs
            .iter()
            .any(|e| e.message.contains("unknown identifier")));
    }
}
