//! Abstract syntax tree. Shapes are part of types. Every node carries a span.

use crate::span::Span;

#[derive(Clone, Debug, Default)]
pub struct Module {
    pub items: Vec<Item>,
}

#[derive(Clone, Debug)]
pub enum Item {
    Function(Function),
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: Ident,
    pub params: Vec<Param>,
    pub ret: Type,
    pub body: Block,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: Ident,
    pub ty: Type,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Type {
    pub kind: TypeKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum TypeKind {
    Scalar(ScalarType),
    /// Tensor with element type and shape.
    Tensor {
        elem: ScalarType,
        shape: Shape,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScalarType {
    F16,
    F32,
    I32,
    Q4_0,
    Q8_0,
}

#[derive(Clone, Debug)]
pub struct Shape {
    pub dims: Vec<Dim>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Dim {
    pub kind: DimKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum DimKind {
    Static(u64),
    Dynamic, // resolved at JIT time; written `?` in source (future)
}

#[derive(Clone, Debug, Default)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum StmtKind {
    Let { name: Ident, value: Expr },
    Return(Expr),
    Expr(Expr),
}

#[derive(Clone, Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum ExprKind {
    Ident(String),
    IntLit(i64),
    FloatLit(f64),
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    BinOp {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Call {
        callee: Ident,
        args: Vec<Expr>,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    MatMul, // `@`
}

impl BinOp {
    pub fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::MatMul => "@",
        }
    }
}
