//! Abstract syntax tree. Shapes are part of types.

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
    pub name: String,
    pub params: Vec<Param>,
    pub ret: Type,
    pub body: Block,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub ty: Type,
}

#[derive(Clone, Debug)]
pub enum Type {
    Scalar(ScalarType),
    /// Tensor with element type and (possibly dynamic) shape.
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
pub struct Shape(pub Vec<Dim>);

#[derive(Clone, Debug)]
pub enum Dim {
    Static(u32),
    Dynamic, // resolved at JIT time
}

#[derive(Clone, Debug, Default)]
pub struct Block {
    pub stmts: Vec<Stmt>,
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Let { name: String, value: Expr },
    Return(Expr),
    Expr(Expr),
}

#[derive(Clone, Debug)]
pub enum Expr {
    Ident(String),
    IntLit(i64),
    FloatLit(f64),
    BinOp {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Call {
        callee: String,
        args: Vec<Expr>,
    },
}

#[derive(Copy, Clone, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    MatMul, // `@`
}
