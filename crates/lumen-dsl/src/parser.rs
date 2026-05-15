//! Recursive-descent parser with Pratt-style expression precedence.
//!
//! Grammar (informal):
//!
//! ```text
//! module     = item*
//! item       = function
//! function   = "fn" IDENT "(" params? ")" "->" type block
//! params     = param ("," param)* ","?
//! param      = IDENT ":" type
//! type       = scalar
//!            | "tensor" "<" scalar "," shape ">"
//! scalar     = "f32" | "f16" | "i32" | "q4_0" | "q8_0"
//! shape      = "[" INT ("," INT)* "]"
//! block      = "{" stmt* "}"
//! stmt       = "let" IDENT "=" expr ";"
//!            | "return" expr ";"
//!            | expr ";"
//! expr       = pratt(0)
//! ```
//!
//! Expression precedence (high → low):
//!   - prefix `-`
//!   - `@` (matmul)   — left-assoc
//!   - `*`, `/`       — left-assoc
//!   - `+`, `-`       — left-assoc

use crate::ast::*;
use crate::diagnostic::Diagnostic;
use crate::lexer::{Lexer, Token, TokenKind};

pub struct Parser<'src> {
    source: &'src str,
    tokens: Vec<Token>,
    cursor: usize,
}

impl<'src> Parser<'src> {
    pub fn new(source: &'src str) -> Result<Self, Vec<Diagnostic>> {
        let tokens = Lexer::new(source).tokenize()?;
        Ok(Self {
            source,
            tokens,
            cursor: 0,
        })
    }

    pub fn parse(source: &'src str) -> Result<Module, Vec<Diagnostic>> {
        let mut p = Parser::new(source)?;
        p.parse_module()
    }

    // ----- token helpers --------------------------------------------------

    fn peek(&self) -> Token {
        self.tokens[self.cursor]
    }

    fn bump(&mut self) -> Token {
        let t = self.peek();
        if t.kind != TokenKind::Eof {
            self.cursor += 1;
        }
        t
    }

    fn at(&self, k: TokenKind) -> bool {
        self.peek().kind == k
    }

    fn eat(&mut self, k: TokenKind) -> Option<Token> {
        if self.at(k) {
            Some(self.bump())
        } else {
            None
        }
    }

    fn expect(&mut self, k: TokenKind, ctx: &str) -> Result<Token, Diagnostic> {
        let t = self.peek();
        if t.kind == k {
            Ok(self.bump())
        } else {
            Err(Diagnostic::error(
                format!(
                    "expected {} in {}, found `{}`",
                    token_label(k),
                    ctx,
                    t.span.slice(self.source)
                ),
                t.span,
            ))
        }
    }

    // ----- top level ------------------------------------------------------

    fn parse_module(&mut self) -> Result<Module, Vec<Diagnostic>> {
        let mut items = Vec::new();
        let mut errors = Vec::new();
        while !self.at(TokenKind::Eof) {
            match self.parse_item() {
                Ok(item) => items.push(item),
                Err(d) => {
                    errors.push(d);
                    self.recover_to_top_level();
                }
            }
        }
        if errors.is_empty() {
            Ok(Module { items })
        } else {
            Err(errors)
        }
    }

    fn recover_to_top_level(&mut self) {
        while !matches!(self.peek().kind, TokenKind::Eof | TokenKind::KwFn) {
            self.bump();
        }
    }

    fn parse_item(&mut self) -> Result<Item, Diagnostic> {
        let t = self.peek();
        match t.kind {
            TokenKind::KwFn => self.parse_function().map(Item::Function),
            _ => Err(Diagnostic::error(
                format!("expected `fn`, found `{}`", t.span.slice(self.source)),
                t.span,
            )),
        }
    }

    // ----- function -------------------------------------------------------

    fn parse_function(&mut self) -> Result<Function, Diagnostic> {
        let kw = self.expect(TokenKind::KwFn, "function declaration")?;
        let name = self.parse_ident("function name")?;
        self.expect(TokenKind::LParen, "function signature")?;
        let mut params = Vec::new();
        if !self.at(TokenKind::RParen) {
            loop {
                params.push(self.parse_param()?);
                if self.eat(TokenKind::Comma).is_none() {
                    break;
                }
                if self.at(TokenKind::RParen) {
                    break; // trailing comma
                }
            }
        }
        self.expect(TokenKind::RParen, "function signature")?;
        self.expect(TokenKind::Arrow, "function return type")?;
        let ret = self.parse_type()?;
        let body = self.parse_block()?;
        let span = kw.span.merge(body.span);
        Ok(Function {
            name,
            params,
            ret,
            body,
            span,
        })
    }

    fn parse_param(&mut self) -> Result<Param, Diagnostic> {
        let name = self.parse_ident("parameter name")?;
        self.expect(TokenKind::Colon, "parameter")?;
        let ty = self.parse_type()?;
        let span = name.span.merge(ty.span);
        Ok(Param { name, ty, span })
    }

    fn parse_ident(&mut self, ctx: &str) -> Result<Ident, Diagnostic> {
        let t = self.peek();
        if t.kind == TokenKind::Ident {
            self.bump();
            Ok(Ident {
                name: t.span.slice(self.source).to_string(),
                span: t.span,
            })
        } else {
            Err(Diagnostic::error(
                format!(
                    "expected identifier in {}, found `{}`",
                    ctx,
                    t.span.slice(self.source)
                ),
                t.span,
            ))
        }
    }

    // ----- types ----------------------------------------------------------

    fn parse_type(&mut self) -> Result<Type, Diagnostic> {
        let t = self.peek();
        match t.kind {
            TokenKind::TyF32
            | TokenKind::TyF16
            | TokenKind::TyI32
            | TokenKind::TyQ4_0
            | TokenKind::TyQ8_0 => {
                self.bump();
                Ok(Type {
                    kind: TypeKind::Scalar(scalar_from_kw(t.kind)),
                    span: t.span,
                })
            }
            TokenKind::KwTensor => {
                let kw = self.bump();
                self.expect(TokenKind::LAngle, "tensor type")?;
                let elem_tok = self.peek();
                let elem = match elem_tok.kind {
                    TokenKind::TyF32
                    | TokenKind::TyF16
                    | TokenKind::TyI32
                    | TokenKind::TyQ4_0
                    | TokenKind::TyQ8_0 => {
                        self.bump();
                        scalar_from_kw(elem_tok.kind)
                    }
                    _ => {
                        return Err(Diagnostic::error(
                            format!(
                                "expected scalar type in tensor type, found `{}`",
                                elem_tok.span.slice(self.source)
                            ),
                            elem_tok.span,
                        ));
                    }
                };
                self.expect(TokenKind::Comma, "tensor type")?;
                let shape = self.parse_shape()?;
                let close = self.expect(TokenKind::RAngle, "tensor type")?;
                let span = kw.span.merge(close.span);
                Ok(Type {
                    kind: TypeKind::Tensor { elem, shape },
                    span,
                })
            }
            _ => Err(Diagnostic::error(
                format!("expected type, found `{}`", t.span.slice(self.source)),
                t.span,
            )),
        }
    }

    fn parse_shape(&mut self) -> Result<Shape, Diagnostic> {
        let open = self.expect(TokenKind::LBracket, "tensor shape")?;
        let mut dims = Vec::new();
        if !self.at(TokenKind::RBracket) {
            loop {
                dims.push(self.parse_dim()?);
                if self.eat(TokenKind::Comma).is_none() {
                    break;
                }
                if self.at(TokenKind::RBracket) {
                    break;
                }
            }
        }
        let close = self.expect(TokenKind::RBracket, "tensor shape")?;
        Ok(Shape {
            dims,
            span: open.span.merge(close.span),
        })
    }

    fn parse_dim(&mut self) -> Result<Dim, Diagnostic> {
        let t = self.peek();
        if t.kind == TokenKind::Int {
            self.bump();
            let text = t.span.slice(self.source);
            let value: u64 = text.parse().map_err(|_| {
                Diagnostic::error(format!("invalid shape dimension `{}`", text), t.span)
            })?;
            Ok(Dim {
                kind: DimKind::Static(value),
                span: t.span,
            })
        } else {
            Err(Diagnostic::error(
                format!(
                    "expected integer in tensor shape, found `{}`",
                    t.span.slice(self.source)
                ),
                t.span,
            ))
        }
    }

    // ----- statements / blocks --------------------------------------------

    fn parse_block(&mut self) -> Result<Block, Diagnostic> {
        let open = self.expect(TokenKind::LBrace, "block")?;
        let mut stmts = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            stmts.push(self.parse_stmt()?);
        }
        let close = self.expect(TokenKind::RBrace, "block")?;
        Ok(Block {
            stmts,
            span: open.span.merge(close.span),
        })
    }

    fn parse_stmt(&mut self) -> Result<Stmt, Diagnostic> {
        let t = self.peek();
        match t.kind {
            TokenKind::KwLet => self.parse_let_stmt(),
            TokenKind::KwReturn => self.parse_return_stmt(),
            _ => {
                let start = self.peek().span;
                let e = self.parse_expr()?;
                let semi = self.expect(TokenKind::Semicolon, "expression statement")?;
                let span = start.merge(semi.span);
                Ok(Stmt {
                    kind: StmtKind::Expr(e),
                    span,
                })
            }
        }
    }

    fn parse_let_stmt(&mut self) -> Result<Stmt, Diagnostic> {
        let kw = self.expect(TokenKind::KwLet, "let statement")?;
        let name = self.parse_ident("let binding")?;
        self.expect(TokenKind::Eq, "let binding")?;
        let value = self.parse_expr()?;
        let semi = self.expect(TokenKind::Semicolon, "let statement")?;
        Ok(Stmt {
            kind: StmtKind::Let { name, value },
            span: kw.span.merge(semi.span),
        })
    }

    fn parse_return_stmt(&mut self) -> Result<Stmt, Diagnostic> {
        let kw = self.expect(TokenKind::KwReturn, "return statement")?;
        let value = self.parse_expr()?;
        let semi = self.expect(TokenKind::Semicolon, "return statement")?;
        Ok(Stmt {
            kind: StmtKind::Return(value),
            span: kw.span.merge(semi.span),
        })
    }

    // ----- expressions (Pratt) --------------------------------------------

    fn parse_expr(&mut self) -> Result<Expr, Diagnostic> {
        self.parse_expr_bp(0)
    }

    /// Pratt parser. `min_bp` is the minimum binding power for left operators
    /// to keep consuming.
    fn parse_expr_bp(&mut self, min_bp: u8) -> Result<Expr, Diagnostic> {
        let mut lhs = self.parse_prefix()?;

        loop {
            let op_tok = self.peek();
            let Some((op, l_bp, r_bp)) = infix_binding_power(op_tok.kind) else {
                break;
            };
            if l_bp < min_bp {
                break;
            }
            self.bump();
            let rhs = self.parse_expr_bp(r_bp)?;
            let span = lhs.span.merge(rhs.span);
            lhs = Expr {
                kind: ExprKind::BinOp {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_prefix(&mut self) -> Result<Expr, Diagnostic> {
        let t = self.peek();
        match t.kind {
            TokenKind::Minus => {
                self.bump();
                let inner = self.parse_expr_bp(prefix_binding_power())?;
                let span = t.span.merge(inner.span);
                Ok(Expr {
                    kind: ExprKind::Unary {
                        op: UnaryOp::Neg,
                        expr: Box::new(inner),
                    },
                    span,
                })
            }
            TokenKind::LParen => {
                self.bump();
                let e = self.parse_expr_bp(0)?;
                self.expect(TokenKind::RParen, "parenthesized expression")?;
                Ok(e)
            }
            TokenKind::Int => self.parse_int_lit(),
            TokenKind::Float => self.parse_float_lit(),
            TokenKind::Ident => self.parse_ident_or_call(),
            _ => Err(Diagnostic::error(
                format!("expected expression, found `{}`", t.span.slice(self.source)),
                t.span,
            )),
        }
    }

    fn parse_int_lit(&mut self) -> Result<Expr, Diagnostic> {
        let t = self.bump();
        let text = t.span.slice(self.source);
        let value: i64 = text
            .parse()
            .map_err(|_| Diagnostic::error(format!("invalid integer `{}`", text), t.span))?;
        Ok(Expr {
            kind: ExprKind::IntLit(value),
            span: t.span,
        })
    }

    fn parse_float_lit(&mut self) -> Result<Expr, Diagnostic> {
        let t = self.bump();
        let text = t.span.slice(self.source);
        let value: f64 = text
            .parse()
            .map_err(|_| Diagnostic::error(format!("invalid float `{}`", text), t.span))?;
        Ok(Expr {
            kind: ExprKind::FloatLit(value),
            span: t.span,
        })
    }

    fn parse_ident_or_call(&mut self) -> Result<Expr, Diagnostic> {
        let name = self.parse_ident("expression")?;
        if self.at(TokenKind::LParen) {
            self.bump();
            let mut args = Vec::new();
            if !self.at(TokenKind::RParen) {
                loop {
                    args.push(self.parse_expr()?);
                    if self.eat(TokenKind::Comma).is_none() {
                        break;
                    }
                    if self.at(TokenKind::RParen) {
                        break;
                    }
                }
            }
            let close = self.expect(TokenKind::RParen, "function call")?;
            let span = name.span.merge(close.span);
            Ok(Expr {
                kind: ExprKind::Call { callee: name, args },
                span,
            })
        } else {
            let span = name.span;
            Ok(Expr {
                kind: ExprKind::Ident(name.name),
                span,
            })
        }
    }
}

// ---- precedence tables ---------------------------------------------------

/// Returns `(BinOp, left_bp, right_bp)` for a left-associative operator if the
/// token is one. `right_bp > left_bp` means left-assoc.
fn infix_binding_power(k: TokenKind) -> Option<(BinOp, u8, u8)> {
    Some(match k {
        TokenKind::Plus => (BinOp::Add, 10, 11),
        TokenKind::Minus => (BinOp::Sub, 10, 11),
        TokenKind::Star => (BinOp::Mul, 20, 21),
        TokenKind::Slash => (BinOp::Div, 20, 21),
        TokenKind::At => (BinOp::MatMul, 30, 31),
        _ => return None,
    })
}

fn prefix_binding_power() -> u8 {
    // Unary minus binds tighter than `@`.
    40
}

fn scalar_from_kw(k: TokenKind) -> ScalarType {
    match k {
        TokenKind::TyF32 => ScalarType::F32,
        TokenKind::TyF16 => ScalarType::F16,
        TokenKind::TyI32 => ScalarType::I32,
        TokenKind::TyQ4_0 => ScalarType::Q4_0,
        TokenKind::TyQ8_0 => ScalarType::Q8_0,
        _ => unreachable!("scalar_from_kw with non-scalar kind"),
    }
}

fn token_label(k: TokenKind) -> &'static str {
    match k {
        TokenKind::Ident => "identifier",
        TokenKind::Int => "integer",
        TokenKind::Float => "float",
        TokenKind::KwFn => "`fn`",
        TokenKind::KwLet => "`let`",
        TokenKind::KwReturn => "`return`",
        TokenKind::KwIf => "`if`",
        TokenKind::KwElse => "`else`",
        TokenKind::KwTensor => "`tensor`",
        TokenKind::TyF32 => "`f32`",
        TokenKind::TyF16 => "`f16`",
        TokenKind::TyI32 => "`i32`",
        TokenKind::TyQ4_0 => "`q4_0`",
        TokenKind::TyQ8_0 => "`q8_0`",
        TokenKind::LParen => "`(`",
        TokenKind::RParen => "`)`",
        TokenKind::LBrace => "`{`",
        TokenKind::RBrace => "`}`",
        TokenKind::LBracket => "`[`",
        TokenKind::RBracket => "`]`",
        TokenKind::LAngle => "`<`",
        TokenKind::RAngle => "`>`",
        TokenKind::Comma => "`,`",
        TokenKind::Semicolon => "`;`",
        TokenKind::Colon => "`:`",
        TokenKind::Arrow => "`->`",
        TokenKind::Eq => "`=`",
        TokenKind::Plus => "`+`",
        TokenKind::Minus => "`-`",
        TokenKind::Star => "`*`",
        TokenKind::Slash => "`/`",
        TokenKind::At => "`@`",
        TokenKind::Eof => "end of input",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> Module {
        Parser::parse(src).unwrap_or_else(|errs| panic!("parse errors: {:?}", errs))
    }

    #[test]
    fn empty_module() {
        let m = parse("");
        assert!(m.items.is_empty());
    }

    #[test]
    fn matmul_function() {
        let src = r#"
            fn matmul(
                a: tensor<f32, [64, 128]>,
                b: tensor<f32, [128, 32]>,
            ) -> tensor<f32, [64, 32]> {
                return a @ b;
            }
        "#;
        let m = parse(src);
        assert_eq!(m.items.len(), 1);
        let Item::Function(f) = &m.items[0];
        assert_eq!(f.name.name, "matmul");
        assert_eq!(f.params.len(), 2);
        assert_eq!(f.body.stmts.len(), 1);
    }

    #[test]
    fn precedence_at_vs_plus() {
        // `a + b @ c` should parse as `a + (b @ c)`.
        let src = r#"
            fn t(a: f32, b: f32, c: f32) -> f32 {
                return a + b @ c;
            }
        "#;
        let m = parse(src);
        let Item::Function(f) = &m.items[0];
        let StmtKind::Return(e) = &f.body.stmts[0].kind else {
            panic!("expected return");
        };
        let ExprKind::BinOp { op, rhs, .. } = &e.kind else {
            panic!("expected binop, got {:?}", e.kind);
        };
        assert_eq!(*op, BinOp::Add);
        let ExprKind::BinOp { op: inner, .. } = &rhs.kind else {
            panic!("rhs not binop");
        };
        assert_eq!(*inner, BinOp::MatMul);
    }

    #[test]
    fn let_and_return() {
        let src = r#"
            fn f(a: f32) -> f32 {
                let x = a + 1;
                return x * 2;
            }
        "#;
        let m = parse(src);
        let Item::Function(f) = &m.items[0];
        assert!(matches!(f.body.stmts[0].kind, StmtKind::Let { .. }));
        assert!(matches!(f.body.stmts[1].kind, StmtKind::Return(_)));
    }

    #[test]
    fn unary_neg() {
        let src = r#"fn f(a: f32) -> f32 { return -a; }"#;
        let m = parse(src);
        let Item::Function(f) = &m.items[0];
        let StmtKind::Return(e) = &f.body.stmts[0].kind else {
            unreachable!()
        };
        assert!(matches!(
            e.kind,
            ExprKind::Unary {
                op: UnaryOp::Neg,
                ..
            }
        ));
    }

    #[test]
    fn parens_override_precedence() {
        let src = r#"fn f(a: f32, b: f32, c: f32) -> f32 { return (a + b) * c; }"#;
        let m = parse(src);
        let Item::Function(f) = &m.items[0];
        let StmtKind::Return(e) = &f.body.stmts[0].kind else {
            unreachable!()
        };
        let ExprKind::BinOp { op, .. } = &e.kind else {
            panic!("not binop")
        };
        assert_eq!(*op, BinOp::Mul);
    }

    #[test]
    fn errors_on_missing_arrow() {
        let src = "fn f() tensor<f32, [1]> { return 1; }";
        let errs = Parser::parse(src).unwrap_err();
        assert!(!errs.is_empty());
    }
}
