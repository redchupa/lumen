//! Lexer. Phase 1 work — currently a stub.

use crate::span::Span;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenKind {
    // Literals
    Ident,
    Int,
    Float,
    String,
    // Keywords
    Fn,
    Let,
    Return,
    If,
    Else,
    Tensor,
    // Types
    F32,
    F16,
    I32,
    Q4_0,
    Q8_0,
    // Punctuation
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Semicolon,
    Colon,
    Arrow,
    Eq,
    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    At, // matmul
    // End
    Eof,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

pub struct Lexer<'src> {
    _source: &'src str,
}

impl<'src> Lexer<'src> {
    pub fn new(source: &'src str) -> Self {
        Self { _source: source }
    }

    pub fn next_token(&mut self) -> Option<Token> {
        // TODO Phase 1
        None
    }
}
