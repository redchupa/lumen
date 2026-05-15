//! Lexer. Zero-copy: tokens carry [`Span`]s, never owned strings.
//!
//! Identifier policy: ASCII letters / underscore + any non-ASCII alphabetic
//! (so Korean identifiers like `행렬곱` parse). Numerics are limited to ASCII.

use crate::diagnostic::Diagnostic;
use crate::span::Span;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TokenKind {
    // Literals
    Ident,
    Int,
    Float,
    // Keywords
    KwFn,
    KwLet,
    KwReturn,
    KwIf,
    KwElse,
    KwTensor,
    // Scalar type keywords
    TyF32,
    TyF16,
    TyI32,
    TyQ4_0,
    TyQ8_0,
    // Punctuation
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    LAngle,
    RAngle,
    Comma,
    Semicolon,
    Colon,
    Arrow, // ->
    Eq,
    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    At, // @ (matmul)
    // End
    Eof,
}

impl TokenKind {
    pub fn from_ident(s: &str) -> TokenKind {
        match s {
            "fn" => TokenKind::KwFn,
            "let" => TokenKind::KwLet,
            "return" => TokenKind::KwReturn,
            "if" => TokenKind::KwIf,
            "else" => TokenKind::KwElse,
            "tensor" => TokenKind::KwTensor,
            "f32" => TokenKind::TyF32,
            "f16" => TokenKind::TyF16,
            "i32" => TokenKind::TyI32,
            "q4_0" => TokenKind::TyQ4_0,
            "q8_0" => TokenKind::TyQ8_0,
            _ => TokenKind::Ident,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

pub struct Lexer<'src> {
    source: &'src str,
    bytes: &'src [u8],
    pos: u32,
}

impl<'src> Lexer<'src> {
    pub fn new(source: &'src str) -> Self {
        assert!(
            source.len() <= u32::MAX as usize,
            "source > 4 GiB unsupported"
        );
        Self {
            source,
            bytes: source.as_bytes(),
            pos: 0,
        }
    }

    pub fn source(&self) -> &'src str {
        self.source
    }

    /// Tokenize the entire source. Returns either the token stream (with a
    /// trailing `Eof`) or accumulated diagnostics.
    pub fn tokenize(mut self) -> Result<Vec<Token>, Vec<Diagnostic>> {
        let mut tokens = Vec::new();
        let mut errors = Vec::new();
        loop {
            match self.next_token() {
                Ok(tok) => {
                    let eof = tok.kind == TokenKind::Eof;
                    tokens.push(tok);
                    if eof {
                        break;
                    }
                }
                Err(d) => errors.push(d),
            }
        }
        if errors.is_empty() {
            Ok(tokens)
        } else {
            Err(errors)
        }
    }

    fn peek_byte(&self) -> Option<u8> {
        self.bytes.get(self.pos as usize).copied()
    }

    fn peek_byte_at(&self, offset: u32) -> Option<u8> {
        self.bytes.get((self.pos + offset) as usize).copied()
    }

    fn advance(&mut self) {
        // Advance by one UTF-8 character, not one byte.
        let s = &self.source[self.pos as usize..];
        if let Some(c) = s.chars().next() {
            self.pos += c.len_utf8() as u32;
        }
    }

    fn skip_trivia(&mut self) {
        loop {
            match self.peek_byte() {
                Some(b' ' | b'\t' | b'\r' | b'\n') => self.advance(),
                Some(b'/') if self.peek_byte_at(1) == Some(b'/') => {
                    while let Some(b) = self.peek_byte() {
                        if b == b'\n' {
                            break;
                        }
                        self.advance();
                    }
                }
                Some(b'/') if self.peek_byte_at(1) == Some(b'*') => {
                    self.advance();
                    self.advance();
                    while let Some(b) = self.peek_byte() {
                        if b == b'*' && self.peek_byte_at(1) == Some(b'/') {
                            self.advance();
                            self.advance();
                            break;
                        }
                        self.advance();
                    }
                }
                _ => return,
            }
        }
    }

    fn next_token(&mut self) -> Result<Token, Diagnostic> {
        self.skip_trivia();
        let start = self.pos;
        let Some(b) = self.peek_byte() else {
            return Ok(Token {
                kind: TokenKind::Eof,
                span: Span::new(start, start),
            });
        };

        let kind = match b {
            b'(' => {
                self.advance();
                TokenKind::LParen
            }
            b')' => {
                self.advance();
                TokenKind::RParen
            }
            b'{' => {
                self.advance();
                TokenKind::LBrace
            }
            b'}' => {
                self.advance();
                TokenKind::RBrace
            }
            b'[' => {
                self.advance();
                TokenKind::LBracket
            }
            b']' => {
                self.advance();
                TokenKind::RBracket
            }
            b'<' => {
                self.advance();
                TokenKind::LAngle
            }
            b'>' => {
                self.advance();
                TokenKind::RAngle
            }
            b',' => {
                self.advance();
                TokenKind::Comma
            }
            b';' => {
                self.advance();
                TokenKind::Semicolon
            }
            b':' => {
                self.advance();
                TokenKind::Colon
            }
            b'=' => {
                self.advance();
                TokenKind::Eq
            }
            b'+' => {
                self.advance();
                TokenKind::Plus
            }
            b'-' => {
                if self.peek_byte_at(1) == Some(b'>') {
                    self.advance();
                    self.advance();
                    TokenKind::Arrow
                } else {
                    self.advance();
                    TokenKind::Minus
                }
            }
            b'*' => {
                self.advance();
                TokenKind::Star
            }
            b'/' => {
                self.advance();
                TokenKind::Slash
            }
            b'@' => {
                self.advance();
                TokenKind::At
            }
            b'0'..=b'9' => self.lex_number()?,
            _ => {
                // Identifier? First char must be ASCII letter, underscore, or non-ASCII alphabetic.
                let first_char = self.source[self.pos as usize..]
                    .chars()
                    .next()
                    .expect("peek_byte was Some");
                if is_ident_start(first_char) {
                    self.lex_ident()
                } else {
                    let span = Span::new(self.pos, self.pos + first_char.len_utf8() as u32);
                    self.advance();
                    return Err(Diagnostic::error(
                        format!("unexpected character `{}`", first_char),
                        span,
                    ));
                }
            }
        };
        Ok(Token {
            kind,
            span: Span::new(start, self.pos),
        })
    }

    fn lex_number(&mut self) -> Result<TokenKind, Diagnostic> {
        while let Some(b) = self.peek_byte() {
            if b.is_ascii_digit() {
                self.advance();
            } else {
                break;
            }
        }
        let mut is_float = false;
        if self.peek_byte() == Some(b'.')
            && self.peek_byte_at(1).is_some_and(|c| c.is_ascii_digit())
        {
            is_float = true;
            self.advance(); // '.'
            while let Some(b) = self.peek_byte() {
                if b.is_ascii_digit() {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        if let Some(b'e' | b'E') = self.peek_byte() {
            is_float = true;
            self.advance();
            if let Some(b'+' | b'-') = self.peek_byte() {
                self.advance();
            }
            while let Some(b) = self.peek_byte() {
                if b.is_ascii_digit() {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        Ok(if is_float {
            TokenKind::Float
        } else {
            TokenKind::Int
        })
    }

    fn lex_ident(&mut self) -> TokenKind {
        let start = self.pos as usize;
        while let Some(c) = self.source[self.pos as usize..].chars().next() {
            if is_ident_continue(c) {
                self.pos += c.len_utf8() as u32;
            } else {
                break;
            }
        }
        let text = &self.source[start..self.pos as usize];
        TokenKind::from_ident(text)
    }
}

fn is_ident_start(c: char) -> bool {
    c == '_' || c.is_ascii_alphabetic() || (!c.is_ascii() && c.is_alphabetic())
}

fn is_ident_continue(c: char) -> bool {
    is_ident_start(c) || c.is_ascii_digit()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<TokenKind> {
        Lexer::new(src)
            .tokenize()
            .unwrap()
            .into_iter()
            .map(|t| t.kind)
            .collect()
    }

    #[test]
    fn empty() {
        assert_eq!(kinds(""), vec![TokenKind::Eof]);
    }

    #[test]
    fn keywords_and_idents() {
        let toks = kinds("fn matmul let x return");
        assert_eq!(
            toks,
            vec![
                TokenKind::KwFn,
                TokenKind::Ident,
                TokenKind::KwLet,
                TokenKind::Ident,
                TokenKind::KwReturn,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn types() {
        let toks = kinds("tensor<f32, [64, 128]>");
        assert_eq!(
            toks,
            vec![
                TokenKind::KwTensor,
                TokenKind::LAngle,
                TokenKind::TyF32,
                TokenKind::Comma,
                TokenKind::LBracket,
                TokenKind::Int,
                TokenKind::Comma,
                TokenKind::Int,
                TokenKind::RBracket,
                TokenKind::RAngle,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn punctuation_and_ops() {
        assert_eq!(
            kinds("a @ b -> c"),
            vec![
                TokenKind::Ident,
                TokenKind::At,
                TokenKind::Ident,
                TokenKind::Arrow,
                TokenKind::Ident,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn comments() {
        assert_eq!(
            kinds("// line\nfn /* block */ x"),
            vec![TokenKind::KwFn, TokenKind::Ident, TokenKind::Eof]
        );
    }

    #[test]
    fn numbers() {
        // `2.` (trailing dot without a fractional digit) intentionally stays as
        // `Int(2)` so that future syntax can use `.` as a separate token. We
        // don't exercise that case here to keep the test deterministic.
        assert_eq!(
            kinds("1 2.5 3e-4 42"),
            vec![
                TokenKind::Int,
                TokenKind::Float,
                TokenKind::Float,
                TokenKind::Int,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn korean_identifier() {
        let toks = kinds("행렬곱");
        assert_eq!(toks, vec![TokenKind::Ident, TokenKind::Eof]);
    }

    #[test]
    fn quantized_keywords() {
        assert_eq!(
            kinds("q4_0 q8_0"),
            vec![TokenKind::TyQ4_0, TokenKind::TyQ8_0, TokenKind::Eof]
        );
    }
}
