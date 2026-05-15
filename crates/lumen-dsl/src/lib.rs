//! Lumen DSL — source text to typed AST.
//!
//! Pipeline: `source: &str` → [`lexer::Lexer`] → [`parser::Parser`] → [`ast::Module`]
//! → [`typeck::TypeChecker`] → typed [`ast::Module`].

pub mod ast;
pub mod diagnostic;
pub mod lexer;
pub mod parser;
pub mod span;
pub mod typeck;

pub use ast::Module;
pub use diagnostic::{render_all, Diagnostic};
pub use parser::Parser;
pub use span::Span;
pub use typeck::TypeChecker;

/// Convenience: lex → parse → type-check.
pub fn compile(source: &str) -> Result<Module, Vec<Diagnostic>> {
    let module = Parser::parse(source)?;
    TypeChecker::check(&module)?;
    Ok(module)
}
