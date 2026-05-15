//! Lumen DSL — source text to typed AST.
//!
//! Pipeline: `source: &str` → [`lexer::Lexer`] → [`parser::Parser`] → [`ast::Module`]
//! → [`typeck::TypeChecker`] → typed [`ast::Module`].

#![allow(dead_code)] // Phase 0 scaffolding; remove once consumers exist.

pub mod ast;
pub mod diagnostic;
pub mod lexer;
pub mod parser;
pub mod span;
pub mod typeck;

pub use ast::Module;
pub use diagnostic::Diagnostic;
pub use span::Span;

/// Convenience: source → typed module, collecting diagnostics.
pub fn compile(_source: &str) -> Result<Module, Vec<Diagnostic>> {
    // Phase 1 will wire lexer → parser → typeck here.
    Err(vec![Diagnostic::todo(
        "lumen-dsl::compile not implemented yet",
    )])
}
