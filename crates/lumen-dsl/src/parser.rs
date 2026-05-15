//! Pratt parser. Phase 1 work — currently a stub.

use crate::ast::Module;
use crate::diagnostic::Diagnostic;

pub struct Parser;

impl Parser {
    pub fn parse(_source: &str) -> Result<Module, Vec<Diagnostic>> {
        Err(vec![Diagnostic::todo("Parser::parse not implemented yet")])
    }
}
