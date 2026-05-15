//! Type checker. Phase 1 — currently a stub.

use crate::ast::Module;
use crate::diagnostic::Diagnostic;

pub struct TypeChecker;

impl TypeChecker {
    pub fn check(_module: &mut Module) -> Result<(), Vec<Diagnostic>> {
        Err(vec![Diagnostic::todo(
            "TypeChecker::check not implemented yet",
        )])
    }
}
