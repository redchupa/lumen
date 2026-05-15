//! Compiler diagnostics (errors, warnings, notes).

use crate::span::Span;

#[derive(Clone, Debug)]
pub enum Severity {
    Error,
    Warning,
    Note,
}

#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub severity: Severity,
    pub message: String,
    pub span: Option<Span>,
}

impl Diagnostic {
    pub fn error(message: impl Into<String>, span: Span) -> Self {
        Self {
            severity: Severity::Error,
            message: message.into(),
            span: Some(span),
        }
    }

    pub fn todo(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Error,
            message: format!("TODO: {}", message.into()),
            span: None,
        }
    }
}
