//! Compiler diagnostics (errors, warnings, notes) and pretty-printer.

use std::fmt;

use crate::span::Span;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Note,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
        })
    }
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

    /// Codespan-style multi-line rendering for terminals. `file_name` shows in the
    /// banner. `source` must be the same text the span was extracted from.
    pub fn render(&self, source: &str, file_name: &str) -> String {
        let mut out = format!("{}: {}\n", self.severity, self.message);
        let Some(span) = self.span else {
            return out;
        };
        let (line, col) = span.line_col(source);
        let line_text = nth_line(source, line);
        let line_num_str = line.to_string();
        let gutter = " ".repeat(line_num_str.len());
        out.push_str(&format!("  --> {}:{}:{}\n", file_name, line, col));
        out.push_str(&format!("{} |\n", gutter));
        out.push_str(&format!("{} | {}\n", line_num_str, line_text));
        let underline_len = (span.len() as usize).max(1);
        let pad = " ".repeat((col as usize).saturating_sub(1));
        let underline = "^".repeat(underline_len);
        out.push_str(&format!("{} | {}{}\n", gutter, pad, underline));
        out
    }
}

fn nth_line(source: &str, line: u32) -> &str {
    source.lines().nth((line - 1) as usize).unwrap_or("")
}

/// Renders a batch of diagnostics, separated by blank lines.
pub fn render_all(diags: &[Diagnostic], source: &str, file_name: &str) -> String {
    let mut out = String::new();
    for d in diags {
        out.push_str(&d.render(source, file_name));
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_basic() {
        let src = "fn m() -> f32 {\n    return zz;\n}\n";
        // span over `zz` (find it manually)
        let start = src.find("zz").unwrap() as u32;
        let span = Span::new(start, start + 2);
        let d = Diagnostic::error("unknown identifier `zz`", span);
        let out = d.render(src, "test.lum");
        assert!(out.contains("error: unknown identifier"));
        assert!(out.contains("test.lum:2:12"));
        assert!(out.contains("^^"));
    }
}
