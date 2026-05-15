//! Byte spans into source text. Zero allocation.

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub const fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    pub const fn len(self) -> u32 {
        self.end - self.start
    }

    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }

    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    pub fn slice(self, source: &str) -> &str {
        &source[self.start as usize..self.end as usize]
    }

    /// 1-indexed `(line, column)` of [`Span::start`], computed by scanning `source`
    /// from the beginning. O(n) — call only on the error path.
    pub fn line_col(self, source: &str) -> (u32, u32) {
        let start = self.start as usize;
        debug_assert!(start <= source.len());
        let prefix = &source[..start.min(source.len())];
        let mut line = 1u32;
        let mut col = 1u32;
        for c in prefix.chars() {
            if c == '\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
        }
        (line, col)
    }
}
