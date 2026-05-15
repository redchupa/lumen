//! Kernel dispatch: pick a JIT-cached kernel for the given op + input shapes,
//! or compile a new one.

pub struct Dispatcher;

impl Dispatcher {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Dispatcher {
    fn default() -> Self {
        Self::new()
    }
}
