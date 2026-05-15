//! Pass manager scaffolding. Real passes land in Phase 2.

use crate::module::IrModule;

pub trait Pass {
    fn name(&self) -> &'static str;
    fn run(&self, module: &mut IrModule);
}

#[derive(Default)]
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
}

impl PassManager {
    pub fn add(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }

    pub fn run(&self, module: &mut IrModule) {
        for p in &self.passes {
            p.run(module);
        }
    }
}
