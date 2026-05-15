//! AST → IR lowering. Phase 2 work.

use crate::module::IrModule;
use lumen_dsl::Module;

pub fn lower(_ast: &Module) -> IrModule {
    IrModule::default()
}
