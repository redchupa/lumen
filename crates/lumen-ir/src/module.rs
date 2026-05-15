//! IR container: Module → Function → Block → Value(Op).

use crate::op::Op;
use crate::ty::TensorType;

/// Stable index into [`Function::values`]. Survives reordering of basic blocks.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

#[derive(Clone, Debug)]
pub struct Value {
    pub op: Op,
    pub ty: TensorType,
}

#[derive(Clone, Debug, Default)]
pub struct Block {
    pub values: Vec<ValueId>,
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub params: Vec<TensorType>,
    pub ret: TensorType,
    pub values: Vec<Value>,
    pub blocks: Vec<Block>,
}

impl Function {
    pub fn new(name: impl Into<String>, params: Vec<TensorType>, ret: TensorType) -> Self {
        Self {
            name: name.into(),
            params,
            ret,
            values: Vec::new(),
            blocks: vec![Block::default()],
        }
    }

    pub fn push(&mut self, value: Value) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.push(value);
        self.blocks.last_mut().expect("at least one block").values.push(id);
        id
    }
}

#[derive(Clone, Debug, Default)]
pub struct IrModule {
    pub functions: Vec<Function>,
}
