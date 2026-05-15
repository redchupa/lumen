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
    /// Result IDs of the synthetic `Op::Param { index: i }` values prepended
    /// to `values`. Same order as `params`.
    pub param_values: Vec<ValueId>,
}

impl Function {
    /// Create a new function with a single empty entry block. Each parameter is
    /// represented by a synthetic `Op::Param { index }` value pushed into the
    /// entry block, so callers can reference parameters by [`ValueId`].
    pub fn new(name: impl Into<String>, params: Vec<TensorType>, ret: TensorType) -> Self {
        let mut f = Self {
            name: name.into(),
            params: params.clone(),
            ret,
            values: Vec::new(),
            blocks: vec![Block::default()],
            param_values: Vec::with_capacity(params.len()),
        };
        for (i, pty) in params.into_iter().enumerate() {
            let id = f.push(Value {
                op: Op::Param { index: i as u32 },
                ty: pty,
            });
            f.param_values.push(id);
        }
        f
    }

    pub fn push(&mut self, value: Value) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.push(value);
        self.blocks
            .last_mut()
            .expect("at least one block")
            .values
            .push(id);
        id
    }

    pub fn value(&self, id: ValueId) -> &Value {
        &self.values[id.0 as usize]
    }

    pub fn type_of(&self, id: ValueId) -> &TensorType {
        &self.values[id.0 as usize].ty
    }
}

#[derive(Clone, Debug, Default)]
pub struct IrModule {
    pub functions: Vec<Function>,
}
