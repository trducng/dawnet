import logging
import uuid
from collections import OrderedDict
from typing import Callable

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class Handler:
    """The handler run along side each module

    Args:
        name: the name of the module
        stage: the running stage of the handler, can be: forward, forward_pre,
            backward, backward_pre
        inspector: the inspector that this handler belongs to
    """

    def __init__(self, name: str, stage: str, inspector: "Inspector"):
        self._name = name
        self._stage = stage
        self._inspector = inspector

    def __call__(self, *args, **kwargs):
        if self._stage == "forward":
            return self.forward(*args, **kwargs)
        elif self._stage == "forward_pre":
            return self.forward_pre(*args, **kwargs)
        elif self._stage == "backward":
            return self.backward(*args, **kwargs)
        elif self._stage == "backward_pre":
            return self.backward_pre(*args, **kwargs)

        raise ValueError(
            f"Unknown stage {self._stage}. Should be one of "
            '["forward", "forward_pre", "backward", "backward_pre"]'
        )

    def forward_pre(self, module, args, kwargs):
        args, kwargs = args, kwargs
        if self._name in self._inspector._module_to_op:
            for op in reversed(self._inspector._module_to_op[self._name]):
                if self._inspector._ops[op.id][2]:
                    args, kwargs = op.forward_pre(
                        self._inspector, self._name, module, args, kwargs
                    )
        return args, kwargs

    def forward(self, module, args, kwargs, output):
        output = output
        if self._name in self._inspector._module_to_op:
            for op in self._inspector._module_to_op[self._name]:
                if self._inspector._ops[op.id][2]:
                    output = op.forward(
                        self._inspector, self._name, module, args, kwargs, output
                    )
        return output

    def backward(self, module, grad_input, grad_output):
        grad_input = grad_input
        if self._name in self._inspector._module_to_op:
            for op in self._inspector._module_to_op[self._name]:
                if self._inspector._ops[op.id][2]:
                    grad_input = op.backward(
                        self._inspector, self._name, module, grad_input, grad_output
                    )
        return grad_input

    def backward_pre(self, module, grad_output):
        grad_output = grad_output
        if self._name in self._inspector._module_to_op:
            for op in self._inspector._module_to_op[self._name]:
                if self._inspector._ops[op.id][2]:
                    grad_output = op.backward_pre(
                        self._inspector, self._name, module, grad_output
                    )
        return grad_output


class Op:
    """Stateless operation"""

    def __init__(self):
        self.id = str(uuid.uuid4())

    def forward(self, inspector: "Inspector", name: str, module, args, kwargs, output):
        return output

    def forward_pre(self, inspector: "Inspector", name: str, module, args, kwargs):
        return args, kwargs

    def backward(
        self, inspector: "Inspector", name: str, module, grad_input, grad_output
    ):
        return grad_input

    def backward_pre(self, inspector: "Inspector", name: str, module, grad_output):
        return grad_output

    def add(self, inspector: "Inspector"):
        pass

    def remove(self, inspector: "Inspector"):
        pass

    def enable(self, inspector: "Inspector"):
        pass

    def disable(self, inspector: "Inspector"):
        pass


class CacheModuleInputOutput(Op):
    """Cache the input and output of a module

    Args:
        no_input: if True, don't cache the input
        no_output: if True, don't cache the output
        input_getter: a callback to get the desired input, should take [args], {kwargs}
        output_getter: a callback to get the output, should take `output` object
    """

    def __ini__(
        self,
        no_input: bool = False,
        no_output: bool = False,
        input_getter: Callable | None = None,
        output_getter: Callable | None = None,
    ):
        super().__init__()
        self._no_input = no_input
        self._no_output = no_output
        self._input_getter = input_getter
        self._output_getter = output_getter

    def forward(self, inspector: "Inspector", name: str, module, args, kwargs, output):
        if self._no_output:
            return output

        if self._output_getter is None:
            inspector.state.output[name] = output
        else:
            inspector.state.output[name] = self._output_getter(output)

        return output

    def forward_pre(self, inspector: "Inspector", name: str, module, args, kwargs):
        if self._no_input:
            return args, kwargs

        if self._input_getter is None:
            inspector.state.input[name] = args, kwargs
        else:
            inspector.state.input[name] = self._input_getter(args, kwargs)

        return args, kwargs


class HookOp(Op):
    """Convenient object to register hook to Pytorch nn.Module using dawnet framework"""

    def __init__(
        self,
        forward: Callable | None = None,
        forward_pre: Callable | None = None,
        backward: Callable | None = None,
        backward_pre: Callable | None = None,
    ):
        super().__init__()
        self._forward = forward
        self._forward_pre = forward_pre
        self._backward = backward
        self._backward_pre = backward_pre

    def forward(self, inspector: "Inspector", name: str, module, args, kwargs, output):
        if self._forward is not None:
            return self._forward(inspector, name, module, args, kwargs, output)
        return output

    def forward_pre(self, inspector: "Inspector", name: str, module, args, kwargs):
        if self._forward_pre is not None:
            return self._forward_pre(inspector, name, module, args, kwargs)
        return args, kwargs

    def backward(
        self, inspector: "Inspector", name: str, module, grad_input, grad_output
    ):
        if self._backward is not None:
            return self._backward(inspector, name, module, grad_input, grad_output)
        return grad_input

    def backward_pre(self, inspector: "Inspector", name: str, module, grad_output):
        if self._backward_pre is not None:
            return self._backward_pre(inspector, name, module, grad_output)
        return grad_output


class SwapStateDict(Op):
    """Swap the state dict of a module"""

    def __init__(self, state_dict: dict, prefix: str | None = None):
        super().__init__()
        self._prefix = prefix
        if prefix is not None:
            self._state_dict = {
                k[len(prefix) :]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
        else:
            self._state_dict = state_dict

    def add(self, inspector: "Inspector"):
        # TODO: check if the layer already has SwapStateDict op added
        self.enable(inspector)

    def enable(self, inspector: "Inspector"):
        if self.id in inspector._private_op_state:
            return

        module_name = inspector._ops[self.id][1]
        module = inspector._model.get_submodule(module_name)
        if self.id not in inspector._private_op_state:
            inspector._private_op_state[self.id] = module.state_dict()
            module.load_state_dict(self._state_dict)

    def disable(self, inspector: "Inspector"):
        if self.id not in inspector._private_op_state:
            return

        module_name = inspector._ops[self.id][1]
        module = inspector._model.get_submodule(module_name)
        module.load_state_dict(inspector._private_op_state[self.id])
        del inspector._private_op_state[self.id]


class SwapModule(Op):
    """Swap the module of a layer"""

    def __init__(self, module: nn.Module):
        super().__init__()
        self._module = module

    def add(self, inspector: "Inspector"):
        # TODO: check if the module is already swapped
        # TODO: inform about operations of the original child module will be ignored
        name = inspector._ops[self.id][1]
        self._module.register_forward_pre_hook(
            Handler(name, "forward_pre", inspector), with_kwargs=True
        )
        self._module.register_forward_hook(
            Handler(name, "forward", inspector), with_kwargs=True, always_call=True
        )
        self.enable(inspector)

    def enable(self, inspector: "Inspector"):
        if self.id in inspector._private_op_state:
            return

        full_module_name = inspector._ops[self.id][1]
        if "." not in full_module_name:
            parent = inspector._model
            module_name = full_module_name
        else:
            module_parent, module_name = full_module_name.rsplit(".", 1)
            parent = inspector._model.get_submodule(module_parent)

        inspector._private_op_state[self.id] = parent._modules[module_name]
        parent._module[module_name] = self._module

    def disable(self, inspector: "Inspector"):
        if self.id not in inspector._private_op_state:
            return

        full_module_name = inspector._ops[self.id][1]
        if "." not in full_module_name:
            parent = inspector._model
            module_name = full_module_name
        else:
            module_parent, module_name = full_module_name.rsplit(".", 1)
            parent = inspector._model.get_submodule(module_parent)

        parent._module[module_name] = inspector._private_op_state[self.id]
        del inspector._private_op_state[self.id]


class SetOutput(Op):
    def __init__(self, output):
        super().__init__()
        self._output = output

    def forward(self, inspector: "Inspector", name: str, module, args, kwargs, output):
        return self._output


class SetBreakpoint(Op):
    def __init__(self, filename: str, lineno: int, pdb_cls: type | None = None):
        super().__init__()
        self._filename = filename
        self._lineno = lineno
        self._pdb_cls = pdb_cls

    def forward_pre(self, inspector: "Inspector", name: str, module, args, kwargs):
        if "in_break" not in inspector._private_op_state:
            # construct the pdb object
            if self._pdb_cls is None:
                from pdb import Pdb

                pdb = Pdb()
            else:
                pdb = self._pdb_cls()

            def pdb_wrapper(module):
                def wrapped(*args, **kwargs):
                    return pdb.runcall(module, *args, **kwargs)

                return wrapped

            full_module_name = inspector._ops[self.id][1]
            module = inspector._model.get_submodule(full_module_name)
            inspector._private_op_state[self.id] = module.forward
            module.forward = pdb_wrapper(module.forward)

            inspector._private_op_state["in_break"] = pdb
            inspector._private_op_state["break_id"] = self.id
        else:
            pdb = inspector._private_op_state["in_break"]

        pdb.set_break(filename=self._filename, lineno=self._lineno)

        return args, kwargs

    def forward(self, inspector: "Inspector", name: str, module, args, kwargs, output):
        if inspector._private_op_state["break_id"] == self.id:
            # we should be the one to clean up
            module.forward = inspector._private_op_state[self.id]
            del inspector._private_op_state["in_break"]
            del inspector._private_op_state["break_id"]
            del inspector._private_op_state[self.id]

        return output


class RunState:
    def __init__(self):
        self.output = {}
        self.input = {}

    def clear(self):
        pass

    def __getitem__(self, item, default=None):
        return getattr(self, item)


def copy_model(module: nn.Module) -> nn.Module:
    """Deep cop a model but shallow copy the parameters to not blow up the memory"""
    from copy import deepcopy, _deepcopy_dispatch, _deepcopy_atomic

    _deepcopy_dispatch[torch.Tensor] = _deepcopy_atomic
    _deepcopy_dispatch[nn.Parameter] = _deepcopy_atomic
    new_module = deepcopy(module)
    del _deepcopy_dispatch[torch.Tensor]
    del _deepcopy_dispatch[nn.Parameter]
    return new_module


class Inspector(nn.Module):
    """Wrap around a model to help inspect its activity

    Attributes:
        _original_model: the supplied model
        _model: the shallow cloned original model will be used to inspect
        _module_to_op: mapping from module name to list of `op` objects
        _ops: contain ops information: op_id -> (op, module_name)
        _private_op_state: private space for the op to store its progress and to
            communicate between layers All `ops` can have access to this space.
        state: contains pulbic information that the op populates
    """

    def __init__(
        self,
        model: nn.Module,
        state: None | RunState = None,
    ):
        super().__init__()
        self._original_model = model
        self._model = copy_model(model)

        self._module_to_op: dict[str, list[Op]] = OrderedDict()
        self._ops: dict[str, list] = OrderedDict()  # op, module name, enable

        self.state: RunState = state or RunState()
        self._private_op_state = {}

        for name, module in self._model.named_modules():
            module.register_forward_pre_hook(
                Handler(name, "forward_pre", self), with_kwargs=True
            )
            module.register_forward_hook(
                Handler(name, "forward", self), with_kwargs=True, always_call=True
            )

    def add_op(self, name: str, op: Op) -> str:
        """Add op to the inspector

        Returns:
            the op id
        """
        if name == "." or self._model.get_submodule(name):
            if name not in self._module_to_op:
                self._module_to_op[name] = []
            self._module_to_op[name].append(op)
            self._ops[op.id] = [op, name, True]
            op.add(self)
            return op.id
        else:
            raise ValueError(f"Module with name {name} doesn't exist")

    def list_ops(self, ids: str | list[str] | None = None, name: str | None = None):
        ops = {}
        if ids is not None:
            if isinstance(ids, str):
                ids = [ids]
            ops.update({op_id: self._ops[op_id][0] for op_id in ids})

        if name is not None:
            if name in self._module_to_op:
                ops.update({op.id: op for op in self._module_to_op[name]})

        if ids is None and name is None:
            ops.update({op_id: op for op_id, (op, _) in self._ops.items()})

        return ops

    def get_op(self, id: str):
        if id not in self._ops:
            raise ValueError(f"Op with id {id} doesn't exist")
        return self._ops[id][0]

    def remove_op(self, op_id: str):
        if op_id not in self._ops:
            raise ValueError(f"Op with id {op_id} doesn't exist")
        self._ops[op_id][0].remove(self)
        layer_name = self._ops[op_id][1]
        self._module_to_op[layer_name] = [
            op for op in self._module_to_op[layer_name] if op.id != op_id
        ]
        del self._ops[op_id]

    def enable_op(self, op_id: str):
        if op_id not in self._ops:
            raise ValueError(f"Op with id {op_id} doesn't exist")
        self._ops[op_id][2] = True
        self._ops[op_id][0].enable(self)

    def disable_op(self, op_id: str):
        if op_id not in self._ops:
            raise ValueError(f"Op with id {op_id} doesn't exist")
        self._ops[op_id][2] = False
        self._ops[op_id][0].disable(self)

    def copy(self) -> "Inspector":
        """Create a copy of the inspector"""
        return Inspector(self._original_model)

    def __call__(self, *args, **kwargs):
        """Execute the model"""
        if self._pdb or self._breaks:
            pdb = Pdb()
            for bp in self._breaks:
                pdb.set_break(filename=bp[0], lineno=bp[1])
            return pdb.runcall(self._model, *args, **kwargs)

        return self._model(*args, **kwargs)

    def __getstate__(self):
        """Save the runner session"""
        state = {
            "model": self._model,
            "ops": {k: v.clone() for k, v in self._module_to_op.items()},
            "hooks": self._hooks,
            "ctx": self._ctx,
            "input": self._input,
            "output": self._output,
        }
        return state

    def __setstate__(self, state: dict):
        self._model = state["model"]
        self._module_to_op: dict[str, list[Op]] = OrderedDict()

        self._hooks = OrderedDict()
        self._ctx = OrderedDict()
        self._input = OrderedDict()
        self._output = OrderedDict()
        self._pdb = False

        for k, v in state["ops"].items():
            v.apply(self)
            self._module_to_op[k] = v

        self._hooks = OrderedDict(state["hooks"])
        self._ctx = OrderedDict(state["ctx"])
        self._input = OrderedDict(state["input"])
        self._output = OrderedDict(state["output"])
        self._pdb = state["pdb"]

    def save(self, location):
        import dill

        with open(location, "wb") as fo:
            dill.dump(self, fo)

    @classmethod
    def load(cls, location):
        import dill

        with open(location, "rb") as fi:
            obj = dill.load(fi)
        return obj

    def get_handlers(self):
        """Get handlers"""
        return [Handler(hook_id, self) for hook_id in self._hooks.keys()]

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)
