import logging
import uuid
from collections import OrderedDict
from pdb import Pdb

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

    def forward(self, module, args, kwargs, output):
        output = output
        if self._name in self._inspector._ops:
            for op in self._inspector._ops[self._name]:
                output = op.forward(
                    self._inspector, self._name, module, args, kwargs, output
                )
        return output

    def forward_pre(self, module, args, kwargs):
        args, kwargs = args, kwargs
        if self._name in self._inspector._ops:
            for op in self._inspector._ops[self._name]:
                args, kwargs = op.forward_pre(
                    self._inspector, self._name, module, args, kwargs
                )
        return args, kwargs

    def backward(self, module, grad_input, grad_output):
        grad_input = grad_input
        if self._name in self._inspector._ops:
            for op in self._inspector._ops[self._name]:
                grad_input = op.backward(
                    self._inspector, self._name, module, grad_input, grad_output
                )
        return grad_input

    def backward_pre(self, module, grad_output):
        grad_output = grad_output
        if self._name in self._inspector._ops:
            for op in self._inspector._ops[self._name]:
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


class RunState(nn.Module):
    def clear(self):
        pass


def copy_model(module: nn.Module):
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
        _state: contain information that the op populates (e.g. this is where
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

        self._ops: dict[str, list[Op]] = OrderedDict()
        self.state: RunState = state or RunState()
        self._state = {}

        for name, module in self._model.named_modules():
            module.register_forward_pre_hook(
                Handler(name, "forward_pre", self), with_kwargs=True
            )
            module.register_forward_hook(
                Handler(name, "forward", self), with_kwargs=True, always_call=True
            )

    def register(self, name: str, op: Op):
        if name == "." or self._model.get_submodule(name):
            if name not in self._ops:
                self._ops[name] = []
            self._ops[name].append(op)
        else:
            raise ValueError(f"Module with name {name} doesn't exist")

    def deregister(self, op_id: str, name: str | None = None):
        pass

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
            "ops": {k: v.clone() for k, v in self._ops.items()},
            "hooks": self._hooks,
            "ctx": self._ctx,
            "input": self._input,
            "output": self._output,
            "pdb": self._pdb,
        }
        return state

    def __setstate__(self, state: dict):
        self._model = state["model"]
        self._ops: dict[str, list[Op]] = OrderedDict()

        self._hooks = OrderedDict()
        self._ctx = OrderedDict()
        self._input = OrderedDict()
        self._output = OrderedDict()
        self._pdb = False

        for k, v in state["ops"].items():
            v.apply(self)
            self._ops[k] = v

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

    def run(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)
