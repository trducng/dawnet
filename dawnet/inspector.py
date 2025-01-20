from contextlib import contextmanager
from dataclasses import dataclass
import logging
import uuid
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)

@dataclass
class OpInfo:
    op: "Op"
    layer: str
    enabled: bool


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
        if self._name not in self._inspector._module_to_op:
            return args, kwargs

        args, kwargs = args, kwargs
        for op in self._inspector._module_to_op[self._name]:
            if self._inspector._ops[op.id].enabled:
                if self._inspector._debug > 3:
                    print(f"(v) {self._name}: {op} forward_pre")
                args, kwargs = op.forward_pre(
                    self._inspector, self._name, module, args, kwargs
                )

        return args, kwargs

    def forward(self, module, args, kwargs, output):
        if self._name not in self._inspector._module_to_op:
            return output

        output = output
        for op in reversed(self._inspector._module_to_op[self._name]):
            if self._inspector._ops[op.id].enabled:
                if self._inspector._debug > 3:
                    print(f"(v) {self._name}: {op} forward")
                output = op.forward(
                    self._inspector, self._name, module, args, kwargs, output
                )

        return output

    def backward_pre(self, module, grad_output):
        if self._name not in self._inspector._module_to_op:
            return grad_output

        grad_output = grad_output
        for op in self._inspector._module_to_op[self._name]:
            if self._inspector._ops[op.id].enabled:
                if self._inspector._debug > 3:
                    print(f"(v) {self._name}: {op} backward_pre")
                grad_output = op.backward_pre(
                    self._inspector, self._name, module, grad_output
                )

        return grad_output

    def backward(self, module, grad_input, grad_output):
        if self._name not in self._inspector._module_to_op:
            return grad_input

        grad_input = grad_input
        for op in self._inspector._module_to_op[self._name]:
            if self._inspector._ops[op.id].enabled:
                if self._inspector._debug > 3:
                    print(f"(v) {self._name}: {op} backward")
                grad_input = op.backward(
                    self._inspector, self._name, module, grad_input, grad_output
                )

        return grad_input


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

    def inspector_pre_run(self, inspector: "Inspector"):
        ...

    def inspector_post_run(self, inspector: "Inspector"):
        ...

    def add(self, inspector: "Inspector"):
        pass

    def remove(self, inspector: "Inspector"):
        pass

    def enable(self, inspector: "Inspector"):
        pass

    def disable(self, inspector: "Inspector"):
        pass

    def run_params(self, *args, **kwargs) -> dict:
        return {"id": self.id, **kwargs}


class RunState:
    def __init__(self):
        self._defs = {}
        self._states = {}
        self.register("input", {})
        self.register("output", {})

    def clear(self, names: list[str] | str | None = None):
        """Clear the section, if not specified, clear all"""
        if names is not None:
            if not isinstance(names, list):
                names = [names]
            for name in names:
                if name in self._defs:
                    self._states[name] = deepcopy(self._defs[name])
                else:
                    logger.warning(f"State with name {name} doesn't exist")
            return

        for name, default in self._defs.items():
            self._states[name] = deepcopy(default)

    def __contains__(self, name):
        return name in self._states

    def register(self, name, default=None):
        self._defs[name] = default
        self._states[name] = deepcopy(default)

    def deregister(self, name):
        self._states.pop(name, None)
        self._defs.pop(name, None)

    def keys(self):
        return self._states.keys()

    def values(self):
        return self._states.values()

    def items(self):
        return self._states.items()

    def __getitem__(self, key):
        if key not in self._defs:
            raise KeyError(f"State with name {key} doesn't exist")
        return self._states[key]

    def __setitem__(self, key, value):
        if key not in self._defs:
            raise KeyError(
                f"State with name {key} doesn't exist. Please .register first"
            )
        self._states[key] = value


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

    Args:
        model: the model to inspect
        debug: the debug level: 0 - no debug, 1 - full error stacktrace, 2 - warning,
            3 - info, 4 - debug

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
        debug: int=0,
    ):
        super().__init__()
        self._model = copy_model(model)
        self._original_model = model
        self._debug = debug

        self._module_to_op: dict[str, list[Op]] = OrderedDict()
        self._ops: dict[str, OpInfo] = OrderedDict()  # op, module name, enable
        self.ops = []

        self.state: RunState = state or RunState()
        self._private_op_state = {}
        self._op_params: dict = {}

        for name, module in self._model.named_modules():
            module.register_forward_pre_hook(
                Handler(name, "forward_pre", self), with_kwargs=True
            )
            module.register_forward_hook(
                Handler(name, "forward", self), with_kwargs=True, always_call=True
            )

    @property
    def model(self):
        return self._model

    @property
    def original_model(self):
        return self._original_model

    def add(self, name: str, op: Op) -> Op:
        """Add op to the inspector

        Returns:
            the op id
        """
        if name == "" or self._model.get_submodule(name):
            if name not in self._module_to_op:
                self._module_to_op[name] = []
            self._module_to_op[name].append(op)
            self._ops[op.id] = OpInfo(op=op, layer=name, enabled=True)
            self.ops.append(op)
            op.add(self)

            return op
        else:
            raise ValueError(f"Module with name {name} doesn't exist")

    def remove(self, op: Op):
        if op.id not in self._ops:
            raise ValueError(f"Op with id {op.id} doesn't exist")
        op.remove(self)
        layer_name = self._ops[op.id].layer
        self._module_to_op[layer_name] = [
            each for each in self._module_to_op[layer_name] if each.id != op.id
        ]
        del self._ops[op.id]
        self.ops = [each for each in self.ops if each.id != op.id]

    def enable(self, op: Op):
        if op.id not in self._ops:
            raise ValueError(f"Op with id {op.id} doesn't exist")
        self._ops[op.id].enabled = True
        op.enable(self)

    def disable(self, op: Op):
        if op.id not in self._ops:
            raise ValueError(f"Op with id {op.id} doesn't exist")
        self._ops[op.id].enabled = False
        op.disable(self)

    def copy(self) -> "Inspector":
        """Create a copy of the inspector"""
        return Inspector(self._original_model)

    def __str__(self):
        if not self._ops:
            return "No ops added"

        strs = ["Inspector ops:"]
        for idx, op in enumerate(self.ops):
            op_info = self._ops[op.id]
            s = f"- [{idx}] {op} @ {op_info.layer}"
            if not op_info.enabled:
                s += " (disabled)"
            strs.append(s)
        return "\n".join(strs)

    def __repr__(self):
        return repr(self._model)

    def __setattr__(self, name, value):
        if name in ["_model", "_original_model"]:
            self.__dict__[name] = value
            return
        return super().__setattr__(name, value)

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

        for k, v in state["ops"].items():
            v.apply(self)
            self._module_to_op[k] = v

        self._hooks = OrderedDict(state["hooks"])
        self._ctx = OrderedDict(state["ctx"])
        self._input = OrderedDict(state["input"])
        self._output = OrderedDict(state["output"])

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

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def begin(self, op_params: list[dict] | None = None) -> RunState:
        if op_params:
            for op_param in op_params:
                if op_param["id"] in self._op_params:
                    logger.warning(
                        f"Op with id {op_param['id']} already exists, overwriting"
                    )
                self._op_params[op_param["id"]] = op_param

        for op_info in self._ops.values():
            if op_info.enabled:
                op_info.op.inspector_pre_run(self)

        return self.state

    def finish(self):
        for op_info in reversed(self._ops.values()):
            if op_info.enabled:
                op_info.op.inspector_post_run(self)
        self._op_params.clear()
        self.state.clear()

    @contextmanager
    def ctx(self, op_params: list[dict] | None = None):
        state = self.begin(op_params)
        try:
            yield state
        finally:
            self.finish()

    def run(
        self,
        *args,
        _refresh_state: bool = True,
        _method: str | None = None,
        _op_params: list[dict] | None = None,
        **kwargs,
    ):
        """Run the model"""
        if _refresh_state:
            self.state.clear()
        self._op_params.clear()
        if _op_params:
            for op_param in _op_params:
                if op_param["id"] in self._op_params:
                    logger.warning(
                        f"Op with id {op_param['id']} already exists, overwriting"
                    )
                self._op_params[op_param["id"]] = op_param

        for op_info in self._ops.values():
            if op_info.enabled:
                op_info.op.inspector_pre_run(self)

        if _method:
            output = getattr(self._model, _method)(*args, **kwargs)
        else:
            output = self._model(*args, **kwargs)

        for op_info in reversed(self._ops.values()):
            if op_info.enabled:
                op_info.op.inspector_post_run(self)

        return output, self.state
