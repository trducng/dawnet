import logging
import uuid
from collections import OrderedDict
from pdb import Pdb

import torch.nn as nn

from . import op
from .handler import Handler


logger = logging.getLogger(__name__)


class ModelRunner:

    def __init__(self, model: nn.Module, location: str | None = None):
        self._model = model
        self._location = location

        self._ops: dict[str, op.Op] = OrderedDict()
        self._hooks: dict[str, list[str]] = OrderedDict()

        self._ctx = OrderedDict()
        self._input = OrderedDict()
        self._output = OrderedDict()
        self._breaks = set()

        self._pdb = False

    @property
    def ctx(self):
        return self._ctx

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @classmethod
    def from_huggingface_hub(cls, *args, location: str = "", **kwargs):
        pass

    def add_forward_hooks(self, hook, *layers: str) -> Handler:
        """Add the hook that is meant to be run during model forward pass

        The returned handler can be used to remove the hooks.
        """
        ops = [op.ForwardHookOp(hook, layer) for layer in layers]
        return self.add_ops(*ops)

    def add_forward_pre_hooks(self, hook, *layers: str) -> Handler:
        """Set the pre-hooks to the forward pass, can be used to modify input"""
        ops = [op.ForwardPreHookOp(hook, layer) for layer in layers]
        return self.add_ops(*ops)

    def cache_outputs(self, *layers: str) -> Handler:
        """Store the output of the given layers to self.output"""
        ops = [op.CacheOutputOp(layer) for layer in layers]
        return self.add_ops(*ops)

    def cache_inputs(self, *layers: str) -> Handler:
        """Store the input of the given layers to self.input"""
        ops = [op.CacheInputOp(layer) for layer in layers]
        return self.add_ops(*ops)

    def cache_layers(self, *layers: str) -> Handler:
        """Store the input & output of the given layers to self.input & self.output"""
        ops = [op.CacheLayerOp(layer) for layer in layers]
        return self.add_ops(*ops)

    def add_ctx(self, **kwargs) -> Handler:
        """Add the context and store it inside the runner, allowing any hook to access it"""
        ops = [op.AddContextOp(key, value) for key, value in kwargs.items()]
        return self.add_ops(*ops)

    def swap_state_dict(self, *_layers, **kwargs) -> Handler:
        """Swap the layer state dict"""
        ops = [op.SwapStateDictOp(layer, **kwargs) for layer in _layers]
        return self.add_ops(*ops)

    def swap_module(self, layer: str, module: nn.Module) -> Handler:
        return self.add_ops(op.SwapModuleOp(layer, module))

    def run_from(self, *_layer_input_pairs) -> Handler:
        """Set the input of one or more layers, and start the model flow there.

        The `run_from` operation is different than `set_outputs` operation in that
        `run_from` will skip the forward pass of all layers before the ones set
        here.
        """
        pass

    def set_breakpoint(self, filename: str, lineno: int):
        """Set a breakpoint at a given file and line number"""
        return self.add_ops(op.SetPDBBreakpointOp(filename, lineno))

    def set_outputs(self, *_layer_output_pairs) -> Handler:
        """Set the output of 1 or more layers"""
        ops = []
        for idx in range(0, len(_layer_output_pairs), 2):
            layer = _layer_output_pairs[idx]
            output = _layer_output_pairs[idx + 1]
            ops.append(op.SetOutputOp(layer, output))

        return self.add_ops(*ops)

    def module(self, name):
        """Get module name"""
        attrs = name.split(".")
        m = self._model
        for attr in attrs:
            if attr.isnumeric():
                m = m[int(attr)]
            else:
                m = getattr(m, attr)

        return m

    def add_ops(self, *ops: op.Op) -> Handler:
        hook_id = f"hook_{str(uuid.uuid4())}"
        self._hooks[hook_id] = []

        for op_ in ops:
            op_.apply(self)
            op_id = f"op_{str(uuid.uuid4())}"
            self._ops[op_id] = op_
            self._hooks[hook_id].append(op_id)

        return Handler(hook_id=hook_id, runner=self)

    def delete_op(self, op_id, ignore_errors: bool = True):
        if op_id not in self._ops and not ignore_errors:
            raise ValueError(f"Operation with id {op_id} doesn't exist")
        del self._ops[op_id]

    @property
    def pdb(self):
        return self._pdb

    @pdb.setter
    def pdb(self, value: bool):
        self._pdb = value

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
            "location": self._location,
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
        self._location = state["location"]
        self._ops: dict[str, op.Op] = OrderedDict()

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
