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

    def run_from(self, *_layer_input_pairs) -> Handler:
        """Set the input of one or more layers, and start the model flow there.

        The `run_from` operation is different than `set_outputs` operation in that
        `run_from` will skip the forward pass of all layers before the ones set
        here.
        """
        pass

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
