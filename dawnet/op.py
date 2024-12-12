import logging
import sys
from typing import Callable

import torch.nn as nn

from .inspector import Inspector, Op, Handler


logger = logging.getLogger(__name__)


class CacheModuleInputOutput(Op):
    """Cache the input and output of a module

    Args:
        no_input: if True, don't cache the input
        no_output: if True, don't cache the output
        input_getter: a callback to get the desired input, should take [args], {kwargs}
        output_getter: a callback to get the output, should take `output` object
    """

    def __init__(
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
            inspector.state["output"][name] = output
        else:
            inspector.state["output"][name] = self._output_getter(output)

        return output

    def forward_pre(self, inspector: "Inspector", name: str, module, args, kwargs):
        if self._no_input:
            return args, kwargs

        if self._input_getter is None:
            inspector.state["input"][name] = args, kwargs
        else:
            inspector.state["input"][name] = self._input_getter(args, kwargs)

        return args, kwargs


class Hook(Op):
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

    def __str__(self):
        kwargs = {}
        if self._forward is not None:
            kwargs["forward"] = self._forward.__name__
        if self._forward_pre is not None:
            kwargs["forward_pre"] = self._forward_pre.__name__
        if self._backward is not None:
            kwargs["backward"] = self._backward.__name__
        if self._backward_pre is not None:
            kwargs["backward_pre"] = self._backward_pre.__name__
        return f"Hook({','.join(k+'='+v for k, v in kwargs.items())})"

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

            pdb.botframe = None
            pdb._set_stopinfo(sys._getframe(), None)
            sys.settrace(pdb.trace_dispatch)

            inspector._private_op_state["in_break"] = pdb
            inspector._private_op_state["break_id"] = self.id
        else:
            pdb = inspector._private_op_state["in_break"]

        pdb.set_break(filename=self._filename, lineno=self._lineno)

        return args, kwargs

    def forward(self, inspector: "Inspector", name: str, module, args, kwargs, output):
        pdb = inspector._private_op_state["in_break"]
        pdb.clear_break(filename=self._filename, lineno=self._lineno)
        if inspector._private_op_state["break_id"] == self.id:
            # we should be the one to clean up
            sys.settrace(None)

            del inspector._private_op_state["in_break"]
            del inspector._private_op_state["break_id"]

        return output
