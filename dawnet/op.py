import logging
import sys
from typing import Callable

import torch.nn as nn

from .inspector import Inspector, Op, Handler


logger = logging.getLogger(__name__)


class NotSet:
    def __bool__(self):
        return False

    def __eq__(self, other):
        return isinstance(other, NotSet)

    def __ne__(self, other):
        return not isinstance(other, NotSet)

    def __repr__(self):
        return "notset"


notset = NotSet()


class GetInputOutput(Op):
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


class GetOutput(GetInputOutput):
    """Short-hand for GetInputOutput with only output cached"""

    def __init__(self, output_getter: Callable | None = None):
        super().__init__(no_input=True, output_getter=output_getter)

    def __str__(self):
        return "GetOutput"


class GetInput(GetInputOutput):
    """Short-hand for GetInputOutput with only input cached"""

    def __init__(self, input_getter: Callable | None = None):
        super().__init__(no_output=True, input_getter=input_getter)

    def __str__(self):
        return "GetInput"


class GetGradient(Op):
    """Get the gradient after backward pass"""

    def backward(
        self, inspector: "Inspector", name: str, module, grad_input, grad_output
    ):
        inspector.state["grad_output"][name] = grad_output
        inspector.state["grad_input"][name] = grad_input
        return grad_input

    def add(self, inspector: "Inspector"):
        inspector.state.register("grad_output", {})
        inspector.state.register("grad_input", {})

    def __str__(self):
        return "GetGradient"


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

        old_state = {}
        for layer in inspector._ops[self.id].layers:
            module = inspector._model.get_submodule(layer)
            if self.id not in inspector._private_op_state:
                old_state[layer] = module.state_dict()
                module.load_state_dict(self._state_dict)

        inspector._private_op_state[self.id] = old_state

    def disable(self, inspector: "Inspector"):
        if self.id not in inspector._private_op_state:
            return

        for layer in inspector._ops[self.id].layers:
            module = inspector._model.get_submodule(layer)
            module.load_state_dict(inspector._private_op_state[self.id][layer])

        del inspector._private_op_state[self.id]


class SwapModule(Op):
    """Swap the module of a layer"""

    def __init__(self, module: nn.Module):
        super().__init__()
        self._module = module

    def add(self, inspector: "Inspector"):
        # TODO: check if the module is already swapped
        # TODO: inform about operations of the original child module will be ignored
        layers = inspector._ops[self.id].layers
        if len(layers) > 1:
            raise ValueError("SwapModule can only swap one layer at a time")

        name = list(layers)[0]
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

        layers = inspector._ops[self.id].layers
        if len(layers) > 1:
            raise ValueError("SwapModule can only swap one layer at a time")

        full_module_name = list(layers)[0]
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

        layers = inspector._ops[self.id].layers
        if len(layers) > 1:
            raise ValueError("SwapModule can only swap one layer at a time")

        full_module_name = list(layers)[0]
        if "." not in full_module_name:
            parent = inspector._model
            module_name = full_module_name
        else:
            module_parent, module_name = full_module_name.rsplit(".", 1)
            parent = inspector._model.get_submodule(module_parent)

        parent._module[module_name] = inspector._private_op_state[self.id]
        del inspector._private_op_state[self.id]


class SetInputOutput(Op):

    def forward(self, inspector: "Inspector", name: str, module, args, kwargs, output):
        if self.id in inspector._op_params:
            if "output" in inspector._op_params[self.id]:
                return inspector._op_params[self.id]["output"]
            if "output_fn" in inspector._op_params[self.id]:
                return inspector._op_params[self.id]["output_fn"](output)
        return output

    def forward_pre(self, inspector: "Inspector", name: str, module, args, kwargs):
        if self.id in inspector._op_params:
            if "input" in inspector._op_params[self.id]:
                return inspector._op_params[self.id]["input"]
            if "input_fn" in inspector._op_params[self.id]:
                return inspector._op_params[self.id]["input_fn"](args, kwargs)
        return args, kwargs

    def run_params(
        self,
        input=notset,
        output=notset,
        input_fn: Callable | NotSet = notset,
        output_fn: Callable | NotSet = notset,
    ):
        """Supply input, output

        Args:
            input: the input value to be set. Cannot be used with input_fn.
            output: the output value to be set. Cannot be used with output_fn.
            input_fn: a callback function to set the input, it will take
                [args], {kwargs} and expect to return new [args], {kwargs}. Cannot
                be used with input.
            output_fn: a callback function to set the output, it will take
                the output object and expect to return a new output object. Cannot
                be used with output.
        """
        params = {}
        if input != notset:
            params["input"] = input
        if output != notset:
            params["output"] = output
        if input_fn != notset:
            params["input_fn"] = input_fn
        if output_fn != notset:
            params["output_fn"] = output_fn

        if "input" in params and "input_fn" in params:
            raise ValueError("Cannot set both input and input_fn")

        if "output" in params and "output_fn" in params:
            raise ValueError("Cannot set both output and output_fn")

        return super().run_params(**params)


class SetOutput(SetInputOutput):
    """Short-hand for SetInputOutput with only output set"""

    def forward_pre(self, inspector: "Inspector", name: str, module, args, kwargs):
        return args, kwargs

    def run_params(self, output=notset, output_fn=notset):  # type: ignore
        return super().run_params(output=output, output_fn=output_fn)

    def __str__(self):
        return "SetOutput"


class SetInput(SetInputOutput):
    """Short-hand for SetInputOutput with only input set"""

    def forward(self, inspector: "Inspector", name: str, module, args, kwargs, output):
        return output

    def run_params(self, input=notset, input_fn=notset):  # type: ignore
        return super().run_params(input=input, input_fn=input_fn)

    def __str__(self):
        return "SetInput"


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
