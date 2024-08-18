import logging
import uuid
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn
from platformdirs import user_cache_dir
from torch.utils.hooks import RemovableHandle

if TYPE_CHECKING:
    from .model import ModelRunner


logger = logging.getLogger(__name__)


class Op:
    """Define base methods for an operation.

    An operation is what we apply to the model. For example, saving the intermediate
    layer output is an operation, changing the intermediate output is an operation...
    """

    def apply(self, runner: "ModelRunner"):
        """Apply the operation to the model runner"""
        raise NotImplemented("Need to implement `op.apply` method")

    def clear(self, runner: "ModelRunner"):
        """Clear the operation from the model runner"""
        raise NotImplemented("Need to implement `op.clear` method")

    def clone(self):
        raise NotImplemented("Need to implement `op.clone` method")

class ForwardHookOp(Op):
    """Attach forward hook"""

    def __init__(self, hook: Callable, layer: str):
        self._hook = hook
        self._layer = layer.strip(".")
        self._pt_hook: RemovableHandle | None = None

    def __str__(self):
        return f"Add forward hook to layer {self._layer}: {self._hook.__doc__}"

    def apply(self, runner: "ModelRunner"):
        if self._pt_hook is not None:
            raise ValueError(
                "This operation is already applied. Run `.clone` to create new op"
            )

        module = runner.module(self._layer)
        self._pt_hook = module.register_forward_hook(
            partial(self._hook, runner, self._layer), with_kwargs=True
        )

    def clear(self, runner: "ModelRunner"):
        if self._pt_hook is not None:
            self._pt_hook.remove()

    def clone(self):
        return ForwardHookOp(self._hook, self._layer)


class ForwardPreHookOp(Op):
    """Attach forward pre-hook to an nn.Module layer

    Args:
        hook: a function with the following signature:
            hook(runner, layer, layer_obj, input_args, input_kwargs) -> (args, kwargs)
        layer: the layer name to add this hook
    """
    def __init__(self, hook: Callable, layer: str):
        self._hook = hook
        self._layer = layer.strip(".")
        self._pt_hook: RemovableHandle | None = None

    def __str__(self):
        return f"Add forward pre-hook to layer {self._layer}: {self._hook.__doc__}"

    def apply(self, runner: "ModelRunner"):
        if self._pt_hook is not None:
            raise ValueError(
                "This operation is already applied. Run `.clone` to create new op"
            )

        module = runner.module(self._layer)
        self._pt_hook = module.register_forward_pre_hook(
            partial(self._hook, runner, self._layer), with_kwargs=True
        )

    def clear(self, runner: "ModelRunner"):
        if self._pt_hook is not None:
            self._pt_hook.remove()

    def clone(self):
        return ForwardPreHookOp(self._hook, self._layer)


class CacheOutputOp(Op):
    """Cache the output of a layer"""
    def __init__(self, layer):
        self._layer = layer.strip(".")
        self._pt_hook: RemovableHandle | None = None

    def __str__(self):
        return f"Cache output of layer {self._layer}"

    def apply(self, runner: "ModelRunner"):
        def cache_output_hook(r, n, l, i, o):
            r._output[n] = o
            return o

        module = runner.module(self._layer) 
        self._pt_hook = module.register_forward_hook(
            partial(cache_output_hook, runner, self._layer)
        )

    def clear(self, runner: "ModelRunner"):
        if self._pt_hook is not None:
            self._pt_hook.remove()

        if self._layer in runner._output:
            del runner._output[self._layer]

    def clone(self):
        return CacheOutputOp(self._layer)


class CacheInputOp(Op):
    """Cache the input of a layer"""

    def __init__(self, layer):
        self._layer = layer.strip(".")
        self._pt_hook: RemovableHandle | None = None

    def __str__(self):
        return f"Cache input of layer {self._layer}"

    def apply(self, runner: "ModelRunner"):
        def cache_input_hook(r, n, l, ia, ik, o):
            r._input[n] = {"args": ia, "kwargs": ik}
            return o

        module = runner.module(self._layer) 
        self._pt_hook = module.register_forward_hook(
            partial(cache_input_hook, runner, self._layer), with_kwargs=True
        )

    def clear(self, runner: "ModelRunner"):
        if self._pt_hook is not None:
            self._pt_hook.remove()

        if self._layer in runner._input:
            del runner._input[self._layer]

    def clone(self):
        return CacheInputOp(self._layer)


class CacheLayerOp(Op):
    """Cache the input and output of a layer"""

    def __init__(self, layer):
        self._layer = layer.strip(".")
        self._pt_hook: RemovableHandle | None = None

    def __str__(self):
        return f"Cache input and output of layer {self._layer}"

    def apply(self, runner: "ModelRunner"):
        def cache_layer_hook(r, n, l, ia, ik, o):
            r._input[n] = {"args": ia, "kwargs": ik}
            r._output[n] = o
            return o

        module = runner.module(self._layer) 
        self._pt_hook = module.register_forward_hook(
            partial(cache_layer_hook, runner, self._layer), with_kwargs=True
        )

    def clear(self, runner: "ModelRunner"):
        if self._pt_hook is not None:
            self._pt_hook.remove()

        if self._layer in runner._input:
            del runner._input[self._layer]

        if self._layer in runner._output:
            del runner._output[self._layer]

    def clone(self):
        return CacheLayerOp(self._layer)


class SetOutputOp(Op):
    """Set the output of a layer to a fixed value"""
    def __init__(self, layer, output):
        self._layer = layer.strip(".")
        self._output = output
        self._pt_hook: RemovableHandle | None = None

    def __str__(self):
        return f"Apply fixed output to layer {self._layer}"

    def apply(self, runner:" ModelRunner"):
        def set_output_hook(r, n, l, i, o):
            return o

        module = runner.module(self._layer)
        self._pt_hook = module.register_forward_hook(
            partial(set_output_hook, runner, self._layer, o=self._output)
        )

    def clear(self, runner: "ModelRunner"):
        if self._pt_hook is not None:
            self._pt_hook.remove()


class AddContextOp(Op):
    """Add context to the runner"""
    def __init__(self, key, value):
        self._key = key
        self._value = value

        self._applied = False

    def __str__(self):
        return f"Add context '{self._key}' to the runner"

    def apply(self, runner: "ModelRunner"):
        if self._applied:
            raise ValueError(
                "This operation is already applied. Run `.clone` to create new op"
            )
        runner._ctx[self._key] = self._value
        self._applied = True

    def clear(self, runner: "ModelRunner"):
        if self._key in runner._ctx:
            del runner._ctx[self._key]

    def clone(self):
        return AddContextOp(self._key, self._value)


class SwapStateDictOp(Op):
    """Swap the state dict"""
    def __init__(self, layer, **updated_dict):
        self._layer = layer.strip(".")
        self._updated_dict = updated_dict
        self._backup_path: Path | None = None

    def __str__(self):
        return f"Change state dict {self._updated_dict.keys()} of layer {self._layer}"

    def apply(self, runner: "ModelRunner"):
        if self._backup_path is not None:
            raise ValueError(
                "This operation is already applied. Run `.clone` to create new op"
            )

        self._backup_path = Path(user_cache_dir(appname="dawnet", ensure_exists=True))
        self._backup_path = self._backup_path / f"ssd_{uuid.uuid4()}"

        module = runner.module(self._layer)
        # @TODO: the state_dict here might not be original state_dict.
        state_dict = module.state_dict()

        torch.save(state_dict, self._backup_path)
        state_dict.update(**self._updated_dict)
        module.load_state_dict(state_dict)

    def clear(self, runner: "ModelRunner"):
        if not self._backup_path:
            raise ValueError("This operation hasn't been applied")

        if not self._backup_path.is_file():
            logger.warning(f"State dict doesn't exist. Skip reverting")
            return

        state_dict = torch.load(self._backup_path)
        module = runner.module(self._layer)
        module.load_state_dict(state_dict)
        self._backup_path.unlink()
        self._backup_path = None

    def clone(self):
        return SwapStateDictOp(self._layer, **self._updated_dict)


class SwapModuleOp(Op):
    """Swap an existing module with a new module

    Note: swapping a module will not transfer the hook of the old module to the
    new module.

    @TODO: check for missing hooks
    """
    def __init__(self, layer: str, module: nn.Module):
        self._layer = layer.strip(".")
        self._module = module

        self._ori_module: nn.Module | None = None

    def __str__(self):
        return f"Swap the module for layer {self._layer}"

    def apply(self, runner: "ModelRunner"):
        if self._ori_module is not None:
            raise ValueError(
                "This operation is already applied. Run `.clone` to create new op"
            )
        self._ori_module = runner.module(self._layer)
        layers = self._layer.split(".")
        if len(layers) > 1:
            parent_layer = ".".join(layers[:-1])
            child_name = layers[-1]
            parent_module = runner.module(parent_layer)
            if child_name.isnumeric():
                parent_module[int(child_name)] = self._module
            else:
                setattr(parent_module, child_name, self._module)
        elif len(layers) == 1:
            self._model = self._module
        else:
            raise ValueError(f"Invalid layer {self._layer}")

    def clear(self, runner: "ModelRunner"):
        if self._ori_module is None:
            logger.warning("Module hasn't been applied. Skip.")
            return

        layers = self._layer.split(".")
        if len(layers) > 1:
            parent_layer = ".".join(layers[:-1])
            child_name = layers[-1]
            parent_module = runner.module(parent_layer)
            if child_name.isnumeric():
                parent_module[int(child_name)] = self._ori_module
            else:
                setattr(parent_module, child_name, self._ori_module)
        elif len(layers) == 1:
            runner._model = self._ori_module
        else:
            raise ValueError(f"Invalid layer {self._layer}")

    def clone(self):
        return SwapModuleOp(self._layer, self._module)


class SetPDBBreakpointOp(Op):
    """Set the breakpoint when pdb mode is enabled"""
    def __init__(self, filename: str, lineno: int):
        self._filename = filename
        self._lineno = lineno

    def __str__(self):
        return f"Set breakpoint at {self._filename}:{self._lineno}"

    def apply(self, runner: "ModelRunner"):
        brp = (str(self._filename), int(self._lineno))
        if brp in runner._breaks:
            raise ValueError("The breakpoint is already set")
        runner._breaks.add(brp)

    def clear(self, runner: "ModelRunner"):
        brp = (str(self._filename), int(self._lineno))
        if brp in runner._breaks:
            runner._breaks.remove(brp)

    def clone(self):
        return SetPDBBreakpointOp(self._filename, self._lineno)
