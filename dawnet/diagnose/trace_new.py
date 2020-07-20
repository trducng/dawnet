# Trace the behavior of agent
# @author: johntd54
# ====================================================================================
from collections import OrderedDict

import torch
import torch.nn as nn

from dawnet.utils.dependencies import get_pytorch_layers


def draw_path():
    """
    Likely graphvz
    https://transcranial.github.io/keras-js/#/inception-v3
    tensorspace
    plotly:https://plotly.com/python/network-graphs/
    netron
    Currently it is not easy to graciously trace the model with pure Python, due to
    some functional operations cannot log the operation (e.g. F.relu, torch.cat)
    Also tracking reshaping tensor is hard.
    A better way might be converting to ONNX and examine the block there
    Maybe combining between register_forward_hook, register_backward_hook

    In the mean time, if you want to use old method, make sure to wrap everything
    inside nn.Module in order to use the hooks.

    Or use ONNX, other hacks might not be able to work in future Pytorch versions.
    ONNX on the other hand can be actively maintained.

    Or use torch.jit.trace to turn the model into TorchScript.

    Good enough candidate: torchviz, tensorboardx, hiddenlayer

    """
    pass


def visualize():
    """Maybe using this one:
        https://github.com/utkuozbulak/pytorch-cnn-visualizations
    """
    pass

class Tracer():

    def __init__(self):
        self.dict = OrderedDict()
        self.handlers = []

    def apply(self, agent):
        """Add tracing to all modules"""

        def hook(module, input_, output):
            if not isinstance(module, get_pytorch_layers()):
                return

            idx = 0 if not self.dict else max(list(self.dict.keys())) + 1
            self.dict[idx] = {
                'module': module,
                'input': input_,
                'output': output
            }

        def add_hook(module):
            if isinstance(module, get_pytorch_layers()):
                handler = module.register_forward_hook(hook)
                self.handlers.append(handler)

        agent.apply(add_hook)

    def reset(self, detach_handler=False):
        self.dict = OrderedDict()

        if detach_handler:
            for handler in self.handlers:
                handler.remove()

        self.handlers = []
