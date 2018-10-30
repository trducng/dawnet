# Dawnet is based on Pytorch, hence the need for some knowledge about
# those frameworks.
# @author: _john
# ============================================================================
import inspect
import sys

from IPython.display import Markdown, display
import torch
import torch.nn


MODULES = [
    torch.nn.modules.conv,
    torch.nn.modules.linear,
    torch.nn.modules.activation,
    torch.nn.modules.pooling,
    torch.nn.modules.batchnorm,
    torch.nn.modules.instancenorm,
    torch.nn.modules.rnn
]

SKIP_LAYERS = set([
    torch.nn.modules.module.Module,
    torch.nn.parameter.Parameter
])

ALL_LAYERS = []
for each_module in MODULES:
    for _, obj in inspect.getmembers(each_module):
        if inspect.isclass(obj):
            ALL_LAYERS.append(obj)
ALL_LAYERS = set(ALL_LAYERS)
VALID_LAYERS = ALL_LAYERS.difference(SKIP_LAYERS)


def get_pytorch_layers():
    """Get valid pytorch layers

    This function retrieve all classes from `torch.nn.modules...`.

    # Returns:
        [set of class objects]: list of Pytorch layers
    """
    return VALID_LAYERS


def print_md(string):
    """Print the text with Markdown, mostly for used in Jupyter notebook

    # Arguments
        string [str]: the string to print
    """
    display(Markdown(string))


def colored_md(string, color='black'):
    """Construct the string for Markdown printing

    # Arguments
        string [str]: the string to construct
        color [str]: the color used for CSS `style` color
    
    # Returns
        [str]: the constructed string with desired format
    """
    return '<span style="color: {}">{}</span>'.format(color, string)
