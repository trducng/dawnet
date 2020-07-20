# Dawnet is based on Pytorch, hence the need for some knowledge about
# those frameworks.
# @author: _john
# ============================================================================
import inspect
import sys

from IPython.display import Markdown, display
import torch
import torch.nn as nn


# module classes but are not exactly layer
NON_LAYER_MODULES = set([
    "Module", "ModuleDict", "ModuleList",
    "Parameter", "ParameterDict", "ParameterList",
    "DataParallel", "Sequential"
])

# All valid Pytorch layers
VALID_LAYERS = []
for name, obj in inspect.getmembers(nn):
    if not inspect.isclass(obj):
        continue

    if name in NON_LAYER_MODULES:
        continue

    if 'loss' in name.lower():
        continue

    VALID_LAYERS.append(obj)


def get_pytorch_layers():
    """Get valid pytorch layers

    This function retrieve all classes from `torch.nn.modules...`.

    # Returns:
        [set of class objects]: list of Pytorch layers
    """
    return tuple(VALID_LAYERS)


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
