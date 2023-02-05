"""
Purpose of this code is to work with hooks. The forward hook is
- hook(module, input, output)
But in case you want to arbitrarily modify a specific module, you don't know where
it is, you will need a global place to enable/disable that hook
"""

import torch


