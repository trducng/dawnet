# Trace the behavior of a convnet model.
# Zeiler and Fergus provided some great explanations and inspiration
# for the idea in this code: https://arxiv.org/abs/1311.2901
# Currently we should take notice of these issues:
#   - unfilter: the paper use inverse of conv weights, while we just want to see
#       the position of input neurons that are responsible for the output
#   - the aim of authors is actually to reconstruct the image as it influences
#       the neuron, while my aim is just to see the rectangular region in the
#       image that is responsible for neuron output
#   - for the matter above, we can actually construct 2 modes of diagnoses
#   - when viewing the convolutional weights, it is important to normalize
#       those weights with min and max of the whole channels in that layer,
#       not only of that channel
#   - it is important to stack all the decoding layers to create a decoding
#       model
# @author: _john
# =============================================================================

import torch
import torch.nn as nn


def trace_maxpool2d(indices, output_map, kernel_size, stride=None, padding=0,
                    dilation=1):
    """Find indices of acting input neurons
    
    Find the indices of neurons in the input that contribute to specified
    output. This method either do an unmaxpool2d operation if `idx` is None or
    return the indices of region that create idx

    @NOTE: it might be better to use the `return_indices` from `nn.MaxPoolNd`
    and then use `nn.MaxUnpool2d`

    @TODO: the dilation is basically useless

    # Arguments:
        indices [tuple of 2 ints]: the y and x indices (0-counting)
                [IntTensor]: the `return_indices` from MaxPool2d
        output_map [torch Tensor]: the output map
        kernel_size [int or tuple of 2 ints]: the kernel size of maxpool layer
        stride [int or tuple of 2 ints]: the stride of maxpool layer
        padding [int or tuple of 2 ints]: the padding value of maxpool layer
        dilation [int or tuple of 2 ints]: the dilation value of maxpool layer
    
    # Returns
        [tuple of (top, bottom, left, right)]: the contributive region if `idx`
            is not None
        [torch Tensor]: the reconstructed input map if `idx` is None
    """
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    
    if stride is None:
        stride = kernel_size

    if isinstance(indices, torch.Tensor):
        unmaxpool = nn.MaxUnpool2d(kernel_size, stride=stride, padding=padding)
        return unmaxpool(output_map)
    elif isinstance(indices, tuple) and len(indices) == 2:
        y, x = indices
        # @TODO: this is basically wrong as we have not take into account the
        # value of `slide` != `kernel_size`
        top = y * stride[0]
        bottom = top + kernel_size[0]
        left = x * stride[1]
        right = left + kernel_size[1]
        return top, bottom, left, right
    
    raise AttributeError('`indices` should be output feature map or a '
                         'tuple of 2 ints (y, x)')

