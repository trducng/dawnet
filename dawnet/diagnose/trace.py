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

from dawnet.models.convs import get_conv_input_shape


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
        [tuple of (y, x)]: the contributive region if `idx` is not None
        [torch Tensor]: the reconstructed input map if `idx` is None
    """
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    
    if stride is None:
        stride = kernel_size
    elif isinstance(stride, int):
        stride = [stride, stride]

    if isinstance(indices, torch.Tensor):
        unmaxpool = nn.MaxUnpool2d(kernel_size, stride=stride, padding=padding)
        return unmaxpool(output_map, indices)
    elif (isinstance(indices, tuple) or isinstance(indices, list)
          and len(indices) == 2):
        y, x = indices
        # @TODO: this is basically wrong as we have not take into account the
        # value of `slide` != `kernel_size`
        # @TODO: this also incurs problem as it does not take into account
        # `dilation` and `padding`
        indices_result = []

        top = y * stride[0]
        bottom = top + kernel_size[0]
        left = x * stride[1]
        right = left + kernel_size[1]

        for row in range(top, bottom):
            for col in range(left, right):
                indices_result.append((row, col))
        return indices_result

    raise AttributeError('`indices` should be output feature map or a '
                         'tuple of 2 ints (y, x)')


def trace_conv2d(indices, output_shape, kernel_size, stride=None, padding=0,
    dilation=1):
    """Find indices of acting input neuron

    @TODO: this `trace_conv2d` has not taken into account `dilation` argument

    # Arguments
        indices [tuple of 2 ints]: the y and x indices (0-counting)
        output_shape [torch Tensor]: the output map
        kernel_size [int or tuple of 2 ints]: the kernel size of maxpool layer
        stride [int or tuple of 2 ints]: the stride of maxpool layer
        padding [int or tuple of 2 ints]: the padding value of maxpool layer
        dilation [int or tuple of 2 ints]: the dilation value of maxpool layer

    # Returns
        [tuple of (y, x)]: the contributive region
    """
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]

    if isinstance(padding, int):
        padding = [padding, padding]
    
    if isinstance(stride, int):
        stride = [stride, stride]
    elif stride is None:
        stride = kernel_size
    
    if len(output_shape) != 2:
        raise AttributeError('2D convolution must have 2D shape')

                                                        # pylint: disable=E0632
    input_height, input_width = get_conv_input_shape(
        output_shape, kernel_size, stride, padding)     # input feature map size

    # get the input patch, recalibrated by padding value
    y, x = indices
    top = y * stride[0] - padding[0]
    bottom = top + kernel_size[0]
    left = x * stride[1] - padding[1]
    right = left + kernel_size[1]

    # remove the padding indices
    top = max(top, 0)
    bottom = min(bottom, input_height)
    left = max(left, 0)
    right = min(right, input_width)

    indices_result = []
    for row in range(top, bottom):
        for col in range(left, right):
            indices_result.append((row, col))

    return indices_result


def trace(layer_idx, indices, output_shape, model):
    """Trace the patch of the image that corresponds to a particular excitation

    # Arguments
        layer_idx [int]: index of the layer that contains interested `indices`
        indices [list of tuple of 2 ints]: multiple point, each tuple is
            represented by y and x indices (0-counting)
        output_shape [tuple of 2 ints]: width and height
        model [list of layers or torch nn.Module]: the model in a layer
            representation. If this is a torch Model, it will be converted
            to list of layers representation

    # Returns
        [tuples of (y, x)]: the contributive region
    """
    if layer_idx <= -1:
        return indices
    
    if not isinstance(model, list):
        model = list(model.named_modules())
    
    if isinstance(indices, tuple) and len(indices) == 2:
        indices = [indices]
    
    result = []
    layer = model[layer_idx][1]
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
        kernel_size = layer.kernel_size
        stride = layer.stride
        padding = layer.padding
        dilation = layer.dilation
        for each_index in indices:
            result += trace_conv2d(each_index, output_shape, kernel_size,
                stride, padding, dilation)
        
        return trace(
            layer_idx - 1,
            result,
            get_conv_input_shape(output_shape, kernel_size, stride, padding),
            model)
    
    else:
        return trace(layer_idx - 1, indices, output_shape, model)
