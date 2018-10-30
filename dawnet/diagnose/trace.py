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
import inspect
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from dawnet.data.image import get_rectangle_vertices
from dawnet.models.convs import get_conv_input_shape
from dawnet.utils.dependencies import get_pytorch_layers
from dawnet.diagnose.statistics import normalize_to_range



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


def run_partial_model(model, layer_idx, X):
    """Run the model partially

    @NOTE: this method might be refactored into the model class. It seems
    putting there might be better, but this diagnosis operation might not be
    popular enough to be put into that class. Moreover, this method assumes the
    model to have very basic architecture, as all layers are sequentially
    called, which might not be the case with more complex models.

    # Arguments
        model [torch.nn.Module]: the model
        layer_idx [int]: the final layer index to retrieve output
        X [torch.Tensor]: a valid input to the model

    # Returns
        [torch.Tensor]: the output of `layer_idx` when the `model` is fed w/ `X`
    """
    for idx, (_, layer) in enumerate(model.named_modules()):
        if type(layer) not in get_pytorch_layers():
            continue

        X = layer(X)

        if idx >= layer_idx:
            break

    return X


def get_most_activated_outputs(tensor, channel_dim=1, channel_idx=None):
    """Get the most activated outputs in a 4D tensor

    The most activated neurons should satisfy the following conditions:
        - must be larger than 0
        - must be at the 75th-quartile
        - must have normalized value be above of 50%
    
    @NOTE: the conditions above does not talk about absolute value.

    # Arguments
        tensor [4D torch Tensor]: the output channels of a conv layer
        channel_dim [int]: the index of channel dimension
        channel_idx [int]: the index of specific channel to find the most
            activated outputs. If None, then find the most activated of
            whole tensor

    # Returns
        [list of ints]: indices of most activated output in `tensor`
    """
    if isinstance(channel_idx, int):
        if channel_dim == 1:
            tensor = tensor[:,channel_idx,:,:]
        elif channel_dim == -1 or channel_dim == 3:
            tensor = tensor[:,:,:,channel_dim]
        else:
            raise AttributeError('invalid `channel_idx`, should be 1 or 3 but'
                                 'receive {}'.format(channel_idx))
    larger_0_mask = tensor > 0
                                                        # pylint: disable=E1101
    quantile_75_mask = tensor > torch.kthvalue(
        tensor.view(-1).cpu(),
        int(0.75 * len(tensor.view(-1))))[0].item()

    value_50_mask = tensor > (
        ((torch.max(tensor) - torch.min(tensor)) * 0.50
         + torch.min(tensor)).item())

    # [:,1:] to remove the batch channel
    return (larger_0_mask
            * quantile_75_mask
            * value_50_mask).nonzero().cpu().data.numpy()[:,1:]


def collect_image_patches_for_neuron_activation(layer_idx, indices, model, X):
    """Get the image patches corresponding to an image

    # Arguments
        layer_idx [int]: index of the layer that contains interested `indices`
        indices [list of tuple of 2 ints]: multiple point, each tuple is
            represented by y and x indices (0-counting)
        model [torch nn.Module]: the model in a layer representation.
        X [torch.Tensor]: a valid input to the model
    
    # Returns
        [tuple of 4 ints]: top, bottom, left, right
    """
    output_map = run_partial_model(model, layer_idx, X)
    result = trace(layer_idx, indices, tuple(output_map.shape[2:]), model)
    top, bottom, left, right = get_rectangle_vertices(result)
    
    return top, bottom, left, right


def collect_image_patches_for_feature_map(layer_idx, channel_idx, model, X):
    """Collect the image patches that ignite a feature map

    # Arguments
        layer_idx [int]: index of the layer that contains interested `indices`
        channel_idx [int]: the index of specific channel in the feature map to
            find the most activated outputs
        model [torch nn.Module]: the model in a layer representation.
        X [torch.Tensor]: a valid input to the model
    
    # Returns
        [list of nd arrays]: list of images
        [nd array]: the mask corresponding to the patches
    """

    # get the most activated locations in the channel in the layer
    output_map = run_partial_model(model, layer_idx, X)
    most_activated = get_most_activated_outputs(
        output_map, 1, channel_idx=channel_idx)

    # get the patches and construct the mask
    X_np = X.squeeze().cpu().data.numpy()
    mask = np.zeros(X_np.shape).astype(np.uint8)
    results = []
    for each_item in most_activated:
        indices = tuple(each_item)
        top, bottom, left, right = collect_image_patches_for_neuron_activation(
            13, indices, model, X)
        mask[top:bottom, left:right] = 1
        results.append(255-X_np[top:bottom,left:right])

    return results, mask


def view_3d_tensor(tensor, dim=0, max_columns=5, step=25, notebook=False):
    """Visualize 3D tensor by viewing groups of 2D images

    This function is used in conjunction with ipywidget. Suppose we have a
    3D tensor of CxHxW, this function will view C images of shape HxW, and
    paginated into smaller pages, each page contains `step` images.

    # Arguments
        tensor [3D np array or torch tensor]: the 3D tensor to view
        dim [int]: the dimension to view 2D image (default first dimension)
        max_columns [int]: number of images to show in each row
        step [int]: the number of images to show
        notebook [bool]: whether to view in jupyter notebook

    # Returns
        [ipywidget IntSlider]: the slider for previous, next page
    """

    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().data.numpy()

    tensor = tensor.squeeze()
    if len(tensor.shape) == 2:
        tensor = np.expand_dims(tensor, 0)
    
    if len(tensor.shape) != 3:
        raise AttributeError('the tensor should be 3D shape, get {}'
            .format(len(tensor.shape)))

    fig = plt.figure(figsize=(8,8))        

    def show_images(page=0):
        """Show the image in `tensor`

        # Arguments
            page [int]: the page number (will be controled by IntSlider)
        """
        fig.clf()

        image_list = list(tensor[page*step:(page+1)*step])
        columns = min(max_columns, len(image_list))
        rows = math.ceil(len(image_list) / columns)

        for _idx, each_img in enumerate(image_list):
            plot = fig.add_subplot(rows, columns, _idx+1)
            plot.imshow(each_img, cmap='gray')
        
        if not notebook:
            fig.show()
    
    return show_images


import cv2
def changing_input_view_feature_channel(image, model, layer_idx, channel_idx):
    """View the feature map as input changes

    @NOTE: currently this function supports:
    - blur
    - crop
    - italicize
    - noise
    - random background

    @NOTE:
    - multiple layer_idx and channel_idx

    # Argument
        image [2D nd array]: the image
        model [torch.nn.Dawnet]: the dawnet model
        layer_idx [int]: the layer to view.
        channel_idx [int]: the channel to view.
    """
                                                        # pylint: disable=E1101
    if next(model.parameters()).is_cuda:
        image_torch = torch.FloatTensor(image, device=torch.device('cuda:0'))
        image_torch = image_torch.unsqueeze(0).unsqueeze(0).cuda()
    else:
        image_torch = torch.FloatTensor(image)
        image_torch = image_torch.unsqueeze(0).unsqueeze(0).cuda()

    feature_map = run_partial_model(model, layer_idx, image_torch)
    feature_map = feature_map[:,channel_idx,:,:].squeeze().cpu().data.numpy()
    
    fig = plt.figure()
    image_plot = fig.add_subplot(121)
    channel_plot = fig.add_subplot(122)

    image_plot.imshow(image, cmap='gray')
    channel_plot.imshow(feature_map, cmap='gray')

    def show_images(blur_type, blur_value, blur_kernel_size):
        # enhance the image here
        
        image_new = cv2.medianBlur(image, blur_kernel_size)
        prediction = model.x_infer(image_new)

        print('Prediction: {}'.format(prediction))

        if next(model.parameters()).is_cuda:
            image_torch = torch.FloatTensor(
                image_new, device=torch.device('cuda:0'))
            image_torch = image_torch.unsqueeze(0).unsqueeze(0).cuda()
        else:
            image_torch = torch.FloatTensor(image)
            image_torch = image_torch.unsqueeze(0).unsqueeze(0).cuda()

        feature_map = run_partial_model(model, layer_idx, image_torch)
        feature_map = feature_map[:,channel_idx,:,:]
        feature_map = feature_map.squeeze().cpu().data.numpy()

        image_plot.imshow(image_new, cmap='gray')
        channel_plot.imshow(feature_map, cmap='gray')

        fig.canvas.draw()
    
    return show_images
