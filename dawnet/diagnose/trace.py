# Trace the behavior of a convnet model.
# Zeiler and Fergus provided some great explanations and inspiration
# for the idea in this code: https://arxiv.org/abs/1311.2901
# Currently we should take notice of these issues:
#   - unfilter: the paper use inverse of conv weights, while we just want to
#       see the position of input neurons that are responsible for the output
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
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from ipywidgets import interact_manual, widgets, Layout

from dawnet.data.image import get_rectangle_vertices, get_subplot_rows_cols
from dawnet.models.convs import get_conv_input_shape
from dawnet.utils.dependencies import get_pytorch_layers


def trace_maxpool2d(
    indices, output_map, kernel_size, stride=None, padding=0, dilation=1
):
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
    elif isinstance(indices, tuple) or isinstance(indices, list) and len(indices) == 2:
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

    raise AttributeError(
        "`indices` should be output feature map or a " "tuple of 2 ints (y, x)"
    )


def trace_conv2d(
    indices, output_shape, kernel_size, stride=None, padding=0, dilation=1
):
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
        raise AttributeError("2D convolution must have 2D shape")

        # pylint: disable=E0632
    input_height, input_width = get_conv_input_shape(
        output_shape, kernel_size, stride, padding
    )  # input feature map size

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
            result += trace_conv2d(
                each_index, output_shape, kernel_size, stride, padding, dilation
            )

        return trace(
            layer_idx - 1,
            result,
            get_conv_input_shape(output_shape, kernel_size, stride, padding),
            model,
        )

    else:
        return trace(layer_idx - 1, indices, output_shape, model)


def run_partial_model(model, layer_idx, input_x, preprocess=None):
    """Run the model partially

    @NOTE: this method might be refactored into the model class. It seems
    putting there might be better, but this diagnosis operation might not be
    popular enough to be put into that class. Moreover, this method assumes the
    model to have very basic architecture, as all layers are sequentially
    called, which might not be the case with more complex models.

    # Arguments
        model [torch.nn.Module]: the model
        layer_idx [int]: the final layer index to retrieve output
        input_x [torch.Tensor]: a valid input to the model
        preprocess [function]: preprocessing function apply for `input_x`

    # Returns
        [torch.Tensor]: the output of `layer_idx` when the `model` is fed w/ `X`
    """
    if preprocess is not None:
        input_x = preprocess(input_x)

    idx = 0
    for _, layer in model.named_modules():
        if not isinstance(layer, tuple(get_pytorch_layers())):
            # @TODO: smart, but the order is not guaranteed
            continue

        if isinstance(layer, nn.LSTM):
            input_x = input_x.view(
                input_x.size(0), input_x.size(1) * input_x.size(2), input_x.size(3)
            )
            input_x = input_x.transpose(1, 2)
            input_x = input_x.transpose(0, 1).contiguous()
            input_x, _ = layer(input_x)
        else:
            input_x = layer(input_x)

        if idx >= layer_idx:
            break

        idx += 1

    return input_x


def predict_partial_model(model, layer_idx, input_x):
    """Get output logit running partially

    # Arguments
        model [torch.nn.Module]: the model
        layer_idx [int]: the starting index
        input_x [torch.Tensor]: a valid input to the layer

    # Returns
        [torch.Tensor]: output logits
    """
    idx = 0
    for _, layer in model.named_modules():
        if not isinstance(layer, tuple(get_pytorch_layers())):
            continue

        if idx >= layer_idx:
            if isinstance(layer, nn.LSTM):
                input_x = input_x.view(
                    input_x.size(0), input_x.size(1) * input_x.size(2), input_x.size(3)
                )
                input_x = input_x.transpose(1, 2)
                input_x = input_x.transpose(0, 1).contiguous()
                input_x, _ = layer(input_x)
            else:
                input_x = layer(input_x)

        idx += 1

    return input_x


def get_feature_maps(model, input_x):
    """Get model feature maps

    # Arguments
        model [torch.nn.Module]: the model
        input_x [torch.Tensor]: a valid input to the model

    # Returns
        [list of tuples of 3]: the feature maps (idx, layer class, np output)
    """
    result = []
    for idx, (_, layer) in enumerate(model.named_modules()):
        if not isinstance(layer, tuple(get_pytorch_layers())):
            # skip for non-Pytorch class
            continue

        if isinstance(layer, tuple(get_pytorch_layers())) and not isinstance(
            layer, tuple(get_pytorch_layers(conv=True))
        ):
            # stop when stepping into classifiers
            break

        input_x = layer(input_x)
        result.append((idx, type(layer), input_x.cpu().data.numpy()))

    return result


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
            tensor = tensor[:, channel_idx, :, :]
        elif channel_dim == -1 or channel_dim == 3:
            tensor = tensor[:, :, :, channel_dim]
        else:
            raise AttributeError(
                "invalid `channel_idx`, should be 1 or 3 but"
                "receive {}".format(channel_idx)
            )
    larger_0_mask = tensor > 0
    # pylint: disable=E1101
    quantile_75_mask = (
        tensor
        > torch.kthvalue(tensor.view(-1).cpu(), int(0.75 * len(tensor.view(-1))))[
            0
        ].item()
    )

    value_50_mask = tensor > (
        ((torch.max(tensor) - torch.min(tensor)) * 0.50 + torch.min(tensor)).item()
    )

    # [:,1:] to remove the batch channel
    return (
        (larger_0_mask * quantile_75_mask * value_50_mask)
        .nonzero()
        .cpu()
        .data.numpy()[:, 1:]
    )


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
    most_activated = get_most_activated_outputs(output_map, 1, channel_idx=channel_idx)

    # get the patches and construct the mask
    X_np = X.squeeze().cpu().data.numpy()
    mask = np.zeros(X_np.shape).astype(np.uint8)
    results = []
    for each_item in most_activated:
        indices = tuple(each_item)
        top, bottom, left, right = collect_image_patches_for_neuron_activation(
            13, indices, model, X
        )
        mask[top:bottom, left:right] = 1
        results.append(255 - X_np[top:bottom, left:right])

    return results, mask


def view_3d_tensor(tensor, max_rows=None, max_columns=None, construct_widget=True):
    """Visualize 3D tensor by viewing groups of 2D images

    This function is used in conjunction with ipywidget. Suppose we have a
    3D tensor of CxHxW, this function will view C images of shape HxW, and
    paginated into smaller pages, each page contains `step` images.

    # Arguments
        tensor [3D np array or torch tensor]: the 3D tensor to view
        max_rows [int]: number of rows of images
        max_columns [int]: number of images to show in each row
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
        raise AttributeError(
            "the tensor should be 3D shape, get {}".format(len(tensor.shape))
        )

    max_rows, max_columns = get_subplot_rows_cols(tensor)
    step = max_rows * max_columns

    if construct_widget:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=(8, 8))

    def show_images(page=0):
        """Show the image in `tensor`

        # Arguments
            page [int]: the page number (will be controled by IntSlider)
        """
        fig.clf()

        image_list = list(tensor[page * step : (page + 1) * step])
        columns = min(max_columns, len(image_list))
        rows = math.ceil(len(image_list) / columns)

        for _idx, each_img in enumerate(image_list):
            plot = fig.add_subplot(rows, columns, _idx + 1)
            plot.set_title("{}".format(_idx + page * step))
            plot.axis("off")
            plot.imshow(each_img, cmap="gray")

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01)
        fig.show()

    if not construct_widget:
        return show_images

    page_slider = widgets.IntSlider(
        min=0,
        max=math.ceil(tensor.shape[0] / step - 1),
        step=1,
        value=0,
        description="Page:",
        layout=Layout(width="75%"),
    )
    interact_manual(show_images, page=page_slider)

    return show_images


def view_feature_maps(model, X):
    """View model feature maps

    # Arguments
        model [torch nn.Module]: the model
        X [torch Tensor]: a valid input to the model

    # Returns
        [func]: image showing function for ipywidget interactivity
    """
    feature_maps = get_feature_maps(model, X)

    fig = plt.figure()

    def show_images(layer=0, page=0):
        """Show the image in `tensor`

        # Arguments
            page [int]: the page number (will be controled by IntSlider)
        """
        fig.clf()

        tensor = feature_maps[layer][2][0, :]
        max_rows, max_columns = get_subplot_rows_cols(tensor)
        step = max_rows * max_columns

        image_list = list(tensor[page * step : (page + 1) * step])
        columns = min(max_columns, len(image_list))
        rows = math.ceil(len(image_list) / columns)

        fig.suptitle(
            "Layer {} - Type {}".format(
                feature_maps[layer][0], feature_maps[layer][1].__name__
            ),
            fontsize=16,
        )
        for _idx, each_img in enumerate(image_list):
            plot = fig.add_subplot(rows, columns, _idx + 1)
            plot.set_title("{}".format(_idx + page * step))
            plot.axis("off")
            plot.imshow(each_img, cmap="gray")

        plt.subplots_adjust(
            left=0, right=1, top=0.9, bottom=0, wspace=0.01, hspace=0.01
        )
        fig.show()

    def update_num_pages(*args):
        """Change the number of pages as the layer changes"""
        layer = layer_slider.value
        tensor = feature_maps[layer][2][0, :]
        page_slider.max = math.ceil(
            tensor.shape[0] // np.prod(get_subplot_rows_cols(tensor)) - 1
        )

    tensor = feature_maps[0][2][0, :]
    layer_slider = widgets.IntSlider(
        min=0,
        max=len(feature_maps) - 1,
        step=1,
        value=0,
        description="Layer:",
        layout=Layout(width="75%"),
    )
    page_slider = widgets.IntSlider(
        min=0,
        max=math.ceil(tensor.shape[0] / np.prod(get_subplot_rows_cols(tensor)) - 1),
        step=1,
        value=0,
        description="Page:",
        layout=Layout(width="75%"),
    )
    layer_slider.observe(update_num_pages, "value")
    interact_manual(show_images, layer=layer_slider, page=page_slider)

    return show_images
