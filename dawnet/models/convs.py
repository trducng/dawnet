# Repository of common convolutional architectures
# @author: John
# =============================================================================
import pdb
from abc import ABC, abstractclassmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroPadLayer(nn.Module):
    """Zero padding for shortcut.

    This layer transform a [N x C_in x H_in x W_in] tensor into a
    [N X C_out x H_out x W_out] tenosr. More specifically, this layer:
        (1) pads missing channels with channels of 0 to get C_out
        (2) performs adaptive maxpooling to get the desired output shape
            to get H_out and W_out

    # Arguments
        n_channels_pad [int]: the amount of additional channels to 0 pad
        stride [ints]: the supposed output shape of each
            channel. If `channel_shape` is not None, then the input will go
            through an adaptive max pooling to achieve the desired output shape
            then 0-channel padding, otherwise, skip the pooling phase
    """

    def __init__(self, n_channels_pad, stride=None):
        """Initialize the object"""
        super(ZeroPadLayer, self).__init__()
        self._n_channels_pad = n_channels_pad
        self.adapt_channel_shape = nn.Sequential()
        if stride is not None:
            self.adapt_channel_shape.add_module(
                'max_pool',
                nn.MaxPool2d(kernel_size=stride, stride=stride))

    def forward(self, x):
        """Perform the forward pass

        # Arguments
            x [torch Tensor]: the input feature map
            n_channels [int]: the amount of channels to add
        """
        x = self.adapt_channel_shape(x)
        return F.pad(x, (0, 0, 0, 0, 0, self._n_channels_pad), 'constant', 0)


class SELayer(nn.Module):
    """Implementing a Squeeze and Excitation unit block"""

    def __init__(self, in_channels, scale):
        """Initialize the object"""
        super(SELayer, self).__init__()
        bottleneck = in_channels // scale

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear1 = nn.Linear(in_channels, bottleneck)
        self.linear2 = nn.Linear(bottleneck, in_channels)

    def forward(self, x):
        """Perform the forward pass"""
        batch_size = x.size(0)

        hidden = self.avg_pool(x).view(batch_size, -1).contiguous()
        hidden = F.relu(self.linear1(hidden))
        hidden = F.sigmoid(self.linear2(hidden)).view(batch_size, -1, 1, 1)

        return torch.mul(hidden, x)


class DenseUnit(nn.Module):
    """Implement the DenseBlock, as described in
        https://arxiv.org/abs/1608.06993
    
    This class implements the bottleneck version of the convolutional layer in
    denseunit (referred in the paper as DenseNet-B)
    """

    def __init__(self, in_channels, growth_rate, dropout=0):
        """Initialize the block"""
        super(DenseUnit, self).__init__()

        self.dropout = dropout
        bottleneck_channels = growth_rate * 4

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1,
                               stride=1, padding=0)

        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, growth_rate, kernel_size=3,
                               stride=1, padding=1)

    def forward(self, x):
        """Perform the forward pass"""
        hidden = self.conv1(F.relu(self.bn1(x)))
        if self.dropout > 0:
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = self.conv2(F.relu(self.bn2(hidden)))
        if self.dropout > 0:
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)

                                                        # pylint: disable=E1101
        return torch.cat([x, hidden], dim=1)


class ResidualBasicUnit(nn.Module):
    """A basic residual block"""

    def __init__(self, in_channels, out_channels, stride, se_scale=None,
        name='basic_res'):
        """Initialize the basic residual block

        # Arguments
            in_channels [int]: the number of input channels for the first layer
            out_channels [int]: the number of output channels for the last layer
            stride [int]: the stride of the first convolutional layer in block
        """
        super(ResidualBasicUnit, self).__init__()

        # TODO: confirm the bias term in a residual block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # construct SE layer
        self.se_layer = (
            SELayer(in_channels=out_channels, scale=se_scale)
            if se_scale is not None
            else None)

        # construct the shortcut
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                '{}_shortcut_conv'.format(name),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2,
                          padding=0, bias=False)
            )
            self.shortcut.add_module(
                '{}_shortcut_bn'.format(name),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """Perform the forward pass"""
        hidden = F.relu(self.bn1(self.conv1(x)))
        hidden = self.bn2(self.conv2(hidden))

        if self.se_layer is not None:
            hidden = self.se_layer(hidden)

        hidden += self.shortcut(x)

        return F.relu(hidden)


class ResidualBottleneckUnit(nn.Module):
    """Residual block using bottleneck architecture"""

    def __init__(self, in_channels, out_channels, stride,
        se_scale=None, name='bottle_res'):
        """Initialize the residual block"""
        super(ResidualBottleneckUnit, self).__init__()
        self.expansion = 4

        bottleneck_channels = out_channels // self.expansion

        # TODO: understand where to put batch norm (conv -> batch -> relu) or
        # (conv -> relu -> batch)
        # TODO: learn about other types of layer normalization

        # bottleneck block
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
                               kernel_size=3, stride=stride, padding=0,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.se_layer = (
            SELayer(in_channels=out_channels, scale=se_scale)
            if se_scale is not None
            else None)

        # residual shortcut
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                '{}_shortcut_conv'.format(name),
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, padding=0, bias=False))
            self.shortcut.add_module(
                '{}_shortcut_bn'.format(name),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        """Perform the forward pass"""
        hidden = F.relu(self.bn1(self.conv1(x)))
        hidden = F.relu(self.bn2(self.conv2(hidden)))
        hidden = self.bn3(self.conv3(hidden))

        if self.se_layer is not None:
            hidden = self.se_layer(hidden)

        hidden += self.shortcut(x)

        return F.relu(hidden)


class ResidualBasicPreactUnit(nn.Module):
    """A pre-activation basic block as documented here
        https://arxiv.org/abs/1603.05027 with the option for dropout as
        documented here: https://arxiv.org/abs/1605.07146

    The reason for incorporate wide residual unit into this unit is because the
    wide unit has exactly the same architecture as this unit. The only
    difference is wide unit will use a larger amount of channels.

    @TODO: the preactivation implementation also includes 3 undocumented
    flow, such as: remove the first relu, add one more batch norm, add
    Relu(batchnorm) to the identity. The performance of these 3 behaviors
    should be examined
    """

    def __init__(self, in_channels, out_channels, stride, dropout=0,
        se_scale=None, name='basic_preact'):
        """Initialize the object"""
        super(ResidualBasicPreactUnit, self).__init__()

        self.dropout = dropout
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=0, bias=False)
        self.se_layer = (
            SELayer(in_channels=out_channels, scale=se_scale)
            if se_scale is not None
            else None)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                '{}_shortcut_conv'.format(name),
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          stride=stride, padding=1, bias=False))

    def forward(self, x):
        """Perform the forward pass"""
        residual = self.conv1(F.relu(self.bn1(x)))

        if self.dropout > 0:
            residual = F.dropout(residual, p=self.dropout,
                                 training=self.training, inplace=False)

        residual = self.conv2(F.relu(self.bn2(residual)))
        if self.se_layer is not None:
            residual = self.se_layer(residual)

        return residual + self.shortcut(x)


class ResidualBottleneckPreactUnit(nn.Module):
    """A pre-activation basic block as documented here
        https://arxiv.org/abs/1603.05027

    The implementation also includes several options:
        - squeeze and excitation: https://arxiv.org/abs/1709.01507; to get the
            desired behavior, set `se_scale` == 4
        - skip the first relu & incorporate last batch & zero padding:
            https://arxiv.org/abs/1610.02915, to get the desired behavior, set
            `skip_first_relu=True, last_bn=True, zero_pad=True`
    """

    def __init__(self, in_channels, out_channels, stride, se_scale=None,
        skip_first_relu=False, last_bn=False, zero_pad=False,
        name='bottlenect_preact'):
        """Initialize the object"""
        super(ResidualBottleneckPreactUnit, self).__init__()

        bottleneck_channels = out_channels // (2 * out_channels // in_channels)
        self._skip_first_relu = skip_first_relu
        self._last_bn = last_bn

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
                               kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.se_layer = (
            SELayer(in_channels=out_channels, scale=se_scale)
            if se_scale is not None
            else None)

        # construct the shortcut
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            if zero_pad:
                forward = ZeroPadLayer(
                    n_channels_pad=out_channels-in_channels,
                    stride=stride)
            else:
                forward = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                    padding=0, stride=stride, bias=False)
            self.shortcut.add_module(
                '{}_shortcut_conv'.format(name),
                forward)

    def forward(self, x):
        """Perform the forward pass"""
        residual = self.bn1(x)
        residual = (self.conv1(residual) if self._skip_first_relu else
                    self.conv1(F.relu(residual)))
        residual = self.conv2(F.relu(self.bn2(residual)))
        residual = self.conv3(F.relu(self.bn3(residual)))
        residual = self.bn4(residual) if self._last_bn else residual

        if self.se_layer is not None:
            residual = self.se_layer(residual)

        return residual + self.shortcut(x)


class ResidualNextUnit(nn.Module):
    """A block in used in ResNeXT model, as documented here:
        https://arxiv.org/abs/1611.05431

    Note that this implementation also incorporate pre-activation method by
    Kaiming He.

    @TODO: experiment the performance with the Conv2d group parameter set
    to cardinality. Actually, as also noted in the paper, the group
    convolution is an equavalent form of this unit. Might implement that
    for faster performance.
    """

    def __init__(self, in_channels, out_channels, stride, cardinality,
        se_scale=None, bottleneck_channels=None, name='resnext'):
        """Initialize the unit"""
        super(ResidualNextUnit, self).__init__()

        if bottleneck_channels is None:
            bottleneck_channels = in_channels // (2 * cardinality)

        self.blocks = []
        for _ in range(cardinality):
            self.blocks.append(nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1,
                          stride=1, padding=0, bias=False),

                nn.BatchNorm2d(bottleneck_channels),
                nn.ReLU(),
                nn.Conv2d(bottleneck_channels, bottleneck_channels,
                          kernel_size=3, stride=stride, padding=1, bias=False),

                nn.BatchNorm2d(bottleneck_channels),
                nn.ReLU(),
                nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1,
                          stride=1, padding=0, bias=False)
            ))

        self.se_layer = (
            SELayer(in_channels=out_channels, scale=se_scale)
            if se_scale is not None
            else None)

        # construct the shortcut
        self.shortcuts = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                '{}_shortcut_conv'.format(name),
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          stride=2, padding=1, bias=0))

    def forward(self, x):
        """Perform the forward pass"""
        hiddens = []
        for each_branch in self.blocks:
            hiddens.append(each_branch(x))

                                                        # pylint: disable=E1101
        residual = torch.sum(hiddens, dim=1)
        if self.se_layer is not None:
            residual = self.se_layer(residual)

        return residual + self.shortcuts(x)


def get_conv_output_shape(input_shape, kernel_size, stride, padding):
    """Get the convolutional output shape

    # Arguments
        input_shape [int or tuple of ints]: the array of shapes
        kernel_size [int or tuple of ints]: the kernel size
        stride [int or tuple of ints]: the stride
        padding [int or tuple of ints]: the padding value
    
    # Returns
        [int or tuple of ints]: the output shape
    """
    if isinstance(input_shape, int):
        input_shape = [input_shape]
    
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * len(input_shape)
    
    if isinstance(stride, int):
        stride = [stride] * len(input_shape)
    
    if isinstance(padding, int):
        padding = [padding] * len(input_shape)
    
    result = []
    for idx, each_value in enumerate(input_shape):
        result.append(
            (each_value - kernel_size[idx] + 2 * padding[idx]) / stride[idx]
            + 1)
    
    return result


def get_conv_input_shape(output_shape, kernel_size, stride, padding):
    """Get the convolutional input shape

    # Arguments
        output_shape [int or tuple of ints]: the array of shapes
        kernel_size [int or tuple of ints]: the kernel size
        stride [int or tuple of ints]: the stride
        padding [int or tuple of ints]: the padding value
    
    # Returns
        [int or tuple of ints]: the input shape
    """
    if isinstance(output_shape, int):
        output_shape = [output_shape]
    
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * len(output_shape)
    
    if isinstance(stride, int):
        stride = [stride] * len(output_shape)
    
    if isinstance(padding, int):
        padding = [padding] * len(output_shape)
    
    result = []
    for idx, each_value in enumerate(output_shape):
        result.append(
            (each_value -1) * stride[idx]
            - 2 * padding[idx]
            + kernel_size[idx])
    
    return result
    
