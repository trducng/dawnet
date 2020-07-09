"""Repository of common convolutional architectures
@author: _john
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Perform normal convolution block"""

    def __init__(self, in_channels, out_channels, kernel_size=3, preact=False,
                 bottleneck=False, *args, **kwargs):
        if preact and bottleneck:
            inter_channels = out_channels // 4
            self.convs = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, inter_channels, 1,),
                nn.BatchNorm2d(inter_channels),
                nn.RelU(),
                nn.Conv2d(inter_channels, inter_channels, kernel_size, *args, **kwargs),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(),
                nn.Conv2d(inter_channels, out_channels, 1))
        elif preact and not bottleneck:
            self.convs = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size, *args, **kwargs))
        elif not preact and bottleneck:
            inter_channels = out_channels // 4
            self.convs = nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, 1),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(),
                nn.Conv2d(inter_channels, inter_channels, kernel_size, *args, **kwargs),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(),
                nn.Conv2d(inter_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, *args, **kwargs),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())

    def forward(self, input_tensor):
        return self.convs(input_tensor)


class DeConvBlock(nn.Module):
    """Perform normal deconvolution"""

    def __init__(self, in_channels, out_channels, kernel_size=2, *args, **kwargs):
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, *args, **kwargs),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU())

    def forward(self, input_tensor):
        return self.convs(input_tensor)


class ZeroPadLayer(nn.Module):
    """Zero padding for shortcut.

    This layer transform a [N x C_in x H_in x W_in] tensor into a
    [N X C_out x H_in/stride x W_in/stride] tensor. More specifically,
    this layer:
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

    #pylint: disable=W0221
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

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(in_channels, bottleneck)
        self.linear2 = nn.Linear(bottleneck, in_channels)

    #pylint: disable=W0221
    def forward(self, x):
        """Perform the forward pass"""
        batch_size = x.size(0)

        hidden = self.avg_pool(x).view(batch_size, -1).contiguous()
        hidden = F.relu(self.linear1(hidden))
        hidden = torch.sigmoid(self.linear2(hidden)).view(batch_size, -1, 1, 1)

        return torch.mul(hidden, x)


class DenseUnit(nn.Module):
    """Implement the DenseBlock, as described in
        https://arxiv.org/abs/1608.06993

    This class implements the bottleneck version of the convolutional layer in
    denseunit (referred in the paper as DenseNet-B)
    """

    def __init__(self, in_channels, growth_rate, dropout=None):
        """Initialize the block"""
        super(DenseUnit, self).__init__()

        self.dropout = None if dropout is None else nn.Dropout(dropout)
        bottleneck_channels = growth_rate * 4

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1,
                               stride=1, padding=0, bias=False)

        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, growth_rate, kernel_size=3,
                               stride=1, padding=1, bias=False)

    #pylint: disable=W0221
    def forward(self, x):
        """Perform the forward pass"""
        hidden = self.conv1(F.relu(self.bn1(x)))
        hidden = self.conv2(F.relu(self.bn2(hidden)))
        if self.dropout is not None:
            hidden = self.dropout(hidden)

                                                        # pylint: disable=E1101
        return torch.cat([x, hidden], dim=1)


class ResidualBasicUnit(nn.Module):
    """A basic residual block"""

    def __init__(self, in_channels, out_channels, stride, se_scale=None,
                 name='basic_res'):
        """Initialize the basic residual block

        # Arguments
            in_channels [int]: the number of input channels for the first layer
            out_channels [int]: the number of output channels for the last
                layer
            stride [int]: the stride of the first convolutional layer in block
            se_scale [int]: the reduction ratio used. If not set, skip SE

        """
        super(ResidualBasicUnit, self).__init__()

        # the bias in residual block is unnecessary as it is offset in
        # the next batchnorm
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
        if in_channels != out_channels or stride != 1:
            self.shortcut.add_module(
                '{}_shortcut_conv'.format(name),
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, padding=0, bias=False)
            )
            self.shortcut.add_module(
                '{}_shortcut_bn'.format(name),
                nn.BatchNorm2d(out_channels)
            )

    #pylint: disable=W0221
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

    def __init__(self, in_channels, out_channels, stride, bottleneck_factor=4, se_scale=None,
                 name='bottle_res'):
        """Initialize the residual block"""
        super(ResidualBottleneckUnit, self).__init__()

        bottleneck_channels = out_channels // bottleneck_factor

        # TODO: understand where to put batch norm (conv -> batch -> relu) or
        # (conv -> relu -> batch)
        # TODO: learn about other types of layer normalization

        # bottleneck block
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
                               kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.se_layer = (
            SELayer(in_channels=out_channels, scale=se_scale)
            if se_scale is not None
            else None)

        # residual shortcut
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut.add_module(
                '{}_shortcut_conv'.format(name),
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, padding=0, bias=False))
            self.shortcut.add_module(
                '{}_shortcut_bn'.format(name),
                nn.BatchNorm2d(out_channels))

    #pylint: disable=W0221
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

    # Arguments
        in_channels [int]: the number of input channels
        out_channels [int]: the number of output channels
        stride [int or tuple of ints]: the stride to do downsampling
        dropout [float]: the dropout value if used
        skip_first_relu [bool]: whether to skip the first relu operation (Pyra)
        last_bn [bool]: whether to add the final batch norm (Pyramidal)
        zero_pad [bool]: whether to use zero padding identity or affine
        se_scale [int]: if set, use Squeeze-Excitation mechanism
        name [str]: the name of the block
    """

    def __init__(self, in_channels, out_channels, stride, dropout=None,
                 skip_first_relu=False, last_bn=False, zero_pad=False,
                 se_scale=None, name='basic_preact'):
        """Initialize the object"""
        super(ResidualBasicPreactUnit, self).__init__()

        self._skip_first_relu = skip_first_relu
        self._last_bn = last_bn
        self.dropout = None if dropout is None else nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.se_layer = (
            SELayer(in_channels=out_channels, scale=se_scale)
            if se_scale is not None
            else None)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            if zero_pad:
                forward = ZeroPadLayer(
                    n_channels_pad=out_channels-in_channels,
                    stride=stride)
                self.shortcut.add_module(
                    '{}_shortcut_zeropad'.format(name), forward)
            else:
                forward = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                    stride=stride, padding=0, bias=False)
                self.shortcut.add_module(
                    '{}_shortcut_conv'.format(name), forward)

    #pylint: disable=W0221
    def forward(self, x):
        """Perform the forward pass"""
        residual = self.bn1(x)
        if not self._skip_first_relu:
            residual = F.relu(residual)
        residual = self.conv1(residual)

        if self.dropout is not None:
            residual = self.dropout(residual)

        residual = self.conv2(F.relu(self.bn2(residual)))
        if self._last_bn:
            residual = self.bn3(residual)

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
        - use dropout: https://arxiv.org/abs/1605.07146; only for the non-
            channel scaling convolutions
    """

    def __init__(self, in_channels, out_channels, stride, se_scale=None,
                 skip_first_relu=False, last_bn=False, zero_pad=False,
                 dropout=None, name='bottlenect_preact'):
        """Initialize the object"""
        super(ResidualBottleneckPreactUnit, self).__init__()

        bottleneck_channels = out_channels // (2 * out_channels // in_channels)
        self._skip_first_relu = skip_first_relu
        self._last_bn = last_bn
        self.dropout = None if dropout is None else nn.Dropout(dropout)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels,
                               kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.se_layer = (
            SELayer(in_channels=out_channels, scale=se_scale)
            if se_scale is not None
            else None)

        # construct the shortcut
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            if zero_pad:
                forward = ZeroPadLayer(
                    n_channels_pad=out_channels-in_channels,
                    stride=stride)
                self.shortcut.add_module(
                    '{}_shortcut_zeropad'.format(name),
                    forward)
            else:
                forward = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                    padding=0, stride=stride, bias=False)
                self.shortcut.add_module(
                    '{}_shortcut_conv'.format(name),
                    forward)

    #pylint: disable=W0221
    def forward(self, x):
        """Perform the forward pass"""
        residual = self.bn1(x)
        residual = (self.conv1(residual) if self._skip_first_relu else
                    self.conv1(F.relu(residual)))
        residual = F.relu(self.bn2(residual))
        if self.dropout is not None:
            residual = self.dropout(residual)
        residual = self.conv2(residual)
        residual = self.conv3(F.relu(self.bn3(residual)))
        residual = self.bn4(residual) if self._last_bn else residual

        if self.se_layer is not None:
            residual = self.se_layer(residual)

        return residual + self.shortcut(x)


class ResidualNextUnitDepr(nn.Module):
    """A block in used in ResNeXT model, as documented here:
        https://arxiv.org/abs/1611.05431

    Note that this implementation also incorporate pre-activation method by
    Kaiming He.

    @TODO: experiment the performance with the Conv2d group parameter set
    to cardinality. Actually, as also noted in the paper, the group
    convolution is an equavalent form of this unit. Might implement that
    for faster performance. @NOTE: this unit is deprecated in favor of group
    convolution
    """

    def __init__(self, in_channels, out_channels, stride, cardinality,
                 se_scale=None, bottleneck_channels=None, name='resnextdepr'):
        """Initialize the unit"""
        super(ResidualNextUnitDepr, self).__init__()

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

    def forward(self, input_x):
        """Perform the forward pass"""
        hiddens = []
        for each_branch in self.blocks:
            hiddens.append(each_branch(input_x))

                                                        # pylint: disable=E1101
        residual = torch.sum(hiddens, dim=1)
        if self.se_layer is not None:
            residual = self.se_layer(residual)

        return residual + self.shortcuts(input_x)


class ResidualNextUnit(nn.Module):
    """A block in used in ResNeXT model, as documented here:
        https://arxiv.org/abs/1611.05431

    Note that this implementation also incorporate pre-activation method by
    Kaiming He. This implementation exploits group convolutions and also
    assumes preactivation operations

    # Arguments
        in_channels [int]: the number of input channels
        out_channels [int]: the number of output channels
        stride [int]: first layer in the block convolution stride
        cardinality [int]: the cardinality (think of groups)
        bottleneck [int]: the number of bottleneck channels in each cardinal
        se_scale [int]: the reduction ratio used. If not set, skip SE
        name [str]: the name of the unit
    """

    def __init__(self, in_channels, out_channels, stride, cardinality,
                 bottleneck=4, skip_first_relu=False, last_bn=False,
                 zero_pad=False, dropout=None, se_scale=None, name='resnext'):
        super(ResidualNextUnit, self).__init__()
        bottleneck_channels = cardinality * bottleneck

        self.dropout = None if dropout is None else nn.Dropout(p=dropout)
        self._skip_first_relu = skip_first_relu
        self._last_bn = last_bn

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=bottleneck_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(
            in_channels=bottleneck_channels, out_channels=bottleneck_channels,
            kernel_size=3, stride=stride, padding=1, groups=cardinality,
            bias=False)
        self.bn3 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(
            in_channels=bottleneck_channels, out_channels=out_channels,
            kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.se_layer = (
            SELayer(in_channels=out_channels, scale=se_scale)
            if se_scale is not None
            else None)


        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            if zero_pad:
                forward = ZeroPadLayer(
                    n_channels_pad=out_channels-in_channels,
                    stride=stride)
                self.shortcut.add_module(
                    '{}_shortcut_zeropad'.format(name), forward)
            else:
                forward = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                    padding=0, stride=stride, bias=False)
                self.shortcut.add_module(
                    '{}_shortcut_conv'.format(name), forward)

    #pylint: disable=W0221
    def forward(self, input_x):
        """Perform the forward pass"""
        residual = self.bn1(input_x)
        if not self._skip_first_relu:
            residual = F.relu(residual, inplace=True)
        residual = self.conv1(residual)
        residual = self.conv2(F.relu(self.bn2(residual)))
        residual = self.conv3(F.relu(self.bn3(residual)))
        if self._last_bn:
            residual = self.bn4(residual)

        if self.se_layer is not None:
            residual = self.se_layer(residual)

        if self.dropout is not None:
            residual = self.dropout(residual)

        output = residual + self.shortcut(input_x)

        return output


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
    # vectorized version
    if isinstance(input_shape, torch.Tensor):
        return (input_shape - kernel_size + 2 * padding) // stride + 1

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
            (each_value - kernel_size[idx] + 2 * padding[idx]) // stride[idx]
            + 1)

    return result


def get_conv_input_shape(output_shape, kernel_size, stride, padding):
    """Get the convolutional input shape.

    Precaution: this method might not be absolutely correct, since multiple
    input shapes can result in the same output shape.

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
            (each_value - 1) * stride[idx]
            - 2 * padding[idx]
            + kernel_size[idx])

    return result
