import torch.nn as nn


def construct_residual_block(in_channels, out_channels, stride,
    n_units, unit_type, name='res', **kwargs):
    """Construct a large residual block from smaller blocks

    @TODO: allow the out_channels to accept a list of numbers to allow
    incrementally increasing out_channels (pyramidal residual networks)

    # Arguments
        in_channels [int]: the number of incoming channels for first layer
        out_channels [int]: the number of channels from last layer
        stride [int]: the stride when downsampling
        n_units [int]: the number of blocks to stack
        unit_type [1 of Residual... above]: the type of block to use

    # Returns
        [nn.Module]: the resulting block
    """
    block = nn.Sequential()
    for each_block in range(n_units):
        if each_block > 0:
            in_channels = out_channels
            stride = 1
        block.add_module(
            "{}_{}".format(name, each_block),
            unit_type(in_channels, out_channels, stride=stride, **kwargs))

    return block

def construct_vgg_block(in_channels, out_channels, n_layers,
    batch_norm=True, dropout=False, pooling=True, name='conv'):
    """Construct a vgg block

    # Arguments
        in_channels [int]: the number of first layer's input channels
        out_channels [int]: the number of last layer's ouput channels
        n_layers [int]: the number of convolutional blocks
        batch_norm [bool]: whether to use batch norm
        dropout [bool]: whether to use dropout
        pool [bool]: whether to use pool layer to reduce output size

    # Returns
        [nn.Sequential]: the sequence of pytorch layers
    """
    block = nn.Sequential()
    for layer_idx in range(n_layers):
        if layer_idx > 0:
            in_channels = out_channels

        # add the conv layer
        block.add_module(
            '{}_conv_{}'.format(name, layer_idx),
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=1, padding=0))

        if batch_norm:      # add batch norm layer
            block.add_module(
                '{}_bn_{}'.format(name, layer_idx),
                nn.BatchNorm2d(out_channels))

        # TODO: set dropout option

        # add ReLU layer
        block.add_module('{}_relu_{}'.format(name, layer_idx), nn.ReLU())

    # add pooling layer
    if pooling:
        block.add_module(
            '{}_maxpool_{}'.format(name, layer_idx),
            nn.MaxPool2d(kernel_size=2, stride=2))

    return block


def construct_dense_block(in_channels, growth_rate, n_units,
                          dropout=None, name='dense_unit'):
    """Construct the dense block

    # Arguments
        in_channels [int]: the number of block's input channels
        growth_rate [int]: the number of output channels in each dense unit
        n_units [int]: the number of dense units in a block
        name [str]: the prefix-name of each dense unit

    # Returns
        [nn.Sequential]: the block
        [int]: the number of output channels
    """
    block = nn.Sequential()
    for each_block in range(n_units):
        if each_block > 0:
            in_channels += growth_rate

        block.add_module(
            '{}_{}'.format(name, each_block),
            DenseUnit(in_channels=in_channels, growth_rate=growth_rate,
                      dropout=dropout)
        )

    return block, in_channels + growth_rate

def construct_dense_transition_block(in_channels, compression,
                                     name='dense_transition'):
    """Construct the transition block in densenet

    # Arguments
        in_channels [int]: the number of input channels
        compression [int]: the compression value (be used to calculate the
            number of output channels)
        name [str]: the prefix name for operation

    # Returns
        [nn.Sequential]: the returned block
        [int]: the number of output channels
    """
    out_channels = int(in_channels * compression)
    block = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True)
    )
    block.add_module(
        '{}_1x1conv'.format(name),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=1, stride=1, padding=0, bias=False)
    )
    block.add_module(
        '{}_avgpool'.format(name),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    )

    return block, out_channels


def construct_pyramid_block(in_channels, alpha, depth, stride,
                            n_units, unit_type, multiplicative=False,
                            name='pyramidblock', **kwargs):
    """Construct the pyramid block

    The Pyramid block is described here: https://arxiv.org/abs/1610.02915.
    Basically it contains 2 schemes:
        - Multiplicative: math.floor(last_channel * alpha**(1/N))
        - Additive: math.floor(last_channel + alpha/N)

    # Arguments
        in_channels [int]: the number of blocks' input channels
        alpha [float]: the widening factor
        depth [int]: the total number of residual blocks in the network
        stride [int]: the stride to take when downsampling
        n_units [int]: the number of residual blocks in this pyramid block
        unit_type [nn.Module]: the block type
        multiplicative [bool]: widen using the multiplicative or additive
            scheme
        name [str]: the name for this block

    # Returns
        [nn.Module]: the pyramid block
        [int]: the number of output channels
    """
    def get_out_channels(_in_channels):
        """Get the number of output channels"""
        if multiplicative:
            return math.floor(_in_channels * alpha ** (1 / depth))
        return math.floor(_in_channels + alpha / depth)

    out_channels = get_out_channels(in_channels)
    block = nn.Sequential()
    for each_block in range(n_units):
        if each_block > 0:
            in_channels = out_channels
            stride = 1
        block.add_module(
            '{}_{}'.format(name, each_block),
            unit_type(in_channels, out_channels, stride=stride, **kwargs))
        out_channels = get_out_channels(in_channels)

    return block, out_channels

def weight_bias_init(layer):
    """Initialize the layers using common best-practices

    # Arguments
        layer [PyTorch layer]: the Pytorch layer
    """
    # TODO: understand the default initialization of Conv2d, BatchNorm2d,
    # Linear

    # vgg style
    # if isinstance(layer, nn.Conv2d):
    #     nn.init.kaiming_normal_(layer.weight.data, mode='fan_out')
    #     layer.bias.data.zero_()
    # elif isinstance(layer, nn.BatchNorm2d):
    #     layer.weight.data.uniform_()
    #     layer.bias.data.zero_()
    # elif isinstance(layer, nn.Linear):
    #     layer.bias.data.zero_()

    # resnet, resnet-preact style
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight.data, mode='fan_out')
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.fill_(1)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear):
        layer.bias.data.zero_()


def super_converge(self, max_lr, base_lr, stepsize, patience, omega,
    better_as_larger, optimizer=None, ensemble_folder=None):
    """Enable super-convergence learning

    # Arguments
        optimizer [torch optim]: the optimization object
        ensemble_folder [str]: the folder to save ensemble checkpoints
            If None, then it will be in the current directory
    """
    if self.super_converge_flag:
        warnings.warn('`super_converge_flag` is already True, skip...')

    self.super_converge_flag = True

    # folder
    ensemble_folder = '.' if ensemble_folder is None else ensemble_folder
    self.super_converge_ensemble_folder = ensemble_folder

    # optimizer
    optimizer = self.optimizer if optimizer is None else optimizer

    def save_model():
        self.x_save(ensemble_folder, other_name=self.training_iteration)

    self.lr_scheduler = SuperConvergence(
        optimizer=optimizer, max_lr=max_lr, base_lr=base_lr,
        stepsize=stepsize, patience=patience, omega=omega,
        better_as_larger=better_as_larger)

    self.lr_scheduler.add_save_model(save_model)
    print('Super-convergence set up.')
