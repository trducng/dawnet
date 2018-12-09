# Basic interface for a model
# @author: John
# =============================================================================
import os
import time
import warnings

import torch
import torch.nn as nn

from dawnet.models.convs import DenseUnit
from dawnet.training.hyper import SuperConvergence
from dawnet.utils.dependencies import get_pytorch_layers


class _BaseModel(nn.Module):
    """Base class provides perception interface

    Think of this class as human's perception and tuition. The perception is
    provided with some external stimuli, then automatically and unconsiously
    links those stimuli to some pattern.

    Normal flow when training from scratch
        x_initialize ---> x_learn / x_train ---> x_save

    Normal flow when resume training
        x_initialize ---> x_load ---> x_learn / x_train ---> x_save

    Normal flow when evaluation
        x_initialize ---> x_load ---> x_infer
    """

    def __init__(self, name=None):
        """Initialize the object"""
        super(_BaseModel, self).__init__()
        self._x_forward = self.__call__

        # get the name
        self.name = (
            str(int(time.time()))
            if not isinstance(name, str)
            else name)
        self.name = '{}_{}'.format(self.__class__.__name__, self.name)

    def switch_forward_function(self, forward_fn=None):
        """Switch the forward function

        # Arguments
            forward_fn [Function]: the forward function
        """
        if forward_fn is None:
            self._x_forward = forward_fn
        else:
            self._x_forward = forward_fn

    def get_layer_indices(self, layer_type):
        """Get the indices of layers that has the type `layer_type`

        # Arguments
            layer_type [torch.nn.Module]: the type of layer to compare on

        # Returns
            [list of ints]: list of indices that has layer match `layer_type`
        """
        indices = []
        for idx, (_, layer) in enumerate(self.named_modules()):
            if isinstance(layer, layer_type):
                indices.append(idx)

        return indices

    def get_layer(self, layer_idx):
        """Get the layer that has `layer_idx` in `named_modules()`

        # Arguments
            layer_idx [int]: the final layer index to retrieve output

        # Returns
            [str]: the layer name
            [torch.nn.Module]: the specific layer
        """
        return list(self.named_modules())[layer_idx]

    def get_number_layers(self):
        """Get number of layers

        # Returns
            [int]: get number of layers
        """
        count = 0
        for _, layer in self.named_modules():
            if type(layer) in get_pytorch_layers():
                count += 1

        return count

    def get_number_parameters(self, verbose=False):
        """Get the number of parameters

        # Arguments
            verbose [bool]: whether to print in human easily readable format
        """
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return params

    # _x_ interfaces
    def x_load(self, *args, **kwargs):
        """Load the agent's saved state.

        This loading should be constructed, so that the using of agent is
        self-contained for inference, .i.e. only the saved_path is necessary
        in order for the agent to infer and continue training.
        """
        raise NotImplementedError('`x_load` should be subclassed')

    def x_initialize(self, *args, **kwargs):
        """Initialize the agent's structure"""
        raise NotImplementedError('`x_initialize` should be subclassed')

    def x_learn(self, *args, **kwargs):
        """Learn from a minibatch of data"""
        raise NotImplementedError('`x_learn` should be subclassed')

    def x_train(self, *args, **kwargs):
        """Train from a dataset"""
        raise NotImplementedError('`x_train` should be subclassed')

    def x_infer(self, *args, **kwargs):
        """Infer the output from the input data"""
        raise NotImplementedError('`x_infer` should be subclassed')

    def x_save(self, *args, **kwargs):
        """Save the agent's state"""
        raise NotImplementedError('`x_save` should be subclassed')

    def x_test(self, *args, **kwargs):
        """Perform testing"""
        raise NotImplementedError('x_test` should be subclassed')

    def x_validate(self, *args, **kwargs):
        """Performance testing"""
        raise NotImplementedError('`x_validate` should be subclassed')


class BaseModel(_BaseModel):
    """The base architecture using convolutional layers"""

    def __init__(self, name=None):
        """Initialize the object"""
        super(BaseModel, self).__init__(name=name)

        # initialize the convolutional group: conv, pooling, batchnorm,
        # dropout layers are considered valid
        self.convs = nn.Sequential()

        # training optimizer
        self.optimizer = None

        # super-convergence flag, activate with `self.super_converge()`
        self.super_converge_flag = False
        self.super_converge_ensemble_folder = None
        self.lr_scheduler = None

        # progress history
        self.history = []

        # denote current training iteration
        self.training_iteration = 0

    def _forward_conv(self, x):
        """Make the forward pass for the convolution group"""
        return self.convs(x)

    def _get_progress(self):
        """Get the training progress"""
        progress = self.get_progress()
        progress['itr'] = self.training_iteration

        if self.history and progress['itr'] == self.history[-1]['itr']:
            return

        self.history.append(progress)

    def _get_save_state(self):
        """Get the save state and collect other minor information"""
        self._get_progress()

        state = self.get_save_state()

        state['name'] = self.name
        state['model'] = self.__class__.__name__
        state['history'] = self.history
        state['state_dict'] = self.state_dict()
        state['super_converge_flag'] = self.super_converge_flag
        state['training_iteration'] = self.training_iteration

        if self.super_converge_flag:
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['super_converge_ensemble_folder'] = (
                self.super_converge_ensemble_folder)

        return state

    # Conv architectures
    def construct_vgg_block(self, in_channels, out_channels, n_layers,
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

    def construct_large_residual_block(self, in_channels, out_channels, stride,
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

    def construct_dense_block(self, in_channels, growth_rate, n_units,
                              name='dense_unit'):
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
                DenseUnit(in_channels=in_channels, growth_rate=growth_rate)
            )

        return block, in_channels + growth_rate

    def construct_dense_transition_block(self, in_channels, compression,
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
        block = nn.Sequential()
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

    # Training helper
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

    # Weights initialization scheme
    def weight_bias_init(self, layer):
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

    def restart(self):
        """Restart the model"""
        # @TODO: some weights and biases are untouched
        self.apply(self.weight_bias_init)

    def x_load(self, path):
        """Load the saved model

        # Arguments
            path [str]: the path to saved model
        """
        print('Loading from {}...'.format(path))
        state = torch.load(path)

        # call user-defined updates
        self.load_dict(state)

        if state['model'] != self.__class__.__name__:
            print(
                ':WARNING: incompatible model, this is {} but load {}'
                .format(self.__class__.__name__, state['model']))

        self.name = state['name']
        self.history = state['history']
        self.load_state_dict(state['state_dict'])
        self.training_iteration = state['training_iteration']
        self.super_converge_flag = state['super_converge_flag']

        if self.super_converge_flag:
            self.super_converge_flag = False

            self.super_converge_ensemble_folder = (
                state['super_converge_ensemble_folder'])
            self.super_converge(
                max_lr=10, base_lr=4, stepsize=10,
                patience=10, omega=0.1, better_as_larger=True,  # dummy var
                ensemble_folder=state['super_converge_ensemble_folder'])
            self.lr_scheduler.load_state_dict(state['lr_scheduler'])

            self.super_converge_flag = True

    def x_save(self, outpath, other_name=None):
        """Save the agent's state"""
        state = self._get_save_state()

        filepath = (
            os.path.join(outpath, '{}.john'.format(self.name))
            if other_name is None
            else os.path.join(
                outpath, '{}_{}.john'.format(self.name, other_name))
        )

        try:
            torch.save(state, filepath)
        except KeyboardInterrupt as error:
            to_kill = input(
                'Currently saving state, killing now might '
                'destroy all results. Kill? [y/N]: ')
            if to_kill.lower() != 'y':
                torch.save(state,
                           os.path.join(outpath, '{}.john'.format(self.name)))
            else:
                raise error

    # must subclasses
    def get_save_state(self):
        """Get the dictionary state to save to file"""
        raise NotImplementedError('`get_save_state` should be subclassed')

    def get_progress(self):
        """Get the history"""
        raise NotImplementedError('`collect_history` should be subclassed')

    def load_dict(self, state):
        """Load user-defined variables from a Pytorch state_dict"""
        raise NotImplementedError('`load_dict` should be implemented')


class DataParallel(nn.DataParallel):
    """
    Subclass the data parallel to allow agent operation in parallel GPU
    settings

    The behavior of this class is to:
        - Pass the _x_ calls from `nn.DataParallel` into the wrapped object
        - Reset the wrapped object's forward function with `nn.DataParallel`'s
            forward function (which handles data parallelism)
    """

    def __init__(self, *args):
        """Initialize the data parallelism object"""
        super(DataParallel, self).__init__(*args)

        # let the driver know that it is wrapped by DataParallel
        self.module.switch_forward_function(self.__call__)

        # this attribute should be placed last in `__init__` as it sinifies
        # that instance is basically fully initiated
        self._fully_initialized = True

    def __getattr__(self, name):
        """Get the object and `module`'s attributes

        The `__getattr__` method will be called if the attribute does not
        exist, so that another fallback way can be used to access the
        attribute.

        # Arguments
            name [str]: the name an attribute to be deleted
        """
        if '_fully_initialized' not in self.__dict__:
            return super(DataParallel, self).__getattr__(name)

        try:
            # check for _parameters, _buffers, _modules in nn.Module first
            return super(DataParallel, self).__getattr__(name)
        except AttributeError:
            # now we sure that this instance does not have 'name' attribute
            return getattr(self.module, name)

    def __setattr__(self, name, value):
        """Set object attribute or module's attribute

        # Arguments
            name [str]: the name of the attribute to be set
            value [object]: the corresponding value of the `name` attribute
        """
        if '_fully_initialized' not in self.__dict__:
            # the object is not fully initialized, so any set attribute
            # attempt is to set to the object, not the `module`
            return super(DataParallel, self).__setattr__(name, value)

        if name in self.__dir__():
            # if the instance contains `name`, then this will modify that
            # attribute
            return super(DataParallel, self).__setattr__(name, value)

        if 'module' not in self.__dict__:
            # ignore the non-existence `module` object
            return super(DataParallel, self).__setattr__(name, value)

        if name not in self.module.__dir__():
            # if `module` does not contain attribute `name`, then this will
            # be thought of as setting this instance's attribute
            return super(DataParallel, self).__setattr__(name, value)

        return setattr(self.module, name, value)

    def __delattr__(self, name):
        """Delete an attribute

        # Arguments
            name [str]: the name an attribute to be deleted
        """
        if '_fully_initialized' not in self.__dict__:
            super(DataParallel, self).__delattr__(name)

        try:
            # check for _parameters, _buffers, _modules in nn.Module first
            return super(DataParallel, self).__delattr__(name)
        except AttributeError:
            # now we sure that this instance does not have 'name' attribute
            return self.module.__delattr__(name)

    def get_layer_indices(self, *args, **kwargs):
        """Get indices of specific type of layer"""
        return self.module.get_layer_indices(*args, **kwargs)

    def get_layer(self, *args, **kwargs):
        """Retrieve a layer"""
        return self.module.get_layer(*args, **kwargs)

    def get_number_layers(self, *args, **kwargs):
        """Get number of layers"""
        return self.module.get_number_layers(*args, **kwargs)

    def x_load(self, *args, **kwargs):
        """Load the saved agent

        # Arguments
            path [str]: the path to saved detail
        """
        return self.module.x_load(*args, **kwargs)

    def x_initialize(self, *args, **kwargs):
        """Initialize the agent"""
        return self.module.x_initialize(*args, **kwargs)

    def x_learn(self, *args, **kwargs):
        """Train the agent given a minibatch of training examples

        # Arguments
            *args [...]: all the data from environment that needed for the
                agent to learn
        """
        return self.module.x_learn(*args, **kwargs)

    def x_train(self, *args, **kwargs):
        """Perform training"""
        return self.module.x_train(*args, **kwargs)

    def x_infer(self, *args, **kwargs):
        """Infer given a data instance"""
        return self.module.x_infer(*args, **kwargs)

    def x_save(self, *args, **kwargs):
        """Save the state

        # Arguments
            state_path [str]: the path to file that store the agent's state
        """
        return self.module.x_save(*args, **kwargs)

    def x_test(self, *args, **kwargs):
        """Perform testing"""
        return self.module.x_test(*args, **kwargs)

    def x_validate(self, *args, **kwargs):
        """Perform validation"""
        return self.module.x_validate(*args, **kwargs)
