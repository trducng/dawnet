# Basic interface for a model
# @author: John
# =============================================================================
import uuid
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

import dawnet
from dawnet.utils.dependencies import get_pytorch_layers
from dawnet.utils.names import get_random_name


class Hyperparams(dict):
    """Extend dictionary to allow dot notation"""

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("The object does not have attribute \"{key}\"")

    def __setattr__(self, key, value):
        self[key] = value


class MetaAgent(type):
    """Meta-class for Agent to re-organize agent attributes during object creation
    """

    def __new__(cls, name, bases, attrs, **kwargs):

        # create a default `hparams` that contains hyperparams
        hparams = Hyperparams()
        if 'hparams' in attrs:
            if not isinstance(
                attrs['hparams'],
                (Hyperparams, dawnet.models.perceive.Hyperparams)
            ):
                raise AttributeError(
                    f'`hparams` should has type `Hyperparams`, instead '
                    '{type(attrs["hparams"])}')
            hparams = Hyperparams(attrs['hparams'])

        if 'params' in attrs:
            raise AttributeError(f'`params` is a protected attribute')

        if 'name' in attrs:
            raise AttributeError(f'`name` is a protected attribute')

        callables = []
        new_attrs = {}
        for name, value in attrs.items():

            if isinstance(value, Hyperparams):
                # consolidate into hyperparameters
                if name == 'hparams':
                    continue
                hparams[name] = value

            else:
                new_attrs[name] = value
                if callable(value):
                    new_attrs[f'__func_{name}'] = value
                    callables.append(name)

        new_attrs['hparams'] = hparams
        new_attrs['__callables__'] = callables

        return super().__new__(cls, name, bases, new_attrs, **kwargs)


class Agent(metaclass=MetaAgent):
    """An AI agent"""
    def __init__(self, params=None, name=None):
        params = {} if params is None else params
        self.hparams.update(params)
        self.name = name

    def __getattr__(self, name):
        raise AttributeError(f'The agent does not have attribute "{name}"')

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def eval(self):
        """Alias `nn.Module.eval`"""
        for each_callable in self.__callables__:
            module = getattr(self, each_callable)
            if isinstance(module, nn.Module):
                module.eval()

    def train(self):
        """Alias `nn.Module.train`"""
        for each_callable in self.__callables__:
            module = getattr(self, each_callable)
            if isinstance(module, nn.Module):
                module.train()

    def forward(self, *args, **kwargs):
        """Perform forward pass"""
        modules = []
        for each_callable in self.__callables__:
            module = getattr(self, each_callable)
            if isinstance(module, nn.Module):
                modules.append(module)

        if len(modules) == 1:
            return modules[0](*args, **kwargs)

        raise NotImplementedError('need subclass if there are other than 1 `nn.Module`')

    def learn(self, *args, **kwargs):
        # @TODO: should do some data processing here (can be with augmentation...)
        # the data processing and getting data can be specified based on the
        # environment
        # @TODO: does it introduce complication to users? It will eb if the definition
        # of the environment is complicated and not intuitive for them to define
        modules, optimizers = [], []
        for each_callable in self.__callables__:
            module = getattr(self, each_callable)
            if isinstance(module, nn.Module):
                modules.append(module)
                continue
            if isinstance(module, optim.Optimizer):
                optimizers.append(module)
                continue

        if len(modules) != 1 or len(optimizers) != 1:
            raise NotImplementedError('need subclass if there are other than 1 '
                                      '`nn.Module`')

        loss = modules[0].loss(*args, **kwargs)
        optimizers[0].zero_grad()
        loss.backward()
        optimizers[0].step()

        return loss.detach()

    def wake(self, soul=None, cuda=None):
        """Initialize all modules

        This method automatically initiates:
            - all modules that have `load_state_dict`
            - hparams

        # Args
            soul <str>: path that stores state of the agent. If not provided,
                just initiate the agent
            cuda <bool>: whether to load model into gpu. If None, load to GPU if the
                system has GPU
        """
        if cuda is None:
            cuda = torch.cuda.is_available()

        # initiate all modules
        for key in self.__callables__:
            module = getattr(self, f'__func_{key}', None)
            if module is not None:
                module = module()
                if cuda and hasattr(module, 'cuda'):
                    module.cuda()
                setattr(self, key, module)

        if soul is None:
            return

        # load from previous rest
        state = torch.load(soul, map_location='cpu')
        for key, value in state.items():
            if key == 'hparams':
                self.hparams.update(value)
                continue

            if key in self.__callables__:
                module = getattr(self, key, None)
                if isinstance(module, nn.Module):
                    module.load_state_dict(value)

                if isinstance(module, optim.Optimizer):
                    module.load_state_dict(value)

                continue

            # free style values
            setattr(self, key, value)

    def rest(self, soul):
        """Let the agent rest

        This method will automatically save:
            - all modules that have state_dict
            - hparams

        # Args
            soul <str>: path to store state of the agent
        """
        state = {}
        for attr_str in dir(self):
            attr = getattr(self, attr_str)
            if attr_str.startswith('__'):
                # ignore attr like __dict__, __doc__...
                continue

            if hasattr(attr, 'state_dict'):
                state[attr_str] = attr.state_dict()
                continue

            if callable(attr):
                continue

            state[attr_str] = attr

        torch.save(state, soul)


class Module(nn.Module):
    """Base class provides perception interface

    Think of this class as human's perception and intuition. The perception is
    provided with some external stimuli, then automatically and unconsiously
    links those stimuli to some pattern.

    Normal flow when training from scratch
        x_initialize ---> x_learn / x_train ---> x_save

    Normal flow when resume training
        x_initialize ---> x_load ---> x_learn / x_train ---> x_save

    Normal flow when evaluation
        x_initialize ---> x_load ---> x_infer
    """
    _hooks_def = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hooks = defaultdict(dict)

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

        @TODO: this is not reliable, because the named module might not be in
        correct order

        # Arguments
            layer_type [torch.nn.Module]: the type of layer to compare on

        # Returns
            [list of ints]: list of indices that has layer match `layer_type`
        """
        idx = 0
        indices = []
        for _, (_, layer) in enumerate(self.named_modules()):
            if type(layer) not in get_pytorch_layers():
                continue

            if isinstance(layer, layer_type):
                indices.append(idx)

            idx += 1

        return indices

    def restart(self):
        """Restart the model"""
        # @TODO: there can be restart like reverting to original random state or
        # original real state
        # @TODO: for the first kind of `restart`, maybe rename to `wake`
        # Example data
        self.apply(self.weight_bias_init)

    def get_layer(self, layer_idx):
        """Get the layer that has `layer_idx` in `named_modules()`

        Note that the layer is filterred, in that non-native Pytorch modules
        (those that are constructed by e.g. Sequential...) are ignored.

        # Arguments
            layer_idx [int]: the final layer index to retrieve output

        # Returns
            [torch.nn.Module]: the specific layer
        """
        return self.get_layers()[layer_idx][1]

    def get_layers(self):
        """Get all layers

        # Returns
            [list of tuple of strs and Module]: layer name and specific layer
        """
        # @TODO: it is not reliable to use self.named_modules
        layers = []
        for name, layer in self.named_modules():
            if not isinstance(layer, tuple(get_pytorch_layers())):
                continue

            layers.append((name, layer))

        return layers

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

    def _get_progress(self):
        """Get the training progress"""
        progress = self.get_progress()
        progress['itr'] = self.training_iteration

        if self.history and progress['itr'] == self.history[-1]['itr']:
            return

        self.history.append(progress)

    def _get_save_state(self):
        """Get the save state and collect other minor information

        @NOTE:
            - This method assumes a lot of state here, although it would be better
            to auto-discover this information
        """
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

    def register_hook(self, name, fn) -> List[str]:
        """Register hook with desired name and function"""
        result = []
        if name in self._hooks_def:
            id_ = uuid.uuid4().hex
            self._hooks[name][id_] = fn
            result.append(id_)

        for module in self.children():
            if isinstance(module, Module):
                result += module.register_hook(name, fn)

        return result

    def run_hook(self, name, kwargs) -> Dict:
        """Run hook"""
        if name in self._hooks:
            for fn in self._hooks[name].values():
                kwargs = fn(**kwargs)

        return kwargs


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
