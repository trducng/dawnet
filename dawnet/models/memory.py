"""Memory enhanced operations

@author: _john
"""
import torch
import torch.nn as nn


class BaseMemory(nn.Module):
    """BaseMemory class, mostly defines required operations for an external
    module to work with memory
    """

    def __init__(self, n_slots, slot_dim):
        """Initialize the object"""
        super(BaseMemory, self).__init__()

        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.memory = None

        self._modules = []
        self._last_read_vectors = {}
        self._last_attention_reads = {}
        self._last_attention_writes = {}

    def attach_module(self, module, n_reads, n_writes):
        """Register a module into this memory

        # Arguments
            module [nn.Module]: the module that can use this memory
            n_reads [int]: the number of read heads
            n_writes [int]: the number of write heads

        # Returns
            [int]: the index of the registered module
        """
        if isinstance(module, nn.Linear):
            to_controller = nn.Linear(
                in_features=module.in_features + n_reads * self.slot_dim,
                out_features=module.in_features)
            to_interface = nn.Linear(
                in_features=module.out_features,
                out_features=self.get_interface_vector_size(n_reads, n_writes))
            to_output = nn.Linear(
                in_features=module.out_features + n_reads * self.slot_dim,
                out_features=module.out_features)
        elif isinstance(module, (nn.RNNCell, nn.LSTMCell, nn.GRUCell)):
            to_controller = nn.Linear(
                in_features=module.input_size + n_reads * self.slot_dim,
                out_features=module.in_features)
            to_interface = nn.Linear(
                in_features=module.hidden_size,
                out_features=self.get_interface_vector_size(n_reads, n_writes))
            to_output = nn.Linear(
                in_features=module.hidden_size + n_reads * self.slot_dim,
                out_features=module.hidden_size)
        else:
            raise TypeError('Memory has yet to support {}'.format(type(module)))

        self._modules.append({
            'module': module,
            'to_controller': to_controller,
            'to_interface': to_interface,
            'to_output': to_output,
            'n_reads': n_reads,
            'n_writes': n_writes
        })

        return len(self._modules) - 1

    # pylint: disable=arguments-differ
    def forward(self, index, input_, *args):
        """Perform the forward pass

        # @TODO: how to singal resetting the memory, read heads and read vectors?

        # Arguments
            index [int]: the index of the module to work on
            input_ [torch Tensor]: the first dimension should be batch size

        # Returns
            [*args]: anything that the module returns
        """
        batch_size = input_.size(0)
        self.memory = self.initialize_memory(batch_size, self.n_slots, self.slot_dim)

        last_read_vectors = self._last_read_vectors.get(
            index,
            self.initialize_read_heads(batch_size, self.n_reads, self.slot_dim))

        input_to_controller = self._modules[index]['to_controller'](
            torch.concat([input_] + last_read_vectors, axis=1)
        )

        controller_output = self._modules[index]['module'](
            input_to_controller, *args)

        if isinstance(controller_output, tuple) and controller_output:
            hidden_state = controller_output[1:]
            controller_output = controller_output[0]
        else:
            hidden_state = None

        interface = self._modules[index]['to_interface'](controller_output)
        attention_reads = self.address(index, interface)

        read_vectors = [self.memory * each_read for each_read in attention_reads]
        to_output = self._modules[index]['to_output'](
            torch.cat([controller_output] + read_vectors, axis=1))

        self._last_attention_reads[index] = attention_reads
        self._last_read_vectors[index] = read_vectors

        if hidden_state is None:
            return to_output

        return to_output, hidden_state

    # pylint: disable=invalid-name
    def address(self, index, X):
        """Address the memory with interface vector X

        # Arguments
            X [torch.Tensor]: the interface vector
        """
        raise NotImplementedError('memory addressing mechanism not implemented')

    def get_interface_vector_size(self, n_reads, n_writes):
        """Get the interface vector size based on the number of read and write

        # Arguments
            n_reads [int]: the number of read heads
            n_writes [int]: the number of write heads

        # Returns
            [int]: the interface vector size
        """
        raise NotImplementedError('interface vector size not calculated')

    def initialize_memory(self, batch_size, n_slots, slot_dim):
        """Initialize the memory

        # Arguments
            n_slots [int]: the number of memory slot
            slot_dim [int]: each memory slot dimension

        # Returns
            [2D torch.Tensor]: the memory
        """
        return torch.randn(batch_size, n_slots, slot_dim)

    def initialize_read_heads(self, batch_size, n_reads, slot_dim):
        """Initialize read head vectors

        # Arguments
            n_reads [int]: the number of read heads
            slot_dim [int]: each memory slot dimension

        # Returns
            [list of 1D torch.Tensor]: list of read heads
        """
        return [torch.randn(batch_size, slot_dim) for _ in n_reads]


class MemoryWrapper(nn.Module):
    """Wrap a memory with a module

    Currently, memory only works with Linear and RNN cell.

    # Arguments
        module [nn.Module]: a normal module we wish to run
        memory [BaseMemory]: the memory we want to add to
    """

    def __init__(self, module, memory, n_reads=10, n_writes=10):
        """Initialize the object"""
        super(MemoryWrapper, self).__init__()

        self.memory = memory
        self.index = self.memory.attach_module(module, n_reads, n_writes)

    # pylint: disable=arguments-differ
    def forward(self, *args):
        """Perform the forward pass"""
        return self.memory.forward(index=self.index, *args)
