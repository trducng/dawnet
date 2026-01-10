import re
import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from collections import OrderedDict
from copy import deepcopy
from typing import Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class OpInfo:
  op: "Op"
  layers: list[str]
  enabled: bool


class Handler:
  """The handler run along side each module

    Args:
        name: the name of the module
        stage: the running stage of the handler, can be: forward, forward_pre,
            backward, backward_pre
        inspector: the inspector that this handler belongs to
    """
  def __init__(self, name: str, stage: str, inspector: "Inspector"):
    self._name = name
    self._stage = stage
    self._inspector = inspector

  def __call__(self, *args, **kwargs):
    if self._stage == "forward":
      return self.forward(*args, **kwargs)
    elif self._stage == "forward_pre":
      return self.forward_pre(*args, **kwargs)
    elif self._stage == "backward":
      return self.backward(*args, **kwargs)
    elif self._stage == "backward_pre":
      return self.backward_pre(*args, **kwargs)

    raise ValueError(
        f"Unknown stage {self._stage}. Should be one of "
        '["forward", "forward_pre", "backward", "backward_pre"]'
    )

  def forward_pre(self, module, args, kwargs):
    if self._name not in self._inspector._module_to_op:
      return args, kwargs

    args, kwargs = args, kwargs
    for op in self._inspector._module_to_op[self._name]:
      if self._inspector._ops[op.id].enabled:
        if self._inspector._debug > 3:
          print(f"(v) {self._name}: {op} forward_pre")
        args, kwargs = op.forward_pre(
            self._inspector, self._name, module, args, kwargs
        )

    return args, kwargs

  def forward(self, module, args, kwargs, output):
    if self._name not in self._inspector._module_to_op:
      return output

    output = output
    for op in reversed(self._inspector._module_to_op[self._name]):
      if self._inspector._ops[op.id].enabled:
        if self._inspector._debug > 3:
          print(f"(v) {self._name}: {op} forward")
        output = op.forward(
            self._inspector, self._name, module, args, kwargs, output
        )

    return output

  def backward_pre(self, module, grad_output):
    if self._name not in self._inspector._module_to_op:
      return grad_output

    grad_output = grad_output
    for op in self._inspector._module_to_op[self._name]:
      if self._inspector._ops[op.id].enabled:
        if self._inspector._debug > 3:
          print(f"(v) {self._name}: {op} backward_pre")
        grad_output = op.backward_pre(
            self._inspector, self._name, module, grad_output
        )

    return grad_output

  def backward(self, module, grad_input, grad_output):
    if self._name not in self._inspector._module_to_op:
      return grad_input

    grad_input = grad_input
    for op in self._inspector._module_to_op[self._name]:
      if self._inspector._ops[op.id].enabled:
        if self._inspector._debug > 3:
          print(f"(v) {self._name}: {op} backward")
        grad_input = op.backward(
            self._inspector, self._name, module, grad_input, grad_output
        )

    return grad_input


class Op:
  """Stateless operation"""
  def __init__(self):
    self.id = str(uuid.uuid4())

  def forward(
      self, inspector: "Inspector", name: str, module, args, kwargs, output
  ):
    return output

  def forward_pre(
      self, inspector: "Inspector", name: str, module, args, kwargs
  ):
    return args, kwargs

  def backward(
      self, inspector: "Inspector", name: str, module, grad_input, grad_output
  ):
    return grad_input

  def backward_pre(
      self, inspector: "Inspector", name: str, module, grad_output
  ):
    return grad_output

  def inspector_pre_run(
      self, insp: "Inspector", run_params: dict | None = None
  ):
    ...

  def inspector_post_run(
      self, insp: "Inspector", run_params: dict | None = None
  ):
    ...

  def add(self, inspector: "Inspector"):
    pass

  def remove(self, inspector: "Inspector"):
    pass

  def enable(self, inspector: "Inspector"):
    pass

  def disable(self, inspector: "Inspector"):
    pass

  def run_params(self, *args, **kwargs) -> dict:
    return {"id": self.id, **kwargs}


class RunState:
  def __init__(self):
    self._defs = {}
    self._states = {}
    self.register("input", {})
    self.register("output", {})

  def clear(self, names: list[str] | str | None = None):
    """Clear the section, if not specified, clear all"""
    if names is not None:
      if not isinstance(names, list):
        names = [names]
      for name in names:
        if name in self._defs:
          self._states[name] = deepcopy(self._defs[name])
        else:
          logger.warning(f"State with name {name} doesn't exist")
      return

    for name, default in self._defs.items():
      self._states[name] = deepcopy(default)

  def clone(self, empty: bool = False):
    """Clone the state to a new object

        Args:
            empty: if True, return an empty state with the same definition

        Returns:
            RunState: the cloned state
        """
    state = RunState()
    state._defs = self._defs
    if empty:
      state.clear()
    else:
      state._states = deepcopy(self._states)
    return state

  def __contains__(self, name):
    return name in self._states

  def register(self, name, default=None):
    if name not in self._defs:
      self._defs[name] = default
      self._states[name] = deepcopy(default)

  def deregister(self, name):
    self._states.pop(name, None)
    self._defs.pop(name, None)

  def keys(self):
    return self._states.keys()

  def values(self):
    return self._states.values()

  def items(self):
    return self._states.items()

  def __getitem__(self, key):
    if key not in self._defs:
      logger.info(f"State with name {key} doesn't exist")
    return self._states[key]

  def __setitem__(self, key, value):
    if key not in self._defs:
      logger.info(f"State with name {key} doesn't exist. Might .register first")
    self._states[key] = value

  def __str__(self):
    return str(self._states)

  def __repr__(self):
    return repr(self._states)

  def __len__(self):
    return len(self._states)

  def __iter__(self):
    return iter(self._states)


def copy_model(module: nn.Module) -> nn.Module:
  """Deep cop a model but shallow copy the parameters to not blow up the memory"""
  from copy import deepcopy, _deepcopy_dispatch, _deepcopy_atomic

  _deepcopy_dispatch[torch.Tensor] = _deepcopy_atomic
  _deepcopy_dispatch[nn.Parameter] = _deepcopy_atomic
  try:
    new_module = deepcopy(module)
  finally:
    del _deepcopy_dispatch[torch.Tensor]
    del _deepcopy_dispatch[nn.Parameter]
  return new_module


class Inspector(nn.Module):
  """Wrap around a model to help inspect its activity

    Args:
        model: the model to inspect
        state: the internal state of inspection
        debug: the debug level: 0 - no debug, 1 - full error stacktrace, 2 - warning,
            3 - info, 4 - debug
        include_backward: whether to enable backward hooks

    Attributes:
        original_model: the supplied model
        model: the shallow cloned original model will be used to inspect
        _module_to_op: mapping from module name to list of `op` objects
        _ops: contain ops information: op_id -> (op, module_name)
        _private_op_state: private space for the op to store its progress and to
            communicate between layers All `ops` can have access to this space.
        state: contains pulbic information that the op populates
    """
  def __init__(
      self,
      model: nn.Module,
      state: None | RunState = None,
      debug: int = 0,
      include_backward: bool = False,
  ):
    super().__init__()
    self._model = copy_model(model)
    self._original_model = model
    self._debug = debug
    self._include_backward = include_backward

    self._module_to_op: dict[str, list[Op]] = OrderedDict()
    self._ops: dict[str, OpInfo] = OrderedDict()  # op, module name, enable
    self.ops = []

    self.state: RunState = state or RunState()
    self._private_op_state = {}
    self._op_params: dict = {}

    for name, module in self._model.named_modules():
      module.register_forward_pre_hook(
          Handler(name, "forward_pre", self), with_kwargs=True
      )
      module.register_forward_hook(
          Handler(name, "forward", self), with_kwargs=True, always_call=True
      )
      if include_backward:
        module.register_full_backward_pre_hook(
            Handler(name, "backward_pre", self)
        )
        module.register_full_backward_hook(Handler(name, "backward", self))

  @property
  def model(self):
    return self._model

  @property
  def original_model(self):
    return self._original_model

  def add(
    self,
    op: Op,
    name: str | list[str] | None = None,
    name_filter: Callable | None = None,
    name_regex: str | None = None,
    verbose: bool = True,
  ) -> Op:
    """Add op to the inspector

        Args:
            op: the operation to interfere
            name: the name of layer that the op is hooked on
            name_filter: a callable that filter the name of desired layer
            name_regex: a regex pattern to get the layers with matched name

        Returns:
            the op object
        """
    layers = self.get_layers(name, name_filter, name_regex)
    if not layers:
      raise ValueError("No layer found")

    for layer in list(layers):
      if layer not in self._module_to_op:
        self._module_to_op[layer] = []
      self._module_to_op[layer].append(op)
    if verbose:
      print(f"Added to layer {layers}")

    self._ops[op.id] = OpInfo(op=op, layers=layers, enabled=True)
    self.ops.append(op)
    op.add(self)

    return op

  def move(
      self,
      op: Op,
      name: str | list[str] | None = None,
      name_filter: Callable | None = None,
      name_regex: str | None = None,
  ) -> Op:
    """Move an operation from one layer to another layer

        Args:
            op: the operation to interfere
            name: the name of layer that the op is hooked on
            name_filter: a callable that filter the name of desired layer
            name_regex: a regex pattern to get the layers with matched name

        Returns:
            the op object
        """
    self.remove(op)
    return self.add(op, name, name_filter, name_regex)

  def has_op(self, op: Op) -> bool:
    """Check if the op is in the inspector"""
    return op.id in self._ops

  def get_layers(
      self,
      name: str | list[str] | None = None,
      name_filter: Callable | None = None,
      name_regex: str | None = None,
  ) -> list[str]:
    """Get the name of layers that match the filter"""
    layers: set[str] = set()

    if name is not None:
      if isinstance(name, str):
        name = [name]

      for en in list(set(name)):
        if en == "" or self._model.get_submodule(en):
          layers.add(en)
        else:
          raise ValueError(f"Module with name {en} doesn't exist")

    if name_filter:
      for name, _ in self._model.named_modules():
        if name_filter(name):
          layers.add(name)

    if name_regex:
      regex_name = re.compile(name_regex)
      for name, _ in self._model.named_modules():
        if regex_name.match(name):
          layers.add(name)

    layers_list = [
        name for name, _ in self._model.named_modules() if name in layers
    ]

    return layers_list

  def remove(self, op: Op):
    if op.id not in self._ops:
      raise ValueError(f"Op with id {op.id} doesn't exist")
    op.remove(self)
    layers = self._ops[op.id].layers
    for layer in layers:
      self._module_to_op[layer] = [
          each for each in self._module_to_op[layer] if each.id != op.id
      ]
    del self._ops[op.id]
    self.ops = [each for each in self.ops if each.id != op.id]

  def remove_all(self):
    for _ in range(len(self.ops)):
      self.remove(self.ops[0])

  def enable(self, op: Op):
    if op.id not in self._ops:
      raise ValueError(f"Op with id {op.id} doesn't exist")
    self._ops[op.id].enabled = True
    op.enable(self)

  def disable(self, op: Op):
    if op.id not in self._ops:
      raise ValueError(f"Op with id {op.id} doesn't exist")
    self._ops[op.id].enabled = False
    op.disable(self)

  def copy(self) -> "Inspector":
    """Create a copy of the inspector"""
    return Inspector(self._original_model)

  def __str__(self):
    if not self._ops:
      return "No ops added"

    strs = ["Inspector ops:"]
    for idx, op in enumerate(self.ops):
      op_info = self._ops[op.id]
      layers = [
          name
          for name, _ in self._model.named_modules() if name in op_info.layers
      ]  # print the layers in the order that they are in the model
      s = f"- [{idx}] {op} @ {layers}"
      if not op_info.enabled:
        s += " (disabled)"
      strs.append(s)
    return "\n".join(strs)

  def __repr__(self):
    return repr(self._model)

  def __setattr__(self, name, value):
    if name in ["_model", "_original_model"]:
      self.__dict__[name] = value
      return
    return super().__setattr__(name, value)

  def __getstate__(self):
    """Save the runner session"""
    skips = ["_model"]
    state = {}
    for key, value in self.__dict__.items():
      if key == skips:
        continue
    return state

  def __setstate__(self, state: dict):
    self.__dict__.update(state)
    self._model = copy_model(state["_original_model"])
    for name, module in self._model.named_modules():
      module.register_forward_pre_hook(
          Handler(name, "forward_pre", self), with_kwargs=True
      )
      module.register_forward_hook(
          Handler(name, "forward", self), with_kwargs=True, always_call=True
      )
      if state["_include_backward"]:
        module.register_full_backward_pre_hook(
            Handler(name, "backward_pre", self)
        )
        module.register_full_backward_hook(Handler(name, "backward", self))

  def save(self, location):
    import dill

    with open(location, "wb") as fo:
      dill.dump(self, fo)

  @classmethod
  def load(cls, location):
    import dill

    with open(location, "rb") as fi:
      obj = dill.load(fi)
    return obj

  def forward(self, *args, **kwargs):
    return self._model(*args, **kwargs)

  def begin(self, op_params: list[dict] | None = None) -> RunState:
    if op_params:
      for op_param in op_params:
        if op_param["id"] in self._op_params:
          logger.warning(
              f"Op with id {op_param['id']} already exists, overwriting"
          )
        self._op_params[op_param["id"]] = op_param

    for op_info in self._ops.values():
      if op_info.enabled:
        run_params = (
            self._op_params[op_info.op.id]
            if op_info.op.id in self._op_params else None
        )
        op_info.op.inspector_pre_run(self, run_params)

    return self.state

  def finish(self, detach_state: bool = False):
    for op_info in reversed(self._ops.values()):
      if op_info.enabled:
        run_params = (
            self._op_params[op_info.op.id]
            if op_info.op.id in self._op_params else None
        )
        op_info.op.inspector_post_run(self, run_params)
    self._op_params.clear()
    if detach_state:
      self.state = self.state.clone(empty=True)
    else:
      self.state.clear()

  @contextmanager
  def ctx(
      self, op_params: list[dict] | None = None, detach_state: bool = False
  ):
    state = self.begin(op_params)
    try:
      yield state
    finally:
      self.finish(detach_state=detach_state)

  def run(
      self,
      *args,
      _refresh_state: bool = True,
      _method: str | None = None,
      _op_params: list[dict] | None = None,
      **kwargs,
  ):
    """Run the model"""
    if _refresh_state:
      self.state.clear()
    self._op_params.clear()
    if _op_params:
      for op_param in _op_params:
        if op_param["id"] in self._op_params:
          logger.warning(
              f"Op with id {op_param['id']} already exists, overwriting"
          )
        self._op_params[op_param["id"]] = op_param

    for op_info in self._ops.values():
      if op_info.enabled:
        run_params = (
            self._op_params[op_info.op.id]
            if op_info.op.id in self._op_params else None
        )
        op_info.op.inspector_pre_run(self, run_params)

    if _method:
      output = getattr(self._model, _method)(*args, **kwargs)
    else:
      output = self._model(*args, **kwargs)

    for op_info in reversed(self._ops.values()):
      if op_info.enabled:
        run_params = (
            self._op_params[op_info.op.id]
            if op_info.op.id in self._op_params else None
        )
        op_info.op.inspector_post_run(self, run_params)

    return output, self.state


class LLMInspector(Inspector):
  """Inspector with utilities specialized for LLM"""
  _MODELS = {
    "deepseekv3.2": "deepseek-ai/DeepSeek-V3.2",
    "gemma-7b": "google/gemma-7b-it",
    "gemma2-2b": "google/gemma-2-2b-it",
    "gemma3-4b": "google/gemma-3-4b-it",
    "gemma3-27b": "google/gemma-3-27b-it",
    "glm4.6": "zai-org/GLM-4.6",
    "gptoss-120b": "openai/gpt-oss-120b",
    "gptoss-20b": "openai/gpt-oss-20b",
    "mistral3": "mistralai/Mistral-Large-3-675B-Instruct-2512",
    "ministral3-8b-instruct": "mistralai/Ministral-3-8B-Instruct-2512",
    "olmo3-7b-base": "allenai/Olmo-3-1025-7B",
    "olmo3-7b-think-sft": "allenai/Olmo-3-7B-Think-SFT",
    "olmo3-7b-think-dpo": "allenai/Olmo-3-7B-Think-DPO",
    "olmo3-7b-think": "allenai/Olmo-3-7B-Think",
    "olmo3-7b-instruct-sft": "allenai/Olmo-3-7B-Instruct-SFT",
    "olmo3-7b-instruct-dpo": "allenai/Olmo-3-7B-Instruct-DPO",
    "olmo3-7b-instruct": "allenai/Olmo-3-7B-Instruct",
    "olmo3-7b-zero": "allenai/Olmo-3-7B-RL-Zero-Math",
    "olmo3.1-32b-think": "allenai/Olmo-3.1-32B-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-4b-instruct": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-4b-thinking": "Qwen/Qwen3-4B-Thinking-2507",
    "qwen3-30b": "Qwen/Qwen3-30B-A3B-Base",
    "phi4": "microsoft/phi-4",
  }

  def __init__(
      self,
      model: nn.Module,
      state: None | RunState = None,
      debug: int = 0,
      include_backward: bool = False,
  ):
    super().__init__(
        model=model,
        state=state,
        debug=debug,
        include_backward=include_backward
    )
    self._tokenizer = getattr(model, "tokenizer", None)
    self.model_id = None
    self.device = None

  @property
  def tokenizer(self):
    return self._tokenizer

  @tokenizer.setter
  def tokenizer(self, tokenizer):
    self._tokenizer = tokenizer

  @classmethod
  def from_hf(cls, name):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(name)
    insp = cls(model)
    insp.tokenizer = tokenizer
    insp.model_id = name
    insp.device = model.device
    return insp

  def encode(self, msg: list | str | torch.Tensor, chat: bool = False):
    if chat:
      if isinstance(msg, str):
        msg = [{"role": "user", "content": msg}]
      if isinstance(msg, list):
        if msg and isinstance(msg[0], dict):
          if self.tokenizer is None:
            raise ValueError(
                "Missing tokenizer. Attach tokenizer with insp.tokenizer = ..."
            )
          tokens = self.tokenizer.apply_chat_template(
              msg, add_generation_prompt=True, return_tensors="pt"
          )
        elif msg and isinstance(msg[0], int):
          tokens = torch.tensor(msg, dtype=torch.int)
        else:
          raise ValueError(f"{msg} What is this?")
      else:
        tokens = msg
    else:
      if isinstance(msg, str):
        if self.tokenizer is None:
          raise ValueError(
              "Missing tokenizer. Attach tokenizer with insp.tokenizer = ..."
          )
        tokens = self.tokenizer.encode(msg, return_tensors="pt")
      elif isinstance(msg, list):
        tokens = torch.tensor(msg, dtype=torch.int)
      else:
        tokens = msg

    tokens = tokens.to(device=self.model.device)

    return tokens

  def generate(
      self,
      msg: list | str | torch.Tensor,
      max_new_tokens=4096,
      use_original: bool = False,
      chat: bool = True
  ):
    from .tokens import Tokens
    tokens = self.encode(msg, chat=chat)
    if use_original:
      out = self.original_model.generate(tokens, max_new_tokens=max_new_tokens)
    else:
      out = self.model.generate(tokens, max_new_tokens=max_new_tokens)
    completed_tokens = Tokens(tensor=out, tokenizer=self.tokenizer)
    return completed_tokens

  def infer(
      self,
      msg: list | str | torch.Tensor,
      use_original: bool = False,
      chat: bool = True
  ):
    tokens = self.encode(msg, chat=chat)
    if use_original:
      out_tensor = self.original_model(tokens)
    else:
      out_tensor = self.model(tokens)
    return LogitsTensor(logits=out_tensor.logits[0], tokenizer=self.tokenizer)


class LogitsTensor:
  """A wrapper of logits tensor to aid with quickly inspecting the logits

    Args:
        logits: 1D or 2D tensor, if 1D, we only concern about logits of a single token
            if 2D, we are dealing with logits of multiple tokens (possibly the whole
            generated sequence)
        tokenizer: the tokenizer to interpret this generated logits
        idxs: the index inside the logits that we want to focus on
    """
  def __init__(self, logits, tokenizer, idxs: list | None = None):
    self._logits = logits
    self.tokenizer = tokenizer
    self.idxs: list = idxs if idxs is not None else []

  def __getitem__(self, idx):
    return LogitsTensor(
        logits=self.logits, tokenizer=self.tokenizer, idxs=self.idxs + [idx]
    )

  def argmax(self):
    """Get the highest class for the last token"""
    logits = self.logits
    if len(logits.shape) > 1:
      logits = logits[-1]
    tok_idx = logits.argmax()
    toks = self.tokenizer.decode(tok_idx)
    return tok_idx, toks

  def topk(self, k):
    logits = self.logits
    if len(logits.shape) > 1:
      logits = logits[-1]
    top = logits.topk(k)
    toks = [self.tokenizer.decode(each) for each in top.indices]
    return top.values, top.indices, toks

  @property
  def logits(self):
    logits = self._logits
    for idx in self.idxs:
      logits = logits[idx]
    return logits

  def shape(self):
    return self.logits.shape

  def __str__(self):
    return f"LogitTensors (shape {self.logits.shape}): {self.argmax()}"


def get_attention_contrib():
  most_important = {}
  for layer_idx in range(36):
    attn = state['output'][f'model.layers.{layer_idx}.self_attn'][1]  # BxN
    sorted_attn = attn[0].sort(dim=-1, descending=True)  # N
    sorted_values = (torch.cumsum(sorted_attn.values, dim=-1)
                     > 0.95).nonzero().min(dim=-1) + 1  # N
    for head_idx in range(attn.shape[1]):
      for token_idx in range(attn.shape[2]):
        most_important[(layer_idx, head_idx, token_idx)] = sorted_attn.indices[
            head_idx,
            token_idx, :sorted_values[head_idx,
                                      token_idx].item()].cpu().tolist()
  return most_important


# vim: ts=2 sts=2 sw=2 et
