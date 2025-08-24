"""Trace the computation path of a model"""
import contextlib
import hashlib
import sys
from typing import Callable, Iterator

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map, tree_iter

from .inspector import Inspector

_module_stack = []
_skip_funcs = {
  "aten::_to_copy",
  "aten::slice",
  "aten::transpose",
  "aten::expand",
  "aten::_unsafe_view",
  "aten::view",
  "aten::detach",
  "aten::unsqueeze",
  "aten::masked_fill",
  "aten::clone",
}


class TraceTensor(torch.Tensor):
  elem: torch.Tensor
  module: str
  parent_ids: list[str]
  op: str

  __slots__ = ['elem', 'parent_ids', 'op']

  @staticmethod
  def __new__(cls, elem, *args, **kwargs):
    r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
      cls, elem.size(),
      strides=elem.stride(), storage_offset=elem.storage_offset(),
      dtype=elem.dtype, layout=elem.layout,
      device=elem.device, requires_grad=elem.requires_grad
    )
    r.elem = elem
    r.module = ""
    r.parent_ids = []
    r.op = ""
    return r

  def __repr__(self):
    return f"TraceTensor({self.elem})"

  def __str__(self):
    if self.op and self.parent_ids:
      pretty = f"{self.op}[{self.get_id()}]"
    elif self.op:
      pretty = self.op
    else:
      pretty = ""
    return f"TraceTensor({pretty})"

  def get_id(self):
    if not self.op:
      return ""
    if not self.parent_ids:
      return ""
    parent_ids = ", ".join(self.parent_ids)
    id_ = hashlib.md5(parent_ids.encode()).hexdigest()[:6]
    return id_

  @classmethod
  def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
    def unwrap(e):
      return e.elem if isinstance(e, TraceTensor) else e

    parents = []
    for arg in tree_iter(args):
      if isinstance(arg, torch.Tensor):
        parents.append(arg)
    if kwargs:
      for kw, arg in kwargs.items():
        if isinstance(arg, torch.Tensor):
          parents.append(arg)

    parent_ids = []
    unnamed_counter = 0
    for each in parents:
      if isinstance(each, TraceTensor):
        if name := each.get_id():
          parent_ids.append(name)
          continue
      if _module_stack:
        parent_ids.append(f"{_module_stack[-1]}.tensor_{unnamed_counter}")
      else:
        parent_ids.append(f"tensor_{unnamed_counter}")
      unnamed_counter += 1

    def wrap(e):
      if not isinstance(e, torch.Tensor):
        return e
      ts = TraceTensor(e)
      ts.parent_ids = parent_ids
      ts.op = func._schema.name
      if _module_stack:
        ts.module = _module_stack[-1]
      return ts

    rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
    if isinstance(rs, TraceTensor) and func._schema.name not in _skip_funcs:
      print(f"  - {func._schema.name=}, {rs}")
    return rs


def get_fw_prehook(name) -> Callable:
  def hook(module, args, kwargs):
    print(f"Module start: {name}")
    _module_stack.append(name)
    def wrap(e):
      if isinstance(e, TraceTensor):
        return e
      if not isinstance(e, torch.Tensor):
        return e
      return TraceTensor(e)
    if args:
      args = tree_map(wrap, args)
    if kwargs:
      kwargs = tree_map(wrap, kwargs)
    return args, kwargs

  return hook


def get_fw_hook(name) -> Callable:
  def hook(module, args, kwargs, output):
    print(f"Module end: {name}")
    if _module_stack:
      _module_stack.pop()

  return hook


def equip_path_tracing(module: nn.Module, prefix=""):
  if isinstance(module, Inspector):
    # register the Op
    return

  # register the forward_pre and forward hooks
  for name, sub in module.named_modules():
    sub_name = ".".join([prefix, name]) if prefix else name
    sub.register_forward_pre_hook(get_fw_prehook(sub_name), with_kwargs=True)
    sub.register_forward_hook(get_fw_hook(sub_name), with_kwargs=True)
# vim: ts=2 sts=2 sw=2 et
