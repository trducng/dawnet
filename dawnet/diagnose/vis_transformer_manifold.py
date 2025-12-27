"""Visualize model computational manifold"""
import re
from typing import Literal

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from dawnet.op import GetOutput


class ManifoldExperiment:
  def __init__(self, insp, ndim=2):
    self.insp = insp
    self.ndim = ndim
    self.get_op = None
    self.embed_name = None

    self._add_bos_token = False
    if hasattr(self.insp.tokenizer, "add_bos_token"):
      self._add_bos_token = self.insp.tokenizer.add_bos_token

    # raw columns -> might treat this as sqlite to manage relationship
    self.phrases, self.raw_pos, self.labels, self.styles, self.skip_add_bos = [], [], [], [], []
    self.sizes = []

    # processed from insp
    self.flattened_tokens, self.pos = [], []  # depend on insp model
    self.wlabels, self.wstyles, self.wsizes = [], [], []
    self.acts, self.names = {}, []

    # coordinates for visualization
    self.pcas = {}

  def add_phrase(
   self, phrase: str, pos: str | Literal[-1, 0]=-1, label="default", style="o", size=1, skip_add_bos=True,
  ):
    """Construct the phrase

    Args:
      phrase: the phrase to track
      pos: If str, this is the str in `phrase` that mark the token to track; if -1
        track the last token; if 0, track all token
      label: the str label for the phrase, will determine the vis color in scatter plot
      style: will determine vis shape in scatter plot
    """
    self.phrases.append(phrase)
    self.raw_pos.append(pos)
    self.labels.append(label)
    self.styles.append(style)
    self.sizes.append(size)
    self.skip_add_bos.append(self._add_bos_token and skip_add_bos)

    tokens = self.insp.tokenizer.tokenize(phrase)
    if isinstance(pos, str):
      str_idx, token_idx = [], []
      idx = phrase.index(pos)
      while idx != -1:
        str_idx.append(idx)
        phrase = phrase[:idx] + phrase[idx+len(pos):]
        idx = phrase.index(pos)
      if not str_idx:
        raise ValueError(f"Cannot find pattern {pos} in phrase {phrase}")

      cidx = 0
      for idx, tok in enumerate(tokens):
        if not str_idx:
          break
        cidx += len(tok)
        if cidx > str_idx[0]:
          self.flattened_tokens.append(tokens[idx])
          if self._add_bos_token:
            token_idx.append(idx+1)
          else:
            token_idx.append(idx)
          str_idx = str_idx[1:]
      self.pos.append(token_idx)
    else:
      if pos == -1:
        self.pos.append([-1,])
        self.flattened_tokens.append(tokens[-1])
      else:
        self.flattened_tokens += tokens[-1]
        if self._add_bos_token:
          self.pos.append(list(range(1, len(tokens)+1)))
        else:
          self.pos.append(list(range(len(tokens))))

    self.wlabels += [label] * len(self.pos[-1])
    self.wstyles += [style] * len(self.pos[-1])
    self.wsizes += [size] * len(self.pos[-1])

    if not self.acts:
      return

    with self.insp.ctx() as state:
      token = self.insp.tokenizer.encode(phrase, return_tensors="pt").to(
        self.insp.device
      )
      if self._add_bos_token and skip_add_bos:
        token = token[:,1:]
      _ = self.insp.model(token)

      for name in self.names:
        if name not in state['output']:
          continue

        if isinstance(state['output'][name], tuple):
          # TODO: make this an implementation of state.output
          # usually the 1st item is the main output
          tensor = state['output'][name][0]
        elif isinstance(state['output'][name], torch.Tensor):
          tensor = state['output'][name]
        else:
          print(f"Unknown type for {name}: {type(state['output']['name'])}")
          continue

        # batch has shape b x t x d -> t x d
        act = tensor[0,self.pos[-1],:].cpu().float().numpy()
        self.acts[name] = np.concatenate([self.acts[name], act], axis=0)

  def collect_activations(self, embed_name=None, layers=None):
    """Collect the activations of the support set (phrase)

    Args:
      embed_name: module name of the embedding layer
      layers: list of module to collect the activation, if None, will use common
        heuristic to get output of main layer blocks
    """
    if embed_name is None:
      candidates = []
      for n, _ in self.insp.model.named_modules():
        if "embed" in n:
          candidates.append(n)
      if len(candidates) != 1:
        raise ValueError(
          f"Cannot determine the name of the embedding module {candidates}. "
          "Please supply"
        )
      embed_name = candidates[0]

    self.embed_name = embed_name

    if layers is None:
      layers = []
      for n, _ in self.insp.model.named_modules():
        if re.match(r".*layers.\d+$", n):
          layers.append(n)

    names = [embed_name] + layers

    if self.get_op is not None:
      self.insp.remove(self.get_op)
    self.get_op = self.insp.add(GetOutput(), name=names)
    with self.insp.ctx() as state:
      acts = {n: [] for n in names}
      for idx, phrase in enumerate(self.phrases):
        token = self.insp.tokenizer.encode(phrase, return_tensors="pt").to(
          self.insp.device
        )
        if self.skip_add_bos[idx]:
          token = token[:,1:]
        _ = self.insp.model(token)

        for name in names:
          if name not in state['output']:
            continue

          if isinstance(state['output'][name], tuple):
            # usually the 1st item is the main output
            tensor = state['output'][name][0]
          elif isinstance(state['output'][name], torch.Tensor):
            tensor = state['output'][name]
          else:
            print(f"Unknown type for {name}: {type(state['output']['name'])}")
            continue

          # batch has shape b x t x d -> t x d
          act = tensor[0,self.pos[idx],:].cpu().float().numpy()
          acts[name].append(act)

    # might be inefficient if we have lots of words, but let's refactor for
    # efficiency when we need, for crude research this is good enough
    self.acts, self.names = {}, []
    for k in names:
      v = acts[k]
      if not v:
        continue
      self.acts[k] = np.concatenate(v, axis=0)
      self.names.append(k)

    # TODO: free CUDA, RAM memory, save the acts to disk if necessary then treat it
    # as memmap
    return

  def build_coordinates(self, coord_system=None):
    """Build the coordinate system

    Args:
      coord_system: name of the layer that will be treated as coordinate system to
        transform the activation. If not provided, each layer construct its own
        coordinate system.
    """
    systems = self.names if coord_system is None else [coord_system]
    for s in systems:
      scaler = StandardScaler()
      x_scaled = scaler.fit_transform(self.acts[s])
      pca = PCA(n_components=self.ndim)
      pca.fit(x_scaled)
      self.pcas[s] = (scaler, pca)

    # use similar left-right, top-down alignment
    positive = None
    for key in self.pcas.keys():
      pca = self.pcas[key][1]
      if positive is None:
        positive = (pca.components_ > 0).astype(int)
        continue
      pos_ = (pca.components_ > 0).astype(int)
      for dim in range(pca.components_.shape[0]):
        if np.abs(positive[dim] - pos_[dim]).sum() > (positive[dim].shape[0] // 2):
          pca.components_[dim] = pca.components_[dim] * -1
      positive = (pca.components_ > 0).astype(int)

  def show_notebook(self, coord_system=None, active_labels=None, layers=None):
    """Visualize in a notebook"""
    if layers is None:
      layers = self.names
    elif isinstance(layers, str):
      layers = [layers]

    cols = min(3, len(layers))
    rows = (len(self.acts) + cols - 1) // cols
    figsize = (5 * cols, 4 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    if active_labels is None:
      labels = self.wlabels
    else:
      if not isinstance(active_labels, (list, set, tuple)):
        active_labels = [active_labels]
      active_labels = set(active_labels)
      labels = [l if l in active_labels else "other_" for l in self.wlabels]

    n_labels = len(set(labels))
    for idx, name in enumerate(layers):
      system = name if coord_system is None else coord_system
      if system not in self.pcas:
        self.build_coordinates(system)
      scaler, pca = self.pcas[system]
      x_scaled = scaler.transform(self.acts[name])
      x_pca = pca.transform(x_scaled)
      sns.scatterplot(
        x=x_pca[:,0], y=x_pca[:,1], hue=labels, size=self.wsizes, style=self.wstyles, ax=axes[idx],
      )
      axes[idx].set_title(name)

      # handle legend
      legend_handles, legend_labels = axes[idx].get_legend_handles_labels()
      axes[idx].legend(legend_handles[:n_labels], legend_labels[:n_labels])

    for idx in range(len(layers), rows * cols):
      axes[idx].remove()

    # title
    fig.suptitle(
      f"{self.insp.model_id} - {coord_system or 'All'}", fontsize=16, fontweight="bold"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98 if rows >= 9 else 0.96))
    return fig

  def search_phrase(self, phrase_regex="", verbose: bool = True):
    """Search for phrases to get phrase and associated index that match"""
    if not phrase_regex:
      return []

    pattern = re.compile(phrase_regex)
    result = []
    c = 0
    for idx, phrase in enumerate(self.phrases):
      if pattern.search(phrase):
        result.append((idx, tuple(range(c, c+len(self.pos[idx])))))
      c += len(self.pos[idx])

    return result

  def delete_phrase(self, idx):
    c = 0
    for i in range(idx):
      c += len(self.pos[i])

    self.phrases.pop(idx)
    self.raw_pos.pop(idx)
    self.labels.pop(idx)
    self.styles.pop(idx)
    self.skip_add_bos.pop(idx)

    pos = self.pos.pop(idx)
    s = c-len(pos)
    self.flattened_tokens = self.flattened_tokens[:s] + self.flattened_tokens[c:]
    self.wlabels = self.wlabels[:s] + self.wlabels[c:]
    self.wstyles = self.wstyles[:s] + self.wstyles[c:]
    to_delete = list(range(s,c))
    for key in self.acts:
      self.acts[key] = np.delete(self.acts[key], obj=to_delete, axis=0)

  def save(self, path):
    ...

  @classmethod
  def load(cls, path):
    ...

# vim: ts=2 sts=2 sw=2 et
