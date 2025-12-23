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

    # processed from insp
    self.tokens, self.pos = [], []  # depend on insp model
    self.acts, self.names = {}, []

    # coordinates for visualization
    self.pcas = {}
    self.wlabels, self.wstyles = [], []

  def add_phrase(
   self, phrase: str, pos: str | Literal[-1, 0]=-1, label="default", style="o", skip_add_bos=True,
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
    self.skip_add_bos.append(self._add_bos_token and skip_add_bos)

    if isinstance(pos, str):
      str_idx, token_idx = [], []
      idx = phrase.index(pos)
      while idx != -1:
        str_idx.append(idx)
        phrase = phrase[:idx] + phrase[idx+len(pos):]
        idx = phrase.index(pos)
      if not str_idx:
        raise ValueError(f"Cannot find pattern {pos} in phrase {phrase}")

      tokenized = self.insp.tokenizer.tokenize(phrase)
      cidx = 0
      for idx, tok in enumerate(tokenized):
        if not str_idx:
          break
        cidx += len(tok)
        if cidx > str_idx[0]:
          if self._add_bos_token:
            token_idx.append(idx+1)
          else:
            token_idx.append(idx)
          str_idx = str_idx[1:]
      self.tokens.append(self.insp.tokenizer.encode(phrase))
      self.pos.append(token_idx)
    else:
      tokens = self.insp.tokenizer.encode(phrase)
      self.tokens.append(tokens)
      if pos == -1:
        self.pos.append([-1,])
      else:
        if self._add_bos_token:
          self.pos.append(list(range(1, len(tokens)+1)))
        else:
          self.pos.append(list(range(len(tokens))))

  def collect_activations(self, embed_name=None, layers=None):
    """Build the coordinates to visualize

    Output:
      - PCA coordinate for each layer
      - X_transformed for each layer
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

    # construct self.wlabels, self.wstyles
    v = self.acts[self.names[0]]
    for idx in range(len(self.phrases)):
      self.wlabels += [self.labels[idx]] * len(self.pos[idx])
      self.wstyles += [self.styles[idx]] * len(self.pos[idx])

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

  def show_notebook(self, coord_system=None):
    """Visualize in a notebook"""
    cols = min(3, len(self.acts))
    rows = (len(self.acts) + cols - 1) // cols
    figsize = (5 * cols, 4 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, name in enumerate(self.names):
      system = name if coord_system is None else coord_system
      if system not in self.pcas:
        self.build_coordinates(system)
      scaler, pca = self.pcas[system]
      x_scaled = scaler.transform(self.acts[name])
      x_pca = pca.transform(x_scaled)
      sns.scatterplot(
        x=x_pca[:,0], y=x_pca[:,1], hue=self.wlabels, style=self.wstyles, ax=axes[idx]
      )
      axes[idx].set_title(name)

    fig.tight_layout()
    return fig

  def save(self, path):
    ...

  @classmethod
  def load(cls, path):
    ...

# vim: ts=2 sts=2 sw=2 et
