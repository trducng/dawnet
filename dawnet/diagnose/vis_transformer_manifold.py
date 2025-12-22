"""Visualize model computational manifold"""
import re

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from dawnet.op import GetOutput


class ManifoldExperiment:
  def __init__(self, insp, words, ndim=2):
    self.insp = insp
    self.words = words
    self.ndim = ndim
    self.tokens, self.classes = self.prepare_tokens()
    self.pcas, self.names = {}, []
    self.get_op = None
    self.embed_name = None

  def prepare_tokens(self):
    try:
      add_bos_token = self.insp.tokenizer.add_bos_token
    except:
      add_bos_token = False

    tokens, classes = [], []
    for cat, wl in self.words.items():
      we = {}
      for w in wl:
        _tok = self.insp.tokenizer.encode(f' {w}', return_tensors="pt").to(
          self.insp.model.device
        )
        if add_bos_token:
          _tok = _tok[:, -1:]
        _tok = _tok[0]
        if len(_tok.shape) > 1:
          print(w, _tok.shape)
          continue
        tokens.append(_tok)
        classes.append((cat, w))

    tokens = torch.stack(tokens)
    print(f"Constructed tokens: {tokens.shape=}")
    return tokens, classes  # len(classes) == tokens.shape[0]

  def collect(self, embed_name=None, layers=None):
    # build the embedding and create pca for the embedding
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

    self.names = [embed_name] + layers

    if self.get_op is not None:
      self.insp.remove(self.get_op)
    self.get_op = self.insp.add(GetOutput(), name=self.names)
    with self.insp.ctx(detach_state=True) as state:
      out = self.insp.model(self.tokens)

    result = {}
    for name in self.names:
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

      result[name] = tensor.squeeze().cpu().float().numpy()

    self.pcas = result
    del state
    return out

  def show_notebook(self, figsize=None):

    cols = min(3, len(self.pcas))
    rows = (len(self.pcas) + cols - 1) // cols
    classes = [i[0] for i in self.classes]

    if figsize is None:
      figsize = (5 * cols, 4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    x = self.pcas[self.embed_name]
    # here, each layer will have a different scaler (might not be appropriate to
    # visualize across layers) because it will not be of the same language
    # but we can't expect representation in different layers to share the same scale
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    pca = PCA(n_components=self.ndim)
    x_pca = pca.fit_transform(x_scaled)

    sns.scatterplot(x=x_pca[:,0], y=x_pca[:,1], hue=classes, ax=axes[0])
    axes[0].set_title(self.embed_name)

    for idx, n in enumerate(self.pcas.keys()):
      if n == self.embed_name:
        continue

      x = self.pcas[n]

      scaler = StandardScaler()
      x_scaled = scaler.fit_transform(x)
      pca = PCA(n_components=self.ndim)
      x_pca = pca.fit_transform(x_scaled)
      #x_scaled = scaler.transform(x)
      # x_pca = pca.transform(x_scaled)
      sns.scatterplot(x=x_pca[:,0], y=x_pca[:,1], hue=classes, ax=axes[idx])
      axes[idx].set_title(n)

    fig.tight_layout()
    return fig

  def process_special_phrase(self):
    # attach special phrase -> create a phrase object
    pass


if __name__ == "__main__":
  ...

# vim: ts=2 sts=2 sw=2 et
