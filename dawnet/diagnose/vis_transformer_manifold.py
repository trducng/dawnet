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
    self.tokens, self.classes = self.prepare_tokens()
    self.get_op = None
    self.embed_name = None

    self._add_bos_token = False
    if hasattr("add_bos_token", self.insp.tokenizer):
      self._add_bos_token = self.insp.tokenizer.add_bos_token

    # raw columns -> might treat this as sqlite to manage relationship
    self.phrases, self.raw_pos, self.labels, self.styles, self.skip_add_bos = [], [], [], [], []

    # processed from insp
    self.pos = [], []   # depend on insp model
    self.acts, self.names = {}, []

    # for visualization
    self.words, self.word_to_phrase = [], []

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

  def build_coordinate(self, embed_name=None, layers=None):
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

    self.names = [embed_name] + layers

    if self.get_op is not None:
      self.insp.remove(self.get_op)
    self.get_op = self.insp.add(GetOutput(), name=self.names)
    with self.insp.ctx() as state:
      acts = {n: [] for n in self.names}
      for idx, phrase in enumerate(self.phrases):
        token = self.insp.tokenizer.encode(phrase, return_tensor="pt").to(
          self.insp.device
        )
        if self.skip_add_bos[idx]:
          token = token[:,1:]
        _ = self.insp.model(token)

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

          # batch has shape b x t x d
          act = tensor[0,self.pos[idx],:].cpu().float().numpy()
          self.acts[name].append(act)

    del state

    # free CUDA, RAM memory
    return out

  def show_notebook(self, figsize=None):

    cols = min(3, len(self.acts))
    rows = (len(self.acts) + cols - 1) // cols
    classes = [i[0] for i in self.classes]

    if figsize is None:
      figsize = (5 * cols, 4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    x = self.acts[self.embed_name]
    # here, each layer will have a different scaler (might not be appropriate to
    # visualize across layers) because it will not be of the same language
    # but we can't expect representation in different layers to share the same scale
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    pca = PCA(n_components=self.ndim)
    x_pca = pca.fit_transform(x_scaled)

    sns.scatterplot(x=x_pca[:,0], y=x_pca[:,1], hue=classes, ax=axes[0])
    axes[0].set_title(self.embed_name)

    for idx, n in enumerate(self.acts.keys()):
      if n == self.embed_name:
        continue

      x = self.acts[n]

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

  def save(self, path):
    ...

  @classmethod
  def load(cls, path):
    ...

if __name__ == "__main__":
  ...

# vim: ts=2 sts=2 sw=2 et
