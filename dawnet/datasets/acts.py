import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import trange, tqdm
from dawnet.datasets.llms import get_samples

from dawnet.inspector import LLMInspector
from dawnet import Inspector, op



def get_pairwise_cosine(acts1, acts2=None):
  """Get the pairwise cosine similarity between the activations

  Args:
    acts1: has shape n_tokens x dim
    acts2: has shape n_tokens x dim

  Returns:
    a tensor with shape n_tokens x n_tokens where each entry is the corresponding
      cosine similarity between the tokens
  """
  if acts2 is None:
    acts2 = acts1

  numerator = acts1 @ acts2.T
  norms1 = acts1.norm(p=2, dim=1).unsqueeze(1)
  norms2 = acts2.norm(p=2, dim=1).unsqueeze(0)
  denominator = torch.clip(norms1 @ norms2, min=1e-8)
  return numerator / denominator


def get_activation_scaler(select_tokens, batch_size=64, n_samples=1000):
  """
  Assumption: the activations are represented in Euclidean space. The activations might be better represented
  in polar coordinates, but that is for later to discover.

  Let's scale from -1 to 1, by finding the max absolute value.
  """
  avg = None
  max_vals = None
  with insp.ctx() as state:
    for idx in trange(0, n_samples, batch_size):
      inputs_ = torch.tensor(select_tokens[idx:idx+batch_size]).to("mps")
      _ = insp.model(inputs_)
      acts = state["output"]['model.layers.25'][0]
      reduced = torch.abs(acts).view(-1, 1024).max(dim=0).values
      max_vals = reduced if max_vals is None else torch.maximum(max_vals, reduced)
  return max_vals


class LLMActivationDataset(Dataset):
  """
  Need an approach to:
    - get the text from dataset
    - get the activation from the model (a layer might return more than just an
      activation)
    - getting the activation is expensive, and might be good to prefetch
    - can get the activation that is cached from disk
    - can get the activation from other machine:
      - stream the pre-calculated activation from a server
      - have one ore more dedicated machines to calculate the activations
  """
  def __init__(
    self, insp: LLMInspector, hf: dict=None, scale_act: bool=True, ctx_len=1024
  ):
    self.data = load_dataset(**hf)
    self.insp = insp
    self._ready = False

  def tokenize(self, idxs: list) -> list[list[int]]:
    if isinstance(idxs, int):
      idxs = list[idxs]
    for idx in idxs:
      tokenized

  def prefetch(self, id):
    ...

  def __len__(self):
    return len(self.data)

  @classmethod
  def load(cls, path):
    ...

  def to_dict(self):
    """Load the state to dict"""
    ...

  def save(self, path):
    ...


class TokenizeDataset:
  def __init__(self):
    self.selected_tokens = None
    self.selected_idx = None
    self.doc_appearances = None
    self.doc_appearances_counter = None

  def build(self, hf: dict, tokenizer, total_tokens=int(1e9), context_size=1024):
    data = load_dataset(**hf)

    selected_idx = []
    selected_tokens = []

    n_items = int(total_tokens / context_size)
    prob = n_items / len(data)

    print(f"{n_items=}, {prob=}")
    prob = min(1.0, prob * 1.5)
    if n_items > len(data):
      print(f"{n_items=} > {len(data)=}. Setting n_items to {len(data)}")
      n_items = len(data)

    with tqdm(total=n_items) as pbar:
      n = 0
      idx = 0
      while n < n_items:
        # if random.random() > prob:
        #   idx += 1
        #   continue

        tokenized_text = tokenizer.encode(data[idx]["text"])
        if len(tokenized_text) < context_size:
          idx += 1
          continue

        selected_tokens.append(tokenized_text[:context_size])
        selected_idx.append(idx)
        idx += 1
        n += 1
        pbar.update(1)

    print(f"{len(selected_idx)=}, {len(selected_tokens)=}")
    if len(selected_idx) == n_items:
      print(f"[OK] {n_items=} == {len(selected_idx)=}")
    else:
      print(f"Expected {n_items=} but have {len(selected_idx)=}")

    if len(selected_tokens) == n_items:
      print(f"[OK] {n_items=} == {len(selected_tokens)=}")
    else:
      print(f"Expected {n_items=} but have {len(selected_tokens)=}")

    for _i, (_idx, _s) in enumerate(zip(selected_idx, selected_tokens)):
      if len(_s) != context_size:
        raise ValueError(f"[WRONG] item {_i} (original idx {_idx}) has length {len(_s)} != {context_size=}")

    from collections import defaultdict

    doc_appearances = defaultdict(list)    # idx of selected_tokens that a token appear
    doc_appearances_counter = defaultdict(int)
    for idx in trange(len(selected_tokens)):
      _tokens = set(selected_tokens[idx])
      for _token in _tokens:
        doc_appearances[_token].append(idx)
        doc_appearances_counter[_token] += 1

    self.selected_tokens = selected_tokens
    self.selected_idx = selected_idx
    self.doc_appearances = doc_appearances
    self.doc_appearances_counter = doc_appearances_counter

  def save(self, path):
    if self.selected_idx is None:
      return

    path = Path(path)
    with (path / "selected_idx.pkl").open("wb") as f:
      pickle.dump(self.selected_idx, f)
    with (path / "selected_tokens.pkl").open("wb") as f:
      pickle.dump(self.selected_tokens, f)
    with (path / "doc_appearances.pkl").open("wb") as f:
      pickle.dump(self.doc_appearances, f)
    with (path / "doc_appearances_counter.pkl").open("wb") as f:
      pickle.dump(self.doc_appearances_counter, f)

  @classmethod
  def load(cls, path):
    data = TokenizeDataset()
    path = Path(path)
    with (path / "selected_idx.pkl").open("rb") as f:
      data.selected_idx = pickle.load(f)
    with (path / "selected_tokens.pkl").open("rb") as f:
      data.selected_tokens = pickle.load(f)
    with (path / "doc_appearances.pkl").open("rb") as f:
      data.doc_appearances = pickle.load(f)
    with (path / "doc_appearances_counter.pkl").open("rb") as f:
      data.doc_appearances_counter = pickle.load(f)
    return data

  def __len__(self):
    if self.selected_tokens is None:
      raise AttributeError("Tokenize dataset is not initialized")
    return len(self.selected_tokens)

  def __getitem__(self, idx):
    if self.selected_tokens is None:
      raise AttributeError("Tokenize dataset is not initialized")
    return self.selected_tokens[idx]

class GetActivation:
  """Central point to 
  """

  @dataclass
  class State():
    idx: int = 0

  @dataclass
  class Config():
    # Model
    insp_layer: str | list[str]
    insp_model: Any = None
    insp_getter: Callable | str | int | None = None
    # Data
    data: TokenizeDataset | None = None
    local_tokenized_data: str = ""
    # Data access
    batch_size: int=1

  def __init__(self, cfg: "GetActivation.Config"=None):
    self._cfg = cfg
    self._s = self.State()

    # initialize insp
    if cfg.insp_layer is None:
      raise ValueError(f"Must supply `insp_layer`")

    self._insp: Inspector
    if cfg.insp_model:
      if isinstance(cfg.insp_model, Inspector):
        self._insp = cfg.insp_model
      else:
        self._insp = Inspector(cfg.insp_model)
      self._insp.add(op.GetOutput(output_getter=cfg.insp_getter), cfg.insp_layer)
    else:
      raise ValueError("Must supply either insp or insp_model")

    # initialize data
    if cfg.data:
      self._data: TokenizeDataset = cfg.data
    else:
      self._data = TokenizeDataset.load(cfg.local_tokenized_data)

  def load_config(self):
    ...

  def save_config(self):
    ...

  def local(self, _path, **cfg):      # path is fsspec or sth like mongo://...
    ...

  def background_process(self, path):
    """Run the work in a background process, return the process handler"""
    # setup the config space
    # setup the input queue
    # setup the output queue
    # wrap the `execute` to read from input queue and store result to output queue
    ...

  @torch.no_grad
  def execute(self):
    # programed to maintain internal state in `self`
    # the config and input is very similar, the difference is:
    # input comes from another automated process
    # config comes from human, but anything coming from human can also come from
    # automated process
    with self._insp.ctx() as state:
      samples = self._data[self._s.idx:self._s.idx+self._cfg.batch_size]
      inputs_ = torch.tensor(samples).to(self._insp.model.device)
      _ = self._insp(inputs_)
      if isinstance(self._cfg.insp_layer, str):
        acts = state["output"][self._cfg.insp_layer]
        acts = acts.view(-1, acts.shape[-1])
      else:
        acts = [state["output"][layer] for layer in self._cfg.insp_layer]
        acts = [each.view(-1, each.shape[-1]) for each in acts]

      self._s.idx = (self._s.idx + self._cfg.batch_size) % len(self._data)
      return acts


if __name__ == "__main__":
  """Dataset will have several aspect

  To build (prepare):
    - step 1: config step 1
    - step 2: config step 2
    - orchestrated by pipeline executor, similar to datatrove
  To save:
    - for saving the final artifact, so that the next time, we don't have to do much
  To load & stream (from locally and from huggingface):
    - path and some other stuff
  To upload to huggingface:
    - can construct huggingface dataset

  The idea here is:
    - dataloader can have big steps requiring pre-processing
    - dataloader can perform big steps on some data instances behind the scene,
      and the dataloader can be run as soon as the first few instances are ready
  """
  # DataLoader(
  #   pipeline = [
  #     HFReader(...),   # -> can skip if possible
  #     Tokenization(...),  # -> can stream from disk if necessary
  #     GetActivations(...), #  -> cache if possible
  #   ],
  #   path="..."
  # )
  from transformers import AutoTokenizer, AutoModelForCausalLM
  # get the data
  model_name = "Qwen/Qwen3-0.6B-Base"
  # tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(model_name, device_map="mps")

  activation_loader = GetActivation(
    cfg=GetActivation.Config(
      insp_layer="model.layers.25",
      insp_model=model,
      insp_getter=0,
      local_tokenized_data="/Users/john/dawnet/experiments/temp/_temp_prepare_data",
    ),
  )


# vim: ts=2 sts=2 sw=2 et
