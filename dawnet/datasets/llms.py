"""Utilities for LLM-related datasets"""
import random
from datasets import load_dataset


# data = load_dataset("HuggingFaceFW/clean-wikipedia", name="en", split="train")


def get_snippets_from_text(
  token: int | str, text: str, surround: int, tokenizer, return_int=True
) -> list[list[int]] | list[str]:
  """Pinpoint the part in the text that contain a specific token, with context"""
  if isinstance(token, str):
    tokenized = tokenizer.encode(token)
    if len(tokenized) != 1:
      raise ValueError(f"{token=} requires multiple tokens to represent: {tokenized}")

  tokens = tokenizer.encode(text)
  result = []
  for idx, tok in enumerate(tokens):
    if tok == token:
      s = tokens[max(0, idx - surround) : min(len(tokens), idx + surround)]
      if return_int:
        s = tokenizer.decode(s)
      result.append(s)
  return result



def get_samples(
  token,
  n_samples,
  corpus: list[list[int]],
  tok2docids: dict[int, list[int]],
  tokenizer=None,
) -> tuple[list[list[int]], list[list[int]]]:
  """Get n samples from `corpus` that contain `token`

  Args:
    token: the token (in int or str)
    n_samples: the number of samples to retrieve
    corpus: the corpus that contains list of list of tokens, each inner list is a doc
    tok2docids: mapping from token to the document index in corpus that contain it
    tokenizer: the tokenizer

  Returns:
    list[list[int]]: list of samples, each list is a list of tokens
    list[list[int]]: corresponding location of the `token` in the previous samples
  """
  if isinstance(token, str):
    if tokenizer is None:
      raise ValueError("must supply tokenizer of `token` is not int")
    token_id = tokenizer.encode(token)
    if len(token_id) != 1:
      raise ValueError(f"{token} is encoded to {token_id}")
    token_id = token_id[0]
  else:
    token_id = token
  if token_id not in tok2docids:
    raise ValueError(f"Unknown {token_id=}")
  if n_samples > len(tok2docids[token_id]):
    raise ValueError(f"Not enough {n_samples=} ({len(tok2docids)=})")
  idx = list(range(n_samples))
  random.shuffle(idx)
  samples, locs = [], []
  for each in idx:
    each_loc = []
    selected = corpus[tok2docids[token_id][each]]
    samples.append(selected)
    for _idx, each in enumerate(selected):
      if each == token_id:
        each_loc.append(_idx)
    locs.append(each_loc)
  return samples, locs

# vim: ts=2 sts=2 sw=2 et
