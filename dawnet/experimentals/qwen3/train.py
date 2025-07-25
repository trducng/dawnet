import torch
import torch.optim as optim
from datasets import load_dataset

from model import get_config, Qwen3Tokenizer, Qwen3Model

ds = load_dataset('stas/openwebtext-10k', cache_dir="downloads/openwebtext-10k")
ds_train = ds['train']
tokenizer = Qwen3Tokenizer(
  tokenizer_file_path="/Users/john/Downloads/tokenizer.json",
  repo_id=None,
  add_generation_prompt=False,
  add_thinking=False
)
test_input = tokenizer.encode(ds['train'][0])


config = {   # mini config
  "vocab_size": 151_936,           # Vocabulary size
  "context_length": 40_960,        # Context length that was used to train the model
  "emb_dim": 1024,                 # Embedding dimension
  "n_heads": 4,                   # Number of attention heads
  "n_layers": 7,                  # Number of layers
  "hidden_dim": 1024,              # Size of the intermediate dimension in FeedForward
  "head_dim": 128,                 # Size of the heads in GQA
  "qk_norm": True,                 # Whether to normalize queries and values in GQA
  "n_kv_groups": 2,                # Key-Value groups for grouped-query attention
  "rope_base": 1_000_000.0,        # The base in RoPE's "theta"
  "dtype": torch.bfloat16,         # Lower-precision dtype to reduce memory usage
}

model = Qwen3Model(config)
model.desc()

def train():
  """Training loop, running flawlessly on single-GPU system"""
  # TODO: prepare dataloader

  # TODO: mixed precision training

  # TODO: optimizer

  # TODO: learning rate scheduler

  # TODO: (nice-to-haves) experiment tracking
  # TODO: (nice-to-haves) weights tracking

  for x, y in dataloader:
    # TODO: device synchronization

    # TODO: calculate loss
    # TODO: backward pass, optimizer step, mixed-precision training step, lr-scheduler step
    pass

# vim: ts=2 sts=2 sw=2 et
