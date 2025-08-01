"""
Modified from https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/11_qwen3/standalone-qwen3.ipynb
"""
from typing import Literal
from pathlib import Path

import torch
import torch.nn as nn

from safetensors.torch import load_file


class FeedForward(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
    self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
    self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

  def forward(self, x):
    x_fc1 = self.fc1(x)
    x_fc2 = self.fc2(x)
    x = nn.functional.silu(x_fc1) * x_fc2
    return self.fc3(x)


class RMSNorm(nn.Module):
  def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
    super().__init__()
    self.eps = eps
    self.qwen3_compatible = qwen3_compatible
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

  def forward(self, x):
    input_dtype = x.dtype
    if self.qwen3_compatible:
      x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    norm_x = x * torch.rsqrt(variance + self.eps)
    norm_x = norm_x * self.scale
    if self.shift is not None:
      norm_x = norm_x + self.shift
    return norm_x.to(input_dtype)


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
  assert head_dim % 2 == 0, "Embedding dimension must be even"

  # Compute the inverse frequencies: array from 1 -> 0, shape head_dim//2
  inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

  # Generate the position index
  positions = torch.arange(context_length, dtype=dtype)

  # Compute angles for each vector dimension for each position index
  angles = positions[:,None] * inv_freq[None,:]     # position,head_dim//2

  # Expand angles to match head_dim
  angles = torch.cat([angles, angles], dim=1)   # position,head_dim

  # Precompute sine and cosine
  cos = torch.cos(angles)
  sin = torch.sin(angles)

  return cos, sin


def apply_rope(x, cos, sin):
  # x: batch_size, num_head, seq_length, head_dim
  # cos, sin: max_seq_length, head_dim
  batch_size, num_heads, seq_len, head_dim = x.shape

  # Split x into 1st half and 2nd half
  x1 = x[..., :head_dim // 2]
  x2 = x[..., head_dim // 2:]

  # Adjust sin and cos shapes: 1, 1, seq_len, head_dim
  cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
  sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

  # Apply the rotary transformation
  rotated = torch.cat([-x2, x1], dim=-1)   # bs,heads,seq_len,head_dim
  x_rotated = (x * cos) + (rotated * sin)

  return x_rotated.to(dtype=x.dtype)


class GroupedQueryAttention(nn.Module):
  def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):
    super().__init__()
    assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

    self.num_heads = num_heads
    self.num_kv_groups = num_kv_groups
    self.group_size = num_heads // num_kv_groups

    if head_dim is None:
      assert d_in % num_heads == 0, "d_in must be divisible by num_heads if head_dim is not set"
      head_dim = d_in // num_heads

    self.head_dim = head_dim
    self.d_out = head_dim * num_heads

    self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
    self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
    self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

    self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)
    if qk_norm:
      self.q_norm = RMSNorm(head_dim, eps=1e-6)
      self.k_norm = RMSNorm(head_dim, eps=1e-6)
    else:
      self.q_norm = self.k_norm = None

  def forward(self, x, mask, cos, sin):
    b, num_tokens, _ = x.shape

    # Apply projection
    queries = self.W_query(x)  # b, num_tokens, head_dim * num_heads
    keys = self.W_key(x)   # b, num_tokens, head_dim * num_kv_groups
    values = self.W_value(x)  # b, num_tokens, head_dim * num_kv_groups

    # Reshape
    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
    keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
    values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

    # Optional normalization
    if self.q_norm:
      queries = self.q_norm(queries)
    if self.k_norm:
      keys = self.k_norm(keys)

    # Apply RoPE
    queries = apply_rope(queries, cos, sin)
    keys = apply_rope(keys, cos, sin)

    # Expand K, V to match the number of heads
    keys = keys.repeat_interleave(self.group_size, dim=1)
    values = values.repeat_interleave(self.group_size, dim=1)

    # Attention
    attn_scores = queries @ keys.transpose(2, 3)   # b,num_heads,num_tokens,num_tokens
    attn_scores = attn_scores.masked_fill(mask, -torch.inf)
    attn_weights = torch.softmax(attn_scores / self.head_dim ** 0.5,  dim=-1)

    context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
    return self.out_proj(context)


class TransformerBlock(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.att = GroupedQueryAttention(
      d_in=cfg["emb_dim"],
      num_heads=cfg["n_heads"],
      head_dim=cfg["head_dim"],
      num_kv_groups=cfg["n_kv_groups"],
      qk_norm=cfg["qk_norm"],
      dtype=cfg["dtype"]
    )
    self.ff = FeedForward(cfg)
    self.norm1 = RMSNorm(emb_dim=cfg["emb_dim"], eps=1e-6)
    self.norm2 = RMSNorm(emb_dim=cfg["emb_dim"], eps=1e-6)

  def forward(self, x, mask, cos, sin):
    z = self.norm1(x)
    z = self.att(z, mask, cos, sin)
    x = z + x

    z = self.norm2(x)
    z = self.ff(z)
    x = z + x
    
    return x


class Qwen3Model(nn.Module):
  def __init__(self, cfg, weight_path: str | None = None):
    super().__init__()

    if cfg["dtype"] == "bf16":
      cfg["dtype"] = torch.bfloat16
    else:
      cfg["dtype"] = torch.float16

    self.cfg = cfg
    self._dtype = cfg["dtype"]

    # Main model parameters
    self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=self._dtype)
    self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
    self.final_norm = RMSNorm(emb_dim=cfg["emb_dim"], eps=1e-6)
    self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=self._dtype)

    # Rope utilities
    if cfg["head_dim"] is None:
      head_dim = cfg["emb_dim"] // cfg["n_heads"]
    else:
      head_dim = cfg["head_dim"]
    cos, sin = compute_rope_params(
      head_dim=head_dim,
      theta_base=cfg["rope_base"],
      context_length=cfg["context_length"],
    )
    self.register_buffer("cos", cos, persistent=False)
    self.register_buffer("sin", sin, persistent=False)

    if weight_path:
      weights_dict = load_file(weight_path)
      self.load_weights(weights_dict)

  def forward(self, tokens):
    tok_embeds = self.tok_emb(tokens)
    x = tok_embeds

    num_tokens = x.shape[1]
    mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)

    for block in self.trf_blocks:
      x = block(x, mask, self.cos, self.sin)
    x = self.final_norm(x)
    logits = self.out_head(x.to(self._dtype))
    return logits

  def desc(self):
    total_params = sum(p.numel() for p in self.parameters())
    print(f"Total number of parameters: {total_params:,}")

    # Account for weight tying
    total_params_normalized = total_params - self.tok_emb.weight.numel()
    print(f"\nTotal number of unique parameters: {total_params_normalized:,}")

  @staticmethod
  def default_config(size: Literal["0.6B", "1.7B", "4B", "8B", "14B", "32B"]):
    if size == "0.6B":
        return {
            "vocab_size": 151_936,           # Vocabulary size
            "context_length": 40_960,        # Context length that was used to train the model
            "emb_dim": 1024,                 # Embedding dimension
            "n_heads": 16,                   # Number of attention heads
            "n_layers": 28,                  # Number of layers
            "hidden_dim": 3072,              # Size of the intermediate dimension in FeedForward
            "head_dim": 128,                 # Size of the heads in GQA
            "qk_norm": True,                 # Whether to normalize queries and values in GQA
            "n_kv_groups": 8,                # Key-Value groups for grouped-query attention
            "rope_base": 1_000_000.0,        # The base in RoPE's "theta"
            "dtype": "bf16",         # Lower-precision dtype to reduce memory usage
        }

    elif size == "1.7B":
        return {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "emb_dim": 2048,                 # 2x larger than above
            "n_heads": 16,
            "n_layers": 28,
            "hidden_dim": 6144,              # 2x larger than above
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": "bf16",
        }   

    elif size == "4B":
        return {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "emb_dim": 2560,                 # 25% larger than above
            "n_heads": 32,                   # 2x larger than above
            "n_layers": 36,                  # 29% larger than above
            "hidden_dim": 9728,              # ~3x larger than above
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": "bf16",
        }  

    elif size == "8B":
        return {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "emb_dim": 4096,                 # 60% larger than above
            "n_heads": 32,
            "n_layers": 36,                  # 26% larger than above
            "hidden_dim": 12288,
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": "bf16",
        } 

    elif size == "14B":
        return {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "emb_dim": 5120,                 # 25% larger than above
            "n_heads": 40,                   # 25% larger than above
            "n_layers": 40,                  # 11% larger than above
            "hidden_dim": 17408,             # 42% larger than above
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": "bf16",
        } 

    elif size == "32B":
        return {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "emb_dim": 5120,                
            "n_heads": 64,                   # 60% larger than above
            "n_layers": 64,                  # 60% larger than above
            "hidden_dim": 25600,             # 47% larger than above
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": "bf16",
        } 

    else:
        raise ValueError(f"{size} is not supported.")


  def load_weights(self, params):
    def assign(left, right, tensor_name="unknown"):
      if left.shape != right.shape:
        raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")
      return torch.nn.Parameter(right.clone().detach() if isinstance(right, torch.Tensor) else torch.tensor(right))

    self.tok_emb.weight = assign(
      self.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight"
    )

    for l in range(self.cfg["n_layers"]):
      block = self.trf_blocks[l]
      att = block.att

      # Q, K, V projections
      att.W_query.weight = assign(
        att.W_query.weight,
        params[f"model.layers.{l}.self_attn.q_proj.weight"],
        f"model.layers.{l}.self_attn.q_proj.weight"
      )
      att.W_key.weight = assign(
        att.W_key.weight,
        params[f"model.layers.{l}.self_attn.k_proj.weight"],
        f"model.layers.{l}.self_attn.k_proj.weight"
      )
      att.W_value.weight = assign(
        att.W_value.weight,
        params[f"model.layers.{l}.self_attn.v_proj.weight"],
        f"model.layers.{l}.self_attn.v_proj.weight"
      )

      # Output projection
      att.out_proj.weight = assign(
        att.out_proj.weight,
        params[f"model.layers.{l}.self_attn.o_proj.weight"],
        f"model.layers.{l}.self_attn.o_proj.weight"
      )

      # QK norms
      if hasattr(att, "q_norm") and att.q_norm is not None:
        att.q_norm.scale = assign(
          att.q_norm.scale,
          params[f"model.layers.{l}.self_attn.q_norm.weight"],
          f"model.layers.{l}.self_attn.q_norm.weight"
        )
      if hasattr(att, "k_norm") and att.k_norm is not None:
        att.k_norm.scale = assign(
          att.k_norm.scale,
          params[f"model.layers.{l}.self_attn.k_norm.weight"],
          f"model.layers.{l}.self_attn.k_norm.weight"
        )

      # Attention layernorm
      block.norm1.scale = assign(
        block.norm1.scale,
        params[f"model.layers.{l}.input_layernorm.weight"],
        f"model.layers.{l}.input_layernorm.weight"
      )

      # Feedforward weights
      block.ff.fc1.weight = assign(
        block.ff.fc1.weight,
        params[f"model.layers.{l}.mlp.gate_proj.weight"],
        f"model.layers.{l}.mlp.gate_proj.weight"
      )
      block.ff.fc2.weight = assign(
        block.ff.fc2.weight,
        params[f"model.layers.{l}.mlp.up_proj.weight"],
        f"model.layers.{l}.mlp.up_proj.weight"
      )
      block.ff.fc3.weight = assign(
        block.ff.fc3.weight,
        params[f"model.layers.{l}.mlp.down_proj.weight"],
        f"model.layers.{l}.mlp.down_proj.weight"
      )
      block.norm2.scale = assign(
        block.norm2.scale,
        params[f"model.layers.{l}.post_attention_layernorm.weight"],
        f"model.layers.{l}.post_attention_layernorm.weight"
      )

    # Final normalization and output head
    self.final_norm.scale = assign(
      self.final_norm.scale, params["model.norm.weight"], "model.norm.weight"
    )

    if "lm_head.weight" in params:
      self.out_head.weight = assign(
        self.out_head.weight, params["lm_head.weight"], "lm_head.weight"
      )
    else:
      # Model uses weight tying, hence we reuse the embedding layer weights here
      print("Model uses weight tying.")
      self.out_head.weight = assign(
        self.out_head.weight,
        params["model.embed_tokens.weight"],
        "model.embed_tokens.weight"
      )


from tokenizers import Tokenizer


class Qwen3Tokenizer():
  def __init__(self, tokenizer_file_path="tokenizer.json", repo_id=None, add_generation_prompt=False, add_thinking=False):
    self.tokenizer_file_path = tokenizer_file_path
    self.add_generation_prompt = add_generation_prompt
    self.add_thinking = add_thinking

    tokenizer_file_path_obj = Path(tokenizer_file_path)
    if not tokenizer_file_path_obj.is_file() and repo_id is not None:
      _ = hf_hub_download(
        repo_id=repo_id,
        filename=str(tokenizer_file_path_obj.name),
        local_dir=str(tokenizer_file_path_obj.parent.name)
      )
    self.tokenizer = Tokenizer.from_file(tokenizer_file_path)

  def encode(self, prompt):
    messages = [
      {"role": "user", "content": prompt}
    ]  
    formatted_prompt = self.format_qwen_chat(
      messages,
      add_generation_prompt=self.add_generation_prompt,
      add_thinking=self.add_thinking
    )
    return self.tokenizer.encode(formatted_prompt).ids
              
  def decode(self, token_ids):
    return self.tokenizer.decode(token_ids, skip_special_tokens=False)
  
  @staticmethod
  def format_qwen_chat(messages, add_generation_prompt=False, add_thinking=False):
    prompt = ""
    for msg in messages:
      prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    if add_generation_prompt:
      prompt += "<|im_start|>assistant"
      if not add_thinking:
        prompt += "<|think>\n\n<|/think>\n\n"
      else:
        prompt += "\n"    
    return prompt


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
  # For-loop is the same as before: Get logits, and only focus on last time step
  for _ in range(max_new_tokens):
    try:
      idx_cond = idx[:, -context_size:]
      with torch.no_grad():
        logits = model(idx_cond)
      logits = logits[:, -1, :]

      # Filter logits with top_k sampling
      if top_k is not None:
        # Keep only top_k values
        top_logits, _ = torch.topk(logits, top_k)
        min_val = top_logits[:, -1]
        logits = torch.where(logits < min_val, torch.tensor(-torch.inf).to(logits.device), logits)

      # pply temperature scaling
      if temperature > 0.0:
        logits = logits / temperature

        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

        # Sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

      # Otherwise same as before: get idx of the vocab entry with the highest logits value
      else:
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

      if eos_id is not None and idx_next.item() == eos_id:
        break  # Stop generating early if end-of-sequence token is encountered and eos_id is specified

      # Same as before: append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)
    except KeyboardInterrupt:
      return idx

  return idx



if __name__ == "__main__":
  USE_REASONING_MODEL = False
  CHOOSE_MODEL = "0.6B"
  QWEN3_CONFIG = get_config(CHOOSE_MODEL)

  model = Qwen3Model(QWEN3_CONFIG)
  if torch.cuda.is_available():
    device = torch.device("cuda")
  elif torch.backends.mps.is_available():
    device = torch.device("mps")
  else:
    device = torch.device("cpu")

  model.to(device)

  import json
  import time
  from pathlib import Path
  from huggingface_hub import hf_hub_download, snapshot_download

  weights_dict = load_file("/Users/john/Downloads/model.safetensors")
  load_weights_into_qwen(model, QWEN3_CONFIG, weights_dict)
  model.to(device);

  tokenizer = Qwen3Tokenizer(
    tokenizer_file_path="/Users/john/Downloads/tokenizer.json",
    repo_id=None,
    add_generation_prompt=USE_REASONING_MODEL,
    add_thinking=USE_REASONING_MODEL
  )

  prompt = "Give me a short introduction to large language models."
  input_token_ids = tokenizer.encode(prompt)

  start = time.time()
  output_token_ids = generate(
    model=model,
    idx=torch.tensor(input_token_ids, device=device).unsqueeze(0),
    max_new_tokens=2048,
    context_size=QWEN3_CONFIG["context_length"],
    top_k=1,
    temperature=0.
  )

  print(f"Time: {time.time() - start:.2f} sec")
  output_text = tokenizer.decode(output_token_ids.squeeze(0).tolist())
  print(output_text + "...")

# vim: ts=2 sts=2 sw=2 et
