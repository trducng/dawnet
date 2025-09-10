import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from dawnet.inspector import Inspector, Op


class JumpReLUSAE(nn.Module):
  def __init__(self, d_model, d_sae):
    # Note that we initialise these to zeros because we're loading in pre-trained weights.
    # If you want to train your own SAEs then we recommend using blah
    super().__init__()
    self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
    self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
    self.threshold = nn.Parameter(torch.zeros(d_sae))
    self.b_enc = nn.Parameter(torch.zeros(d_sae))
    self.b_dec = nn.Parameter(torch.zeros(d_model))

  def encode(self, input_acts):
    pre_acts = input_acts @ self.W_enc + self.b_enc
    mask = (pre_acts > self.threshold)
    acts = mask * torch.nn.functional.relu(pre_acts)
    return acts

  def decode(self, acts):
    return acts @ self.W_dec + self.b_dec

  def forward(self, acts):
    acts = self.encode(acts)
    recon = self.decode(acts)
    return recon


class SteerOp(Op):
  """Construct the steering op

  The steering logic very depends on:
    - The layer output: which LLM to steer, which layer it is
    - Which SAE used for steering
    - Which steering method (simple additive or orthogonal projection or etc)
  """
  def __init__(self, sae: nn.Module):
    super().__init__()
    self.sae = sae

  def forward(self, inspector, name, module, args, kwargs, output):
    feature_id = inspector._op_params[self.id]["feature_id"]
    strength = inspector._op_params[self.id]["strength"]
    steering_vector = self.sae.W_dec[feature_id]
    output[0][:,-1,:] += steering_vector * strength
    return output

  def run_params(self, feature_id, strength=150):
    return super().run_params(feature_id=feature_id, strength=strength)


if __name__ == "__main__":
  device = torch.device("mps")
  torch.set_grad_enabled(False)

  model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", device_map=device)
  tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

  # prompt = "Would you be able to travel through time using a wormhole?"
  # inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
  # print(inputs)

  # outputs = model.generate(input_ids=inputs, max_new_tokens=50)
  # print(tokenizer.decode(outputs[0]))

  from huggingface_hub import hf_hub_download
  path_to_params = hf_hub_download(
    repo_id="google/gemma-scope-2b-pt-res",
    filename="layer_20/width_16k/average_l0_71/params.npz",
    force_download=False,
  )
  params = np.load(path_to_params)
  pt_params = {k: torch.from_numpy(v).to(device) for k, v in params.items()}
  sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
  sae.load_state_dict(pt_params)
  sae.to(device)

  prompt = "An interesting animal is"
  inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)

  model = Inspector(model)
  op = model.add(SteerOp(sae), name="model.layers.20")

  # Original
  outputs = model.original_model.generate(
    input_ids=inputs, max_new_tokens=64, do_sample=True, temperature=0.5
  )
  print("Original:", tokenizer.decode(outputs[0]))

  # Steered
  with model.ctx(op_params=[op.run_params(feature_id=12082, strength=150)]) as state:
    steered_outputs = model.model.generate(
      input_ids=inputs, max_new_tokens=64, do_sample=True, temperature=0.5
    )
  print("Steered:", tokenizer.decode(steered_outputs[0]))

# vim: ts=2 sts=2 sw=2 et
