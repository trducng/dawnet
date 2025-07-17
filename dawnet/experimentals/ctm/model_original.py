import math

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision.models import resnet

from dawnet.utils.typing_ import Tensor


class Sync(nn.Module):
  """Synchronization module, as introduced in the CTM paper

  Args:
    chosen: the number of neuron pairs
  """
  def __init__(self, chosen: int, nneurons: int, decay: float=0.7, **config):
    super().__init__()
    self._config = config
    self._chosen = chosen
    self._nneurons = nneurons

    self._decay = nn.Parameter(torch.Tensor([decay]).squeeze(), requires_grad=True)

    self._idx_left = torch.randint(low=0, high=nneurons, size=(chosen,))
    self._idx_right = torch.randint(low=0, high=nneurons, size=(chosen,))

  def forward(self, post_act_hist):
    B, M, T = post_act_hist.shape   # batch, n_neurons, time
    decay = torch.arange(start=T, end=0, step=-1, device=self._decay.device)
    decay = self._decay ** decay
    decay = decay.unsqueeze(0).unsqueeze(0).expand(B, 1, T)

    left = post_act_hist[:,self._idx_left,:] * self._decay
    right = post_act_hist[:,self._idx_right,:] * self._decay

    out = torch.einsum("bct,bct->bc", left, right)
    return out


class Synchronization(nn.Module):
  """Synchronizing the post activation activity, used by CTM

  Args:
    n_sync: the number of neurons to synchronize with each other
    d_model: the CTM model size
  """

  def __init__(self, n_sync: int, d_model: int, **config):
    self._config = config
    self._n_sync: int = n_sync
    self._d_model = d_model
    self._r_decay = nn.Parameter(torch.zeros(self.n_sync), requires_grad=True)
    self._left = nn.Buffer(
      torch.from_numpy(np.random.choice(np.arange(d_model), size=self.n_sync))
    )
    self._right = nn.Buffer(
      torch.from_numpy(np.random.choice(np.arange(d_model), size=self.n_sync))
    )

  @property
  def n_sync(self) -> int:
    return self._n_sync

  def forward(self, post_act, current_tick: int = 0):
    """Synchronize post activation, used by CTM

    Args:
      post_act [B,d_model]: the post activation from CTM
      current_tick: the iteration within CTM thinking

    Returns:
      [B,n_sync]: synchronized values
    """
    if current_tick == 0:
      self._decay_alpha, self._decay_beta = None, None
    left, right = post_act[:,self._left], post_act[:,self._right]
    pairwise_product = left * right
    if self._decay_alpha is None or self._decay_beta is None:
      # Prepare decay rate
      self._r_decay.data = torch.clamp(self._r_decay, 0, 15)
      self._r = torch.exp(-self._r_decay).unsqueeze(0).repeat(post_act.size(0), 1)

      # Prepare moving average
      self._decay_alpha = pairwise_product
      self._decay_beta = torch.ones_like(pairwise_product)
    else:
      self._decay_alpha = self._r * self._decay_alpha + pairwise_product
      self._decay_beta = self._r * self._decay_beta + 1

    out = self._decay_alpha / (torch.sqrt(self._decay_beta))
    return out


class FirstSynchronization(Synchronization):
  ...


class LastSynchronization(Synchronization):
  ...


class RandomSynchronization(Synchronization):
  ...


class VanillaAttention(nn.Module):
  def __init__(self, feat_shape: int):
    super().__init__()
    self._query = nn.LazyLinear(out_features=feat_shape)

  def forward(self, rep, feat: torch.Tensor) -> torch.Tensor:
    B, S, C = feat.shape

    q = self._query(rep)
    q = q.unsqueeze(-1)    # B,C,1
    # print(f"{q.mean()=}")

    qk = torch.bmm(feat, q)   # B,S,1
    qk = f.softmax(qk, dim=1)
    # print(f"{qk.mean()=}")
    v = feat * qk    # B,S,C
    return v


class FeatureEncoder(nn.Module):
  def feat_shape(self):
    raise NotImplementedError()


class ResnetFeatureEncoder(FeatureEncoder):
  def __init__(self):
    super().__init__()
    def _forward_impl(self, x):
      x = self.conv1(x)
      x = self.bn1(x)
      x = self.relu(x)
      x = self.maxpool(x)
      x = self.layer1(x)
      return x

    self._res = resnet.resnet18()
    self._res._forward_impl = _forward_impl.__get__(self._res)

  def forward(self, x):
    return self._res(x)

  def feat_shape(self):
    return 64


class NeuronLevelModel(nn.Module):
  def __init__(self, size: int, memory: int):
    super().__init__()
    self._size = size
    self._memory = memory

    self._weights = nn.Parameter(torch.empty(size, memory).normal_(), requires_grad=True)
    self._bias = nn.Parameter(torch.empty(size).normal_(), requires_grad=True)

  def forward(self, x):
    output = torch.einsum("bsm,sm->bs", x, self._weights)
    output = output + self._bias
    output = f.relu(output)
    return output


class CTM(nn.Module):
  """Reimplementation of the CTM module

  Args:
    size: the number of neurons in a model (S)
    memory: the maximum amount of pre-activation history kept (M)
    nticks: the loop amount
    nout: the output shape
  """
  def __init__(self, d_model, memory, nticks, nout, **config):
    super().__init__()

    self._config = config    # contain sample input
    self._memory = memory
    self._nticks = nticks
    self._d_model = d_model

    # Components
    self._feature_encoder = self.get_feature_encoder()
    self._synapse = self.get_synapse()
    self._attn = self.get_attention()
    self._nlm = self.get_nlm()
    self._sync_act, self._sync_out = self.get_synchronization()

    self._lin_out = nn.LazyLinear(out_features=nout)
    self._lin_act = nn.LazyLinear(out_features=self._feature_encoder.feat_shape())

    # Initial states, maybe added to the corresponding components
    self._i_post_act = nn.Parameter(
      torch.zeros((d_model)).uniform_(-math.sqrt(1/(d_model)), math.sqrt(1/(d_model))),
      requires_grad=True
    )
    self._i_pre_act_mem = nn.Parameter(
      torch.empty(d_model, memory).normal_(), requires_grad=True
    )


  def get_feature_encoder(self) -> FeatureEncoder:
    """Encode the feature"""
    return ResnetFeatureEncoder()

  def get_synapse(self) -> nn.Module:
    """Get the synapse model"""
    # TODO: the default should be a U-Net architecture
    return nn.Sequential(
      nn.LazyLinear(out_features=self._d_model),
      nn.ReLU(),
    )

  def get_attention(self) -> nn.Module:
    """Get attention module, modulate where to look at in the data in each tick

    The attention takes in:
      - rep: internal representation: (B, chosen)
      - feat: the input data: (B,HxW,C)
    Spits out:
      - the modified `feat` (same shape as feat)
    """
    attn = VanillaAttention(feat_shape=self._feature_encoder.feat_shape())
    return attn

  def get_nlm(self) -> nn.Module:
    """Get the neuron-level model

    The nlm module takes in:
      - pre_act_mem: the activation history (B,S,M)
    Spits out:
      - post_act: the post activation value (B,S)
    """
    return NeuronLevelModel(size=self._d_model, memory=self._memory)

  def get_synchronization(self) -> tuple[nn.Module, nn.Module]:
    """Get the synchronization module

    The synchronization module takes in:
      - post_activation_history (B,D,current_tick)
    Spits out:
      - representation (B,chosen)
    """
    act = Synchronization(n_sync=self._config["n_sync_action"], d_model=self._d_model)
    out = Synchronization(n_sync=self._config["n_sync_out"], d_model=self._d_model)
    return act, out

  def run_features(self, x):
    """Convert the input into attention-friendly feature

    Args:
      x: image-type of shape B,3or1,H,W

    Returns:
      B,HxW,C: the attention friendly feature
    """
    feat = self._feature_encoder(x)  # B,C,H,W
    feat = feat.permute(0,2,3,1).contiguous().view(x.size(0),-1,feat.size(1)) # B,HxW,C
    return feat

  def run_synapse(self, sync, feat, post_act):
    """Run the synapse to construct pre-activation

    Args:
      sync (B,sync_act): the synchronization state for selecting action
      feat (B,T,C): the input feature

    Returns:
      (B,d_model): the pre-activation values
    """
    q = self._lin_act(sync).unsqueeze(1)    # B,1,C
    attn_out, attn_weights = self._attn(   # B,C   # B,heads,1,T
      q, feat, feat, average_attn_weights=True, need_weights=True
    )
    attn_out = attn_out.squeeze(1)
    syn_input = torch.concat([post_act, attn_out], dim=-1)
    pre_act = self._synapse(syn_input)
    return syn_input

  def forward(self, x, state=None, return_state=False):
    """CTM go through ticks"""
    B = x.size(0)   # B,C,H,W
    feat = self.run_features(x)  # B,HxW,C

    post_act = self._i_post_act.unsqueeze(0).expand(B, *self._i_post_act.shape)
    pre_act_mem = self._i_pre_act_mem.unsqueeze(0).expand(B, *self._i_pre_act_mem.shape)

    outputs = []

    for self._current_tick in range(self._nticks):
      # synchronization for input (B,sync_act)
      sync_act = self._sync_act(post_act, current_tick=self._current_tick)

      # go through synapse module
      pre_act = self.run_synapse(sync_act, feat, post_act)

      # pre-activation memory
      pre_act_mem = torch.concat([pre_act_mem[:,:,1:], pre_act.unsqueeze(-1)], dim=-1)
      post_act = self._nlm(pre_act_mem)   # B,S

      # synchronization for output (B,sync_out)
      sync_out = self._sync_out(post_act, current_tick=self._current_tick)

      # get output (B,out)
      out = self._lin_out(sync_out)
      outputs.append(out)

    return outputs

# vim: ts=2 sts=2 sw=2 et
