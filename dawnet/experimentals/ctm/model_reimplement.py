from typing import Literal

import einops
import torch
import torch.nn as nn
from torchvision.models import resnet


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
    decay = torch.arange(start=T, end=0, step=-1)
    decay = self._decay ** decay
    decay = decay.unsqueeze(0).unsqueeze(0).expand(B, 1, T)

    left = post_act_hist[:,self._idx_left,:] * self._decay
    right = post_act_hist[:,self._idx_right,:] * self._decay

    out = torch.bmm(left, right.permute(0, 2, 1))
    return out


class VanillaAttention(nn.Module):
  def __init__(self, feat_shape: int):
    super().__init__()
    self._query = nn.LazyLinear(out_features=feat_shape)

  def forward(self, rep, feat):
    B, S, C = feat.shape

    q = self._query(rep)
    q = q.unsqueeze(-1)    # B,C,1

    qk = torch.bmm(feat, q)   # B,S,1
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

    self._weights = nn.Parameter(torch.empty(size, memory).uniform_(), requires_grad=True)
    self._bias = nn.Parameter(torch.empty(size), requires_grad=True)

  def forward(self, x):
    output = torch.einsum("bsm,sm->bs", x, self._weights)
    output = output + self._bias
    return output


class CTM(nn.Module):
  """Reimplementation of the CTM module

  Args:
    size: the number of neurons in a model (S)
    memory: the maximum amount of pre-activation history kept (M)
    nticks: the loop amount
  """
  def __init__(self, size, memory, nticks, nout, nfeat, **config):
    super().__init__()

    self._config = config
    self._memory = memory
    self._nticks = nticks
    self._size = size

    self._feature_encoder = self.get_feature_encoder()
    self._feat_shape = self._feature_encoder.feat_shape()
    self._synapse = self.get_synapse()
    self._attn = self.get_attention()
    self._nlm = self.get_nlm()
    self._lin_out = nn.LazyLinear(out_features=nout)
    self._lin_act = nn.LazyLinear(out_features=nfeat)

    self._i_post_act = nn.Parameter(torch.empty(size).normal_(), requires_grad=True)
    self._i_pre_act_mem = nn.Parameter(
      torch.empty(size, memory).normal_(), requires_grad=True
    )
    self._i_post_act_mem = nn.Parameter(
      torch.empty(size, nticks).normal_(), requires_grad=True
    )
    self._sync = self.get_synchronization()

  def get_feature_encoder(self) -> FeatureEncoder:
    """Encode the feature"""
    return ResnetFeatureEncoder()

  def get_synapse(self) -> nn.Module:
    """Get the synapse model"""
    # TODO: the default should be a U-Net architecture
    return nn.LazyLinear(out_features=self._size)

  def get_attention(self) -> nn.Module:
    """Get attention module, modulate where to look at in the data in each tick

    The attention takes in:
      - rep: internal representation: (B, chosen)
      - feat: the input data: (B,HxW,C)
    Spits out:
      - the modified `feat` (same shape as feat)
    """
    attn = VanillaAttention(feat_shape=self._feat_shape)
    return attn

  def get_nlm(self) -> nn.Module:
    """Get the neuron-level model

    The nlm module takes in:
      - pre_act_mem: the activation history (B,S,M)
    Spits out:
      - post_act: the post activation value (B,S)
    """
    return NeuronLevelModel(size=self._size, memory=self._memory)

  def get_synchronization(self) -> nn.Module:
    """Get the synchronization module

    The synchronization module takes in:
      - post_activation_history (B,D,current_tick)
    Spits out:
      - representation (B,chosen)
    """
    sync = Sync(chosen=self._size // 3, nneurons=self._size, decay=0.7)
    return sync

  def forward(self, x):
    B = x.size(0)   # B,C,H,W
    feat = self._feature_encoder(x)  # B,HxW,C

    post_act = self._i_post_act.unsqueeze(0).expand(B, *self._i_post_act.shape)
    pre_act_mem = self._i_pre_act_mem.unsqueeze(0).expand(B, *self._i_pre_act_mem.shape)
    post_act_mem = self._i_post_act_mem.unsqueeze(0).expand(
      B, *self._i_post_act_mem.shape
    )

    rep_action = None
    post_act_mem = None
    outputs = []
    for t in range(self._nticks):
      # attend to the desired region from the input
      if rep_action is not None:
        modified_feat = self._attn(rep_action, feat)
        syn_input = torch.concat([post_act, modified_feat])
      else:
        syn_input = torch.concat([post_act, feat])

      # go through synapse model
      pre_act = self._synapse(post_act)

      # neuron level module
      pre_act_mem = torch.concat([pre_act_mem[:,:,-1], pre_act], dim=-1)
      post_act = self._nlm(pre_act_mem)   # B,S
      if not isinstance(post_act_mem, torch.Tensor):
        post_act_mem = post_act.unsqueeze(-1)
      else:
        post_act_mem = torch.concat([post_act_mem, post_act.unsqueeze(-1)], dim=-1)

      # synchronization
      rep = self._sync(post_act_mem)  # B,chosen

      # get output
      out = self._lin_out(rep)
      outputs.append(out)

      # get action
      rep_action = self._lin_act(rep)

    return outputs


if __name__ == "__main__":
  feat_shape = 64
  chosen = 256
  B, S, C = 8, 24, feat_shape

  attn = VanillaAttention(feat_shape=feat_shape)
  rep_ = torch.Tensor(size=(B, chosen)).uniform_()
  feat_ = torch.Tensor(size=(B, S, C)).uniform_()
  o_ = attn(rep_, feat_)
  print(o_.shape)

# vim: ts=2 sts=2 sw=2 et
