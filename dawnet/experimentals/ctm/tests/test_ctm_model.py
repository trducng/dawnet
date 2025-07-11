import torch
from dawnet.experimentals.ctm.model_reimplement import (
  NeuronLevelModel, ResnetFeatureEncoder, Sync, VanillaAttention
)


def test_shape_sync():
  chosen = 256
  nneurons = 512
  decay = 0.7

  B, M, T = 8, nneurons, 12

  sync = Sync(chosen=chosen, nneurons=nneurons, decay=decay) 
  i_ = torch.Tensor(size=(B, M, T)).uniform_()
  o_ = sync(i_)
  assert o_.shape == (B, chosen, chosen)


def test_shape_vanilla_attention():
  feat_shape = 64
  chosen = 256
  B, S, C = 8, 24, feat_shape

  attn = VanillaAttention(feat_shape=feat_shape)
  rep_ = torch.Tensor(size=(B, chosen)).uniform_()
  feat_ = torch.Tensor(size=(B, S, C)).uniform_()
  o_ = attn(rep_, feat_)
  assert o_.shape == (B, S, C)


def test_shape_resnet_feature_encoder():
  res = ResnetFeatureEncoder()

  B, C, H, W = 8, 3, 128, 128
  x = torch.empty((B, C, H, W)).uniform_()
  y = res(x)
  assert y.shape == (B, 64, 32, 32)


def test_shape_nlm():
  B, S, M = 8, 512, 15
  model = NeuronLevelModel(size=S, memory=M)
  x = torch.empty((B,S,M)).uniform_()
  out = model(x)
  assert out.shape == (B,S)


# vim: ts=2 sts=2 sw=2 et
