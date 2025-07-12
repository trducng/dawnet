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
    decay = torch.arange(start=T, end=0, step=-1, device=self._decay.device)
    decay = self._decay ** decay
    decay = decay.unsqueeze(0).unsqueeze(0).expand(B, 1, T)

    left = post_act_hist[:,self._idx_left,:] * self._decay
    right = post_act_hist[:,self._idx_right,:] * self._decay

    out = torch.einsum("bct,bct->bc", left, right)
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
  def __init__(self, size, memory, nticks, nout, **config):
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
    self._lin_act = nn.LazyLinear(out_features=1)

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
    feat = self._feature_encoder(x)  # B,C,H,W
    feat = feat.permute(0,2,3,1).contiguous().view(B,-1,feat.shape[1]) # B,HxW,C

    post_act = self._i_post_act.unsqueeze(0).expand(B, *self._i_post_act.shape)
    pre_act_mem = self._i_pre_act_mem.unsqueeze(0).expand(B, *self._i_pre_act_mem.shape)
    post_act_mem = self._i_post_act_mem.unsqueeze(0).expand(
      B, *self._i_post_act_mem.shape
    )

    rep = None
    post_act_mem = None
    outputs = []
    for t in range(self._nticks):
      # attend to the desired region from the input
      if rep is not None:
        modified_feat = self._attn(rep, feat)
      else:
        modified_feat = feat

      modified_feat = self._lin_act(modified_feat).squeeze()
      syn_input = torch.concat([post_act, modified_feat], dim=-1)

      # go through synapse model
      pre_act = self._synapse(syn_input)

      # neuron level module
      pre_act_mem = torch.concat([pre_act_mem[:,:,:-1], pre_act.unsqueeze(-1)], dim=-1)
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

    return outputs

if __name__ == "__main__":
  import random
  import torch.optim as optim
  from torchvision import datasets, transforms
  device = torch.device("mps")
  transform = transforms.Compose(
    [
      # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
      # transforms.RandomRotation(10),
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,)),
    ]
  )
  train_data = datasets.MNIST(
    root="/Users/john/dawnet/temp/data", train=True, download=True, transform=transform
  )
  test_data = datasets.MNIST(
    root="/Users/john/dawnet/temp/data", train=False, download=True, transform=transform
  )

  images = []
  for i in range(5):
    image, _ = train_data[i]
    images.append(image)

  images = torch.stack(images)

  print(f"{images.shape=}")
  model = CTM(size=512, memory=15, nticks=10, nout=10)
  model.train()
  # init_mem = model.memory.clone().detach()
  model = model.to(device=device)

  loss_obj = nn.CrossEntropyLoss()
  optimizer = optim.Adam(params=model.parameters())
  trainloader = torch.utils.data.DataLoader(
    train_data, batch_size=256, shuffle=True, num_workers=1
  )
  testloader = torch.utils.data.DataLoader(
    test_data, batch_size=256, shuffle=False, num_workers=1, drop_last=False
  )
  count = 0
  for x, y in trainloader:
    x = x.expand(-1,3,-1,-1)
    x, y = x.to(device), y.to(device)
    count += 1
    out = model(x)

    out_idx = random.randint(0, len(out)-1)
    loss = loss_obj(out[out_idx], y)

    # for idx, each in enumerate(out):
    #   if idx == 0:
    #     loss = loss_obj(each, y)
    #   else:
    #     loss += loss_obj(each, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if count % 100 == 0:
      print(f"{count=}, {loss.item()=}")

  correct = 0
  total = 0
  model.eval()
  track = None
  # before = model.active_state.clone().detach()
  # before_memory = model.memory.clone().detach()
  # before_w = model.w.clone().detach()
  # before_b = model.b.clone().detach()
  run1_inputs, run1_labels, run1_outputs, run1_preds = [], [], [], []
  with torch.no_grad():
    for _i1, (images, labels) in enumerate(testloader):
      images = images.expand(-1,3,-1,-1)
      if _i1 == 0:
        track = (images, labels)
      images, labels = images.to(device), labels.to(device)
      out = model(images)

      out_idx = random.randint(0, len(out)-1)
      outputs = out[out_idx]

      # outputs = model(images)[-1]
      _, predicted = torch.max(outputs.data, 1)
      run1_inputs.append(images)
      run1_labels.append(labels)
      run1_outputs.append(outputs)
      run1_preds.append(predicted)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  # run1_inputs = [each for sublist in run1_inputs for each in sublist]
  # run1_labels = [each for sublist in run1_labels for each in sublist]
  # run1_outputs = [each for sublist in run1_outputs for each in sublist]
  # run1_preds = [each for sublist in run1_preds for each in sublist]

  accuracy = 100 * correct / total
  # print(f"{correct=}, {total=}")
  print(f"Accuracy: {accuracy:.2f}%")
  # post_mem = model.memory.clone().detach().cpu()
  # print((post_mem - init_mem).sum())

  track2 = None
  run2_inputs, run2_labels, run2_outputs, run2_preds = [], [], [], []
  for _ntick in range(1, 20):
    correct, total = 0, 0
    model._nticks = _ntick
    with torch.no_grad():
      for _i1, (images, labels) in enumerate(testloader):
        images = images.expand(-1,3,-1,-1)
        if _i1 == 0:
          track2 = (images, labels)
        images, labels = images.to(device), labels.to(device)
        out = model(images)

        out_idx = random.randint(0, len(out)-1)
        outputs = out[out_idx]

        # outputs = out[-1]
        _, predicted = torch.max(outputs.data, 1)
        run2_inputs.append(images)
        run2_labels.append(labels)
        run2_outputs.append(outputs)
        run2_preds.append(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    # print(f"{correct=}, {total=}")
    print(f"{_ntick=} Accuracy: {accuracy:.2f}%")
  # after = model.active_state.clone().detach()
  # after_memory = model.memory.clone().detach()
  # after_w = model.w.clone().detach()
  # after_b = model.b.clone().detach()

  # run2_inputs = [each for sublist in run2_inputs for each in sublist]
  # run2_labels = [each for sublist in run2_labels for each in sublist]
  # run2_outputs = [each for sublist in run2_outputs for each in sublist]
  # run2_preds = [each for sublist in run2_preds for each in sublist]

# vim: ts=2 sts=2 sw=2 et
