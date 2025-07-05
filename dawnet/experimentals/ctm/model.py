import random
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torchvision import datasets, transforms


class Model(nn.Module):
  def __init__(self, memory, nticks):
    super().__init__()
    self.feature_extractor = nn.Conv2d(
      in_channels=1,
      out_channels=64,
      kernel_size=2,
      stride=1,
      padding=1,
    )
    self.linear1 = nn.Linear(in_features=64, out_features=1)
    self.linear2 = nn.Linear(in_features=841 + 512, out_features=512)
    self.out = nn.Linear(in_features=512, out_features=10)

    self.register_parameter(
      "active_state", nn.Parameter(torch.empty(512).normal_(), requires_grad=True)
    )
    self.register_parameter(
      "memory", nn.Parameter(torch.empty(512, memory).normal_(), requires_grad=True)
    )
    self.register_parameter(
      "w",
      nn.Parameter(torch.empty(memory, 1).normal_(), requires_grad=True),
    )
    self.register_parameter(
      "b",
      nn.Parameter(torch.empty(1).normal_(), requires_grad=True),
    )
    self.nticks = nticks

  def forward(self, x):
    bs = x.size(0)
    nlm_state = self.memory.unsqueeze(0).expand(bs, -1, -1)
    w = self.w.unsqueeze(0).expand(bs, -1, -1)

    extracted_feature = self.feature_extractor(x)  # B,64,H,W
    extracted_feature = extracted_feature.flatten(2).permute(0, 2, 1)  # B,H*W,64
    extracted_feature = f.relu(self.linear1(extracted_feature).squeeze(-1))  # B,H*W
    active_state = self.active_state.unsqueeze(0).expand(bs, -1)  # B,M

    for tick_idx in range(self.nticks):
      post_act = torch.concat([extracted_feature, active_state], dim=1)  # B,H*W+M
      pre_act = f.relu(self.linear2(post_act).unsqueeze(-1))  # B,512,1
      # print(f"== {tick_idx=}")
      # print(f"{active_state.shape=}")
      # print(f"{extracted_feature.shape=}")
      # print(f"{post_act.shape=}")
      # print(f"{pre_act.shape=}")
      # print(f"{nlm_state.shape=}")
      nlm_state = nlm_state[:, :, 1:]
      nlm_state = torch.concat([nlm_state, pre_act], dim=2)
      active_state = torch.bmm(nlm_state, w).squeeze(-1)
      active_state = active_state + self.b
      # print()

    return self.out(active_state)


class Model2(Model):

  def forward(self, x):
    bs = x.size(0)
    nlm_state = self.memory.unsqueeze(0).expand(bs, -1, -1)
    w = self.w.unsqueeze(0).expand(bs, -1, -1)

    extracted_feature = self.feature_extractor(x)  # B,64,H,W
    extracted_feature = extracted_feature.flatten(2).permute(0, 2, 1)  # B,H*W,64
    extracted_feature = f.relu(self.linear1(extracted_feature).squeeze(-1))  # B,H*W
    active_state = self.active_state.unsqueeze(0).expand(bs, -1)  # B,M

    outs = []
    for tick_idx in range(self.nticks):
      post_act = torch.concat([extracted_feature, active_state], dim=1)  # B,H*W+M
      pre_act = f.relu(self.linear2(post_act).unsqueeze(-1))  # B,512,1
      # print(f"== {tick_idx=}")
      # print(f"{active_state.shape=}")
      # print(f"{extracted_feature.shape=}")
      # print(f"{post_act.shape=}")
      # print(f"{pre_act.shape=}")
      # print(f"{nlm_state.shape=}")
      nlm_state = nlm_state[:, :, 1:]
      nlm_state = torch.concat([nlm_state, pre_act], dim=2)
      active_state = torch.bmm(nlm_state, w).squeeze(-1)
      active_state = active_state + self.b
      outs.append(self.out(active_state))
      # print()

    return outs

if __name__ == "__main__":
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
  model = Model2(memory=15, nticks=10)
  model.train()
  init_mem = model.memory.clone().detach()
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
  before = model.active_state.clone().detach()
  before_memory = model.memory.clone().detach()
  before_w = model.w.clone().detach()
  before_b = model.b.clone().detach()
  run1_inputs, run1_labels, run1_outputs, run1_preds = [], [], [], []
  with torch.no_grad():
    for _i1, (images, labels) in enumerate(testloader):
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
  post_mem = model.memory.clone().detach().cpu()
  print((post_mem - init_mem).sum())

  track2 = None
  run2_inputs, run2_labels, run2_outputs, run2_preds = [], [], [], []
  for _ntick in range(1, 20):
    correct, total = 0, 0
    model.nticks = _ntick
    with torch.no_grad():
      for _i1, (images, labels) in enumerate(testloader):
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
  after = model.active_state.clone().detach()
  after_memory = model.memory.clone().detach()
  after_w = model.w.clone().detach()
  after_b = model.b.clone().detach()

  # run2_inputs = [each for sublist in run2_inputs for each in sublist]
  # run2_labels = [each for sublist in run2_labels for each in sublist]
  # run2_outputs = [each for sublist in run2_outputs for each in sublist]
  # run2_preds = [each for sublist in run2_preds for each in sublist]

  import numpy as np

# vim: ts=2 sts=2 sw=2 et
