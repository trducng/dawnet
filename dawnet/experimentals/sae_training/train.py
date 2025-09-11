import bdb
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

from dawnet.datasets.acts import TokenizeDataset, GetActivation
from dawnet.inspector import Inspector
from dawnet import op
from transformers import AutoTokenizer, AutoModelForCausalLM



class BaseModule(nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._logger: Logger = None

  @property
  def logger(self):
    return self._logger

  @logger.setter
  def logger(self, logger):
    self._logger = logger


class SAE(BaseModule):

  @dataclass
  class Config:
    d_act: int    # input activation
    d_feat: int   # feature
    subtract_decoder_bias: bool = False
    b_enc: float = 0.0
    b_dec: float = 0.0
    l1_coeff: float = 5
    train_with_scaled_version: bool = False

  def __init__(self, cfg: "SAE.Config"):
    super().__init__()
    self.cfg = cfg
    self.b_dec = nn.Parameter(torch.ones(self.cfg.d_act) * self.cfg.b_dec)
    self.W_dec = nn.Parameter(torch.empty(self.cfg.d_feat, self.cfg.d_act))
    nn.init.kaiming_uniform_(self.W_dec.data)
    self.b_enc = nn.Parameter(torch.zeros(self.cfg.d_feat) * self.cfg.b_enc)
    self.W_enc = nn.Parameter(  # act x feat
      self.W_dec.data.T.clone().detach().contiguous()
    )

    self.scaler = nn.Buffer(torch.zeros(1, self.cfg.d_act) + 1e-8)
    self.n_iter_since_last_active = nn.Buffer(torch.zeros(self.cfg.d_feat))

  def preprocess_act(self, x):
    # x: batch x act
    if self.training:
      # multiply by 0.9 to decrease the effect of outlier
      self.scaler = torch.maximum(
        self.scaler, torch.abs(x).max(dim=0, keepdim=True).values * 0.9
      )

    scaled_x = x / self.scaler
    return scaled_x

  def postprocess_recon(self, x):
    return x * self.scaler

  def forward(self, x):
    x = self.preprocess_act(x)
    feat = self.encode(x)
    recon = self.decode(feat)
    recon = self.postprocess_recon(recon)

    # update feat
    self.n_iter_since_last_active += 1
    fired_feat = (feat > 0).sum(dim=0)
    self.n_iter_since_last_active[fired_feat > 0] = 0

    return recon, feat

  def encode(self, x):
    # x: batch x act
    if self.cfg.subtract_decoder_bias:
      x = x - self.b_dec
    z = x @ self.W_enc + self.b_enc  # batch x feat
    return F.relu(z)

  def decode(self, feat):
    # feat: batch x feat
    z = feat @ self.W_dec + self.b_dec
    return z

  def loss(self, x):
    if self.cfg.train_with_scaled_version:
      x = self.preprocess_act(x)
      feat = self.encode(x)
      recon = self.decode(feat)

      # update feat
      self.n_iter_since_last_active += 1
      fired_feat = (feat > 0).sum(dim=0)
      self.n_iter_since_last_active[fired_feat > 0] = 0
    else:
      recon, feat = self(x)
    mse = ((recon - x) ** 2).sum(dim=1).mean()

    weighted_feat = feat * self.W_dec.norm(dim=1)
    l1 = weighted_feat.norm(p=1, dim=1).mean()
    loss = mse + self.cfg.l1_coeff * l1

    info = {
      "loss/mse": round(mse.item(), 4),
      "loss/l1": round(l1.item(), 4),
      "loss/total": round(loss.item(), 4),
    }

    if self.logger is not None and self.logger.in_tracking_step():
      self.logger.track_step(info)

    return loss, info

  def evaluate(self, input):
    # input, recon: batch x act
    # feat: batch x feat
    recon, feat = self(input)
    l0 = (feat > 0).float().sum(-1)
    l0 = l0 / feat.shape[-1]
    l0 = l0.mean()
    mse = ((recon - input) ** 2).sum(dim=1).mean()
    variance = ((input - input.mean(0, keepdim=True)) ** 2).sum(-1).mean()
    explained_variance = 1 - mse / variance

    return l0, mse, explained_variance

from dawnet.utils.logger import  LoggingConfig, Logger


@dataclass
class DataConfig:
  model: str
  layer: str
  getter: int
  local_tokenized_data: str


@dataclass
class Config:
  # sae config
  sae: SAE.Config

  # data will depend on the type of training, should overlap with other types
  # of training as much as possible, but will have differences here
  data: DataConfig

  # logging config (will be similar across type of models & training)
  logging: LoggingConfig = field(default_factory=lambda: LoggingConfig(
    tracker="local", log_every_n_steps=10)
  )

  epochs: int = 1
  load_from_ckpt: str = ""


if __name__ == "__main__":
  cfg = Config(
    sae=SAE.Config(
      d_act=2560, d_feat=2560*16, subtract_decoder_bias=True,
      train_with_scaled_version=True, l1_coeff = 1,
    ),
    data=DataConfig(
      model="Qwen/Qwen3-4B-Base",
      layer="model.layers.33",
      getter=0,
      local_tokenized_data="/Users/john/dawnet/experiments/temp/_temp_prepare_data",
    ),
    logging = LoggingConfig(
      tracker="wandb",
      tracker_config=LoggingConfig.WandbConfig(
        project="sae-qwen4b-layer33",
        name="Train using scaled input",
        notes="We calculate based on the scaled input and output.",
      ),
      log_every_n_steps=20
    )
  )

  ckpt = None
  if cfg.load_from_ckpt:
    ckpt = torch.load(cfg.load_from_ckpt, weights_only=False)


  # get the data
  model_name = "Qwen/Qwen3-4B-Base"
  # tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(model_name, device_map="mps")

  # insp = Inspector(model)
  activation_loader = GetActivation(
    cfg=GetActivation.Config(
      insp_layer="model.layers.33",
      insp_model=model,
      insp_getter=0,
      local_tokenized_data="/Users/john/dawnet/experiments/temp/_temp_prepare_data",
    ),
  )
  # load from checkpoint
  if ckpt is not None:
    activation_loader._s.idx = ckpt["idx"]

  sae_name = "qwen4b-base_scaled_input_coeff1.pkl"

  # load from checkpoint
  if ckpt is not None:
    cfg.logging.tracker_config.id = ckpt['tracker']
  logger = Logger.from_config(cfg.logging)

  # initialize the SAE (standard SAE)
  sae = SAE(cfg=cfg.sae)
  sae = sae.to("mps")
  # load from checkpoint
  if ckpt is not None:
    sae.load_state_dict(ckpt["sae"])
  sae.logger = logger

  optimizer = optim.Adam(sae.parameters(), lr=5e-4)
  # load from checkpoint
  if ckpt is not None:
    optimizer.load_state_dict(ckpt["opt"])

  if ckpt is not None:
    state_dict = {
      "sae": sae.state_dict(),
      "opt": optimizer.state_dict()
    }
    torch.save(state_dict, "before_" + sae_name)

  idx = 0   
  # load from checkpoint 
  if ckpt is not None:
    idx = ckpt["idx"]
  start_time = time.time()
  should_train = True
  while should_train:
    try:
      batch = activation_loader.execute()
      loss, info = sae.loss(batch)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      if idx % 100 == 0:
        print(
          f"{idx=}: {loss.item()=} {info} "
          f"{round((time.time() - start_time) / (idx+1), 4)}s/it"
        )
      if idx % 5000 == 0:
        with torch.no_grad():
          l0, mse, explained_variance = sae.evaluate(batch)
          dead_over_5000 = (sae.n_iter_since_last_active > 5000).sum()
          dead_over_10000 = (sae.n_iter_since_last_active > 10000).sum()
          dead_over_15000 = (sae.n_iter_since_last_active > 15000).sum()
        logger.track_step({
          "val/l0": round(l0.item(), 4),
          "val/mse": round(mse.item(), 4),
          "val/explained_variance": round(explained_variance.item(), 4),
          "val/dead_over_5000": int(dead_over_5000.item()),
          "val/dead_over_10000": int(dead_over_10000.item()),
          "val/dead_over_15000": int(dead_over_15000.item()),
        })

      idx += 1
      logger.step()
      if idx >= len(activation_loader._data):
        break
    except KeyboardInterrupt:
      print(
        "\n1. Drop into pdb\n"
        "2. Dump the current state to a file\n"
        "3. Continue training\n"
        "4. Grace stop training"
      )
      in_ = int(input("Select: "))
      while True:
        if in_ in list(range(1, 5)):
          break
        in_ = int(input(
          f"Expect number in {list(range(1,5))} but receive {in_}. Re-select: "
        ))
      if in_ == 1:
        print("Dropped into pdb prompt, can do experiment")
        import pdb; pdb.set_trace()
      elif in_ == 2:
        in2_ = input("Specify path: ")
        state_dict = {
          "sae": sae.state_dict(),
          "opt": optimizer.state_dict()
        }
        torch.save(state_dict, in2_)
      elif in_ == 3:
        print("Continue training...")
      elif in_ == 4:
        should_train = False
    except bdb.BdbQuit:
      print("Closed pdb")
      continue

  state_dict = {
    "sae": sae.state_dict(),
    "opt": optimizer.state_dict()
  }
  torch.save(state_dict, sae_name)
  logger.close()

# vim: ts=2 sts=2 sw=2 et
