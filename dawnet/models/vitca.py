import itertools
import logging
import math
import random
import time
from pathlib import Path

import dill
from lightning.pytorch.core.module import LightningModule
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import optimizer
import torchvision.transforms as T
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader

from dawnet.layers.transformer.selfattention import LocalAttention
from dawnet.losses.ca_loss import CALoss
from dawnet.models.perceive import Module
from dawnet.datasets.landcoverrep import LandCoverRepDataset
from dawnet.data.transforms.vision import GetBands


LOGGER = logging.getLogger(__name__)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def norm_grad(net):
    for _, p in net.named_parameters():
        if p.grad is not None and p.requires_grad:
            p.grad /= p.grad.norm() + 1e-8


def masking_schedule(
    i,
    schedule_start=500,
    schedule_end=10000,
    prob_stages=3,
    max_prob=0.75,
    patch_shape_stages=3,
    max_patch_shape=(10, 10),
    random_seed=None,
):
    """
    Masking schedule gets geometrically harder as i gets larger. The number of choices of masking strategies increases
    as i increases, and they get more challenging. A single one is picked from the pool at random.
    """
    start_prob = 0.25
    start_patch_shape = (1, 1)
    if i == -1:
        i = schedule_end + 1
    probs = np.linspace(start_prob, max_prob, num=prob_stages, dtype=np.float32)
    # TODO: get irregular patch shape combinations working (i.e., separate patch height and width)
    patch_shapes = np.linspace(
        start_patch_shape[0],
        max_patch_shape[0],
        num=patch_shape_stages,
        dtype=np.float32,
    ).astype(np.int32)
    combs = list(itertools.product(probs, patch_shapes))
    sched = np.geomspace(
        schedule_start, schedule_end, num=len(combs), dtype=np.float32
    ).astype(np.int32)
    for idx, stage in enumerate(sched):
        if i <= stage:
            combs = combs[: idx + 1]
            break
    if random_seed is not None:
        random.seed(random_seed)
    p, p_s = random.choice(combs)
    return p, pair(p_s)


def random_mask(
    x, p=0.5, patch_shape=(1, 1), mask_type="dropout", random_seed=None, device="cpu"
):
    assert p >= 0 and p <= 1, "probability p must be between 0 and 1"
    assert (
        x.shape[2] % patch_shape[0] == 0 and x.shape[3] % patch_shape[1] == 0
    ), "Image dimensions must be divisible by the patch size."
    x = rearrange(
        x, "b c (h p1) (w p2) -> b (p1 p2 c) h w", p1=patch_shape[0], p2=patch_shape[1]
    )
    b, _, h, w = x.shape
    if random_seed is not None:
        # for deterministic and reproducible masking
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
    mask = (torch.rand(b, 1, h, w, device=device) + (1 - p)).floor()
    if mask_type == "noise":
        if random_seed is not None:
            # for deterministic and reproducible masking
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
        noise = (1 - mask) * torch.randn_like(mask, device=device)
        masked_x = rearrange(
            mask * x + noise,
            "b (p1 p2 c) h w -> b c (h p1) (w p2)",
            p1=patch_shape[0],
            p2=patch_shape[1],
        )
    elif mask_type == "dropout":
        masked_x = rearrange(
            mask * x,
            "b (p1 p2 c) h w -> b c (h p1) (w p2)",
            p1=patch_shape[0],
            p2=patch_shape[1],
        )
    return masked_x


class LayerPrenorm(nn.Module):
    """
    Args:
        dim: the hidden norm
        fn: the function
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        LayerPrenorm(
                            dim,
                            LocalAttention(
                                dim,
                                attn_neighborhood_size=(3, 3),
                                heads=heads,
                                head_dim=head_dim,
                                dropout=dropout,
                            ),
                        ),
                        LayerPrenorm(dim, FeedForward(dim, mlp_dim, dim, dropout=dropout)),
                    ]
                )
            )

    def encode(self, x, attn, ff, h=None, w=None, **kwargs):
        x = attn(x, h=h, w=w) + x
        x = ff(x) + x
        return x

    def forward(self, x, h=None, w=None, **kwargs):
        if self.training and len(self.layers) > 1:
            # gradient checkpointing to save memory but at the cost of
            # re-computing forward pass during backward pass
            funcs = [
                lambda _x: self.encode(_x, attn, ff, h, w, **kwargs)
                for attn, ff in self.layers
            ]
            x = torch.utils.checkpoint.checkpoint_sequential(
                funcs, segments=len(funcs), input=x
            )
        else:
            for attn, ff in self.layers:
                x = self.encode(x, attn, ff, h, w, **kwargs)
        return x


def vit_positional_encoding(n, dim):
    position = torch.arange(n).unsqueeze(1)
    div_term_even = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    div_term_odd = torch.exp(torch.arange(1, dim, 2) * (-math.log(10000.0) / dim))
    pe = torch.zeros(n, 1, dim)
    pe[:, 0, 0::2] = torch.sin(position * div_term_even)
    pe[:, 0, 1::2] = torch.cos(position * div_term_odd)
    return pe.transpose(0, 1)


class ViTCA(Module):
    _hooks_def = {
        "post-iteration": {
            "help": "Run after an iteration",
            "args_and_result": ["self", "z_star", "idx"],
        },
    }

    def __init__(
        self,
        *,
        depth=1,
        heads=1,
        mlp_dim=64,
        dropout=0.0,
        cell_in_chns=3,
        cell_out_chns=3,
        cell_hidden_chns=9,
        embed_dim=32,
        embed_dropout=0.0,
        device="cpu",
    ):
        super().__init__()
        self.device = device

        self.patch_height, self.patch_width = 1, 1

        # computing dimensions for layers
        self.cell_pe_patch_dim = 0
        self.cell_in_patch_dim = cell_in_chns * self.patch_height * self.patch_width
        self.cell_out_patch_dim = cell_out_chns * self.patch_height * self.patch_width
        self.cell_hidden_chns = cell_hidden_chns
        self.cell_update_dim = self.cell_out_patch_dim + self.cell_hidden_chns
        self.cell_dim = (
            self.cell_pe_patch_dim
            + self.cell_in_patch_dim
            + self.cell_out_patch_dim
            + self.cell_hidden_chns
        )

        # rearranging from 2D grid to 1D sequence
        self.rearrange_cells = Rearrange("b c h w -> b (h w) c")

        self.patchify = Rearrange(
            "b c (h p1) (w p2) -> b (p1 p2 c) h w",
            p1=self.patch_height,
            p2=self.patch_width,
        )
        self.unpatchify = Rearrange(
            "b (p1 p2 c) h w -> b c (h p1) (w p2)",
            p1=self.patch_height,
            p2=self.patch_width,
        )

        self.cell_to_embedding = nn.Linear(self.cell_dim, embed_dim)
        self.dropout = nn.Dropout(embed_dropout)
        self.transformer = Transformer(
            embed_dim, depth, heads, embed_dim // heads, mlp_dim, dropout
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.cell_update_dim),
        )

        # don't update cells before first backward pass or else cell grid will
        # have immensely diverged and grads will be large and unhelpful
        self.mlp_head[1].weight.data.zero_()
        self.mlp_head[1].bias.data.zero_()

    def f(self, cells, update_rate=0.5, **kwargs):
        _cells = cells
        x = self.rearrange_cells(_cells)
        x = self.cell_to_embedding(x)
        x = x + vit_positional_encoding(x.shape[-2], x.shape[-1]).cuda()
        x = self.dropout(x)
        x = self.transformer(x, h=cells.shape[-2], w=cells.shape[-1], **kwargs)

        # stochastic cell state update
        b, _, h, w = cells.shape
        update = rearrange(self.mlp_head(x), "b (h w) c -> b c h w", h=h, w=w)
        if update_rate < 1.0:
            update_mask = (
                torch.rand(b, 1, h, w, device=self.device) + update_rate
            ).floor()
            update = update_mask * update
        updated = cells[:, self.cell_pe_patch_dim + self.cell_in_patch_dim :] + update
        cells = torch.cat(
            [cells[:, : self.cell_pe_patch_dim + self.cell_in_patch_dim], updated], 1
        )

        return cells

    def forward(self, cells, step_n=1, update_rate=0.5, chkpt_segments=1, **kwargs):
        if self.training and chkpt_segments > 1:
            # gradient checkpointing to save memory but at the cost of
            # re-computing forward pass during backward pass
            z_star = torch.utils.checkpoint.checkpoint_sequential(
                self.f,
                cells,
                segments=chkpt_segments,
                seq_length=step_n,
                update_rate=update_rate,
                kwargs=kwargs,
            )
        else:
            z_star = cells
            # z_star[:, :3, :, :] = z_star[:, :3, :, :] / 4
            # image = make_grid(self.get_rgb_out(z_star), nrow=2)
            # print(0,image.sum())
            for _ in range(step_n):
                z_star = self.f(z_star, update_rate, **kwargs)
                z_star = self.run_hook(
                    "post-iteration", kwargs={"self": self, "z_star": z_star, "idx": _}
                )["z_star"]

        return z_star

    def seed(self, rgb_in, sz):
        patch_height, patch_width = self.patch_height, self.patch_width

        assert (
            sz[0] % patch_height == 0 and sz[1] % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        size = (sz[0] // patch_height, sz[1] // patch_width)

        # for storing input from external source
        assert sz[0] == rgb_in.shape[2] and sz[1] == rgb_in.shape[3]
        n = rgb_in.shape[0]
        rgb_in_state = self.patchify(rgb_in)

        # initialize cell output channels with 0.5 (gray image)
        rgb_out_state = (
            torch.zeros(  # JODO: what does self.cell_out_patch_dim mean?
                n, self.cell_out_patch_dim, size[0], size[1], device=self.device
            )
            + 0.5
        )

        # initialize hidden channels with 0 for inter-cell communication
        hidden_state = torch.zeros(
            n, self.cell_hidden_chns, size[0], size[1], device=self.device
        )

        seed_state = torch.cat([rgb_in_state, rgb_out_state, hidden_state], 1)

        return seed_state

    def get_pe_in(self, x):
        pe_patch = x[:, : self.cell_pe_patch_dim]
        pe = self.unpatchify(pe_patch)
        return pe

    def get_rgb_in(self, x):
        rgb_patch = x[
            :, self.cell_pe_patch_dim : self.cell_pe_patch_dim + self.cell_in_patch_dim
        ]
        rgb = self.unpatchify(rgb_patch)
        return rgb

    def get_rgb_out(self, x):
        rgb_patch = x[
            :,
            self.cell_pe_patch_dim
            + self.cell_in_patch_dim : self.cell_pe_patch_dim
            + self.cell_in_patch_dim
            + self.cell_out_patch_dim,
        ]
        rgb = self.unpatchify(rgb_patch)
        return rgb

    def get_rgb(self, x):
        rgb_patch = x[
            :,
            self.cell_pe_patch_dim : self.cell_pe_patch_dim
            + self.cell_in_patch_dim
            + self.cell_out_patch_dim,
        ]
        rgb = self.unpatchify(rgb_patch)
        return rgb

    def get_pe_and_rgb(self, x):
        pe_and_rgb_patch = x[
            :,
            : self.cell_pe_patch_dim + self.cell_in_patch_dim + self.cell_out_patch_dim,
        ]
        pe_and_rgb = self.unpatchify(pe_and_rgb_patch)
        return pe_and_rgb

    def get_hidden(self, x):
        hidden = x[
            :,
            self.cell_pe_patch_dim + self.cell_in_patch_dim + self.cell_out_patch_dim :,
        ]
        return hidden


def hook(self, z_star, idx):
    from torchvision.utils import make_grid, save_image

    outpath = "/home/john/repaper_qad/vitca.pytorch/downloads/landcoverrep-vitca/debug"
    image = make_grid(self.get_rgb_out(z_star), nrow=2)
    print(idx + 1, image.sum())
    save_image(image, f"{outpath}/{idx+1:02}.png")
    if idx == 30:
        z_star[:, 0:3, :, :] = z_star[:, 0:3, :, :] * 0
        image = make_grid(self.get_rgb_out(z_star), nrow=2)
        print("Fixed", image.sum())
        save_image(image, f"{outpath}/change_{idx+1:02}.png")

    return {"self": self, "z_star": z_star, "idx": idx}


def run_vitca_inference():
    # TODO: this is the interaction of the agent with the environment
    model = ViTCA(
        depth=1,
        heads=4,
        mlp_dim=64,
        dropout=0.0,
        cell_in_chns=3,
        cell_out_chns=3,
        cell_hidden_chns=32,
        embed_dim=128,
        embed_dropout=0.0,
        device="cuda:0",
    ).cuda()
    model.register_hook("post-iteration", hook)
    import pickle
    import dill

    ckpt_path = "/home/john/repaper_qad/vitca.pytorch/downloads/landcoverrep-vitca/to_save.pth.tar"
    checkpoint = torch.load(ckpt_path, pickle_module=dill)
    model.load_state_dict(checkpoint["state_dict"])
    with open("/home/john/temp/z0.pkl", "rb") as fi:
        z_0 = pickle.load(fi)
    z_0 = z_0[:8]
    z_0 = torch.FloatTensor(z_0).cuda()
    with torch.no_grad():
        z_T = model(z_0, step_n=64, update_rate=0.5)


class TheAgent(nn.Module):
    # TODO: this is a scoped problem, utilizing different model from different
    # modules. This will usually be the entrypoint to running the training and
    # the inference.
    # You would like to make this reproducible. You would like to make this
    # comparable and exchangeable with other researches.

    def __init__(self):
        super(TheAgent, self).__init__()

        self.device = "cuda:0"
        self.model = ViTCA(
            depth=1,
            heads=4,
            mlp_dim=64,
            dropout=0.0,
            cell_in_chns=3,
            cell_out_chns=3,
            cell_hidden_chns=32,
            embed_dim=128,
            embed_dropout=0.0,
            device="cuda:0",
        ).cuda()
        self.loss = CALoss(rec_factor=1e2, overflow_factor=1e2)
        self.opt = optim.AdamW(params=self.model.parameters(), lr=1e-3)
        # self.lr_sched = lambda: instantiate(cf.gexperiment.trainer.lr_sched)
        self.lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.opt, T_max=100000)

        self.train_losses = {
            "reconstruction_loss": [],
            "reconstruction_factor": [],
            "overflow_loss": [],
            "overflow_factor": [],
            "total_loss": [],
        }
        self.avg_val_losses = {
            "reconstruction_loss": [],
            "reconstruction_factor": [],
            "overflow_loss": [],
            "overflow_factor": [],
            "total_loss": [],
            "psnr": [],
        }
        self.best_avg_val_rec_err = 1e8

        self.x_pool = []
        self.y_pool = []
        self.z_pool = []

    def load_pretrained_model(self, ckpt_pth):
        ckpt_pth = Path(ckpt_pth)
        if ckpt_pth.is_file():
            LOGGER.info("Loading pretrained model checkpoint at '%s'...", ckpt_pth)
            checkpoint = torch.load(str(ckpt_pth), pickle_module=dill)
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            LOGGER.info(
                "Loaded pretrained model checkpoint '%s' (at iter %s)",
                ckpt_pth,
                checkpoint["iter"],
            )
            return True
        else:
            LOGGER.info("Checkpoint '%s' not found.", ckpt_pth)
            return False

    def save_checkpoint(self, i, ckpt_dirname):
        chk_fname = ckpt_dirname / f"ckpt_{i}.pth.tar"
        LOGGER.info("Saving checkpoint at iter %s at %s.", i, chk_fname)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "iter": i,
                "opt": self.opt.state_dict(),
                "lr_sched": self.lr_sched.state_dict(),
                "last_lr": self.opt.param_groups[0]["lr"],
                "train_losses": self.train_losses,
                "avg_val_losses": self.avg_val_losses,
                "best_avg_val_rec_err": self.best_avg_val_rec_err,
                "pools": {"x": self.x_pool, "y": self.y_pool, "z": self.z_pool}
            },
            chk_fname,
            pickle_module=dill,
        )

    def forward(self, step, x, y, phase="train"):
        # TODO: can have "step" as internal counter
        if phase == "train":
            return self.ca_train_forward(step, x, y)
        elif phase == "validation":
            return self.ca_val_forward(step, x, y)
        elif phase == "test":
            return self.ca_test_forward(x, y)
        elif phase == "inference":
            return self.ca_inf_forward(x, y)

    def ca_train_forward(self, step, x, y):
        train_size = (x.shape[-2], x.shape[-1])
        train_batch_size = x.shape[0]
        was_sampled = False
        if len(self.z_pool) > train_batch_size and step % 2 == 0:
            # sample from the nca pool, which includes cell states and associated
            # ground truths, every 2nd iter
            x = torch.cat(self.x_pool[:train_batch_size]).cuda()
            y = torch.cat(self.y_pool[:train_batch_size]).cuda()
            with torch.no_grad():
                z_0 = torch.cat(self.z_pool[:train_batch_size]).cuda()
            was_sampled = True
        else:
            x, y = x.cuda(), y.cuda()
            p, p_s = masking_schedule(
                step,
                schedule_start=500,
                schedule_end=10000,
                max_prob=0.75,
                prob_stages=3,
                max_patch_shape=[4, 4],
                patch_shape_stages=3,
            )
            masked_x = random_mask(
                x,
                p=p,
                mask_type="dropout",
                patch_shape=p_s,
                device=self.device,
            )
            with torch.no_grad():
                z_0 = self.model.seed(rgb_in=masked_x, sz=train_size)

        # iterate nca
        T = np.random.randint(8, 32 + 1)
        z_T = self.model(
            z_0,
            step_n=T,
            update_rate=0.5,
            chkpt_segments=1,
            attn_size=[3, 3],
        )

        return {
            "output_cells": z_T,
            "output_img": self.model.get_rgb_out(z_T),
            "ground_truth": {"x": x, "y": y},
            "was_sampled": was_sampled,
        }

    def ca_val_forward(self, step, x, y):
        validation_size = [32, 32]

        # use all available kinds of patch shapes and probs
        # TODO: come up with random seed that's not dependent on step since val batch size can affect it
        p, p_s = masking_schedule(
            -1,
            max_prob=0.75,
            prob_stages=3,
            max_patch_shape=[4, 4],
            patch_shape_stages=3,
            random_seed=step,
        )
        masked_x = random_mask(
            x,
            p=p,
            mask_type="dropout",
            patch_shape=p_s,
            random_seed=1,
            device=self.model.device,
        )
        z_0 = self.model.seed(rgb_in=masked_x, sz=validation_size)

        # iterate nca
        z_T = self.model(
            z_0,
            step_n=64,
            update_rate=0.5,
            attn_size=[3, 3],
        )

        return {
            "output_cells": z_T,
            "masked_input": self.model.get_rgb_in(z_0),
            "output_img": self.model.get_rgb_out(z_T),
            "ground_truth": {"x": x, "y": y},
        }

    def ca_test_forward(self, x, y):
        input_size = [32, 32]
        p, p_s = masking_schedule(
            -1,
            max_prob=0.75,
            prob_stages=3,
            max_patch_shape=[4, 4],
            patch_shape_stages=3,
        )
        masked_x = random_mask(
            x,
            p=p,
            mask_type="dropout",
            patch_shape=p_s,
            device=self.model.device,
        )

        z_0 = self.model.seed(rgb_in=masked_x, sz=input_size)

        # iterate nca
        z_T = self.model(
            z_0,
            step_n=6,
            update_rate=0.5,
            attn_size=[3, 3],
        )

        return {
            "output_cells": z_T,
            "masked_input": self.model.get_rgb_in(z_0),
            "output_img": self.model.get_rgb_out(z_T),
            "ground_truth": {"x": x, "y": y},
        }

    def ca_inf_forward(self, x, y):
        input_size = [32, 32]
        p = 0.75
        p_s = [4, 4]
        mask_type = "dropout"
        masked_x = random_mask(
            x, p=p, mask_type=mask_type, patch_shape=p_s, device=self.model.device
        )

        z_0 = self.model.seed(rgb_in=masked_x, sz=input_size)

        # iterate nca
        z_T = self.model(
            z_0,
            step_n=1,
            update_rate=0.5,
            attn_size=[3, 3],
        )

        return {
            "output_cells": z_T,
            "masked_input": self.model.get_rgb_in(z_0),
            "output_img": self.model.get_rgb_out(z_T),
            "ground_truth": {"x": x, "y": y},
        }

    def update_pools(self, x, y, z_T):
        """
        If states were newly created, add new states to nca pool, shuffle, and retain first {pool_size} states.
        If states were sampled from the pool, replace their old states, shuffle, and retain first {pool_size}
        states.
        """
        pool_size = 1024
        self.x_pool += list(torch.split(x, 1))
        self.y_pool += list(torch.split(y, 1))
        self.z_pool += list(torch.split(z_T, 1))
        pools = list(zip(self.x_pool, self.y_pool, self.z_pool))
        random.shuffle(pools)
        self.x_pool, self.y_pool, self.z_pool = zip(*pools)
        self.x_pool = list(self.x_pool[:pool_size])
        self.y_pool = list(self.y_pool[:pool_size])
        self.z_pool = list(self.z_pool[:pool_size])

    def update_tracked_scalars(self, scalars, step, phase="train"):
        if phase == "train":
            for scalar_name, scalar in scalars.items():
                self.train_losses[f"{scalar_name}"].append((step, scalar))
        elif phase == "validation":
            for scalar_name, scalar in scalars.items():
                self.avg_val_losses[f"{scalar_name}"].append((step, scalar))


# Main train loop code
def run_vitca_train() -> float:
    dataset_root = "/home/john/repaper_qad/vitca.pytorch/reproduce/code/downloads/datasets"

    save = Path("/home/john/repaper_qad/vitca.pytorch/reproduce/code/downloads/temp")

    # Setup model, loss, opt, and train & val loss arrays
    model_and_trainer = TheAgent()
    model = model_and_trainer.model
    opt = model_and_trainer.opt
    loss = model_and_trainer.loss
    lr_sched = model_and_trainer.lr_sched

    # Setup dataset and dataloader
    train_size = [32, 32]
    train_batch_size = 16

    transforms = T.Compose(
        [T.ToTensor(), GetBands(3), T.RandomCrop(train_size), T.RandomHorizontalFlip(), T.RandomVerticalFlip()])
    train_dataset = LandCoverRepDataset(root=dataset_root).get_subset("train", transform=transforms)
    sampler = torch.utils.data.RandomSampler(
        train_dataset,
        replacement=True,
        num_samples= 100000 * train_batch_size,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True
    )

    # Training loop
    model_and_trainer.train()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    start = time.time()
    for i, (x, y) in enumerate(train_loader, start=1):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            # Forward pass
            results = model_and_trainer(i, x, y)

            # Compute losses
            losses = loss(model, results, phase="train")
            total_loss = losses["rec_loss"] + losses["overflow_loss"]
            if i % 10 == 0:
                print(f"{i}: {total_loss.item()}, {losses}")

        # Backward pass
        opt.zero_grad()

        scaler.scale(total_loss).backward()

        # Normalize gradients
        # TODO: don't normalize gradients for non-CA models
        with torch.no_grad():
            norm_grad(model)

        # opt.step()
        scaler.step(opt)
        scaler.update()
        lr_sched.step()

        # Add new states to nca pool, shuffle, and retain first {pool_size} states
        model_and_trainer.update_pools(
            results["ground_truth"]["x"],
            results["ground_truth"]["y"],
            results["output_cells"].detach(),
        )

        if i % 1000 == 0:
            model_and_trainer.save_checkpoint(i, save)

    print(f"Last: {time.time() - start}")
    return model_and_trainer.best_avg_val_rec_err

"""
Try with Pytorch Lightning + Trainer

- Test if we can simplify the training job with trainer
"""
import lightning.pytorch as pl
from typing import Any


class LitViTCA(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # TODONEXT
        # Would generally want to refactor the VitCA experiments to organize it
        # appropriately: (1) have test to ensure everything working well,
        # (2) independent enough to not unexpectedly change behavior, (3) can be quickly
        # reused, (4) knowledge are accumulated.
        #
        # - should be able to config the model
        # - essentially each LightningModule is the whole experiment, where you define
        # the totality of both the model[s] and how they are trained with the data.
        # - use `norm_grad` in LitViTCA version
        # - use `update_pools` in LitViTCA version
        # - adjust checkpoint frequency
        # - define the expected input (e.g. it's unclear to decide the format of the
        # input), similarly with the output.
        # - provide code to quickly copy code to a new folder, to quickly setup
        # experimentation environment
        # - exploit the `forward()` to do normal inference
        # - make use of Lightning's training_step, validation_step
        # - not directly relating to LitViTCA, but you might want to encapsulate the
        # optimization, backprop process into this module
        # - need to handle config interdependency (e.g. image size correlates to
        # the layer input dim)
        # - the LitViTCA is not the encapsulation of the whole experiment, because
        # you cannot swap data. As a result, the config for the experiment will have
        # data-related config that should be stored and does not relate to the
        # LightningModule
        #   - It means we would still want to use Hydra and its way of initiating model
        self.model = ViTCA(
            depth=1,
            heads=4,
            mlp_dim=64,
            dropout=0.0,
            cell_in_chns=3,
            cell_out_chns=3,
            cell_hidden_chns=32,
            embed_dim=128,
            embed_dropout=0.0,
            device="cuda:0"
        )
        # TODONEXT
        # it's nice to be able to switch the loss
        self.loss = CALoss(rec_factor=1e2, overflow_factor=1e2)
        self.x_pool = []
        self.y_pool = []
        self.z_pool = []

    def configure_optimizers(self) -> Any:
        optimizer = optim.AdamW(params=self.model.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch

        results = self.ca_train_forward(batch_idx, x, y)

        # Compute losses
        losses = self.loss(self.model, results, phase="train")
        total_loss = losses["rec_loss"] + losses["overflow_loss"]
        # if batch_idx % 10 == 0:
        #     print(f"{batch_idx}: {total_loss.item()}, {losses}")

        return total_loss

    def ca_train_forward(self, step, x, y):
        train_size = (x.shape[-2], x.shape[-1])
        train_batch_size = x.shape[0]
        was_sampled = False
        if len(self.z_pool) > train_batch_size and step % 2 == 0:
            # sample from the nca pool, which includes cell states and associated
            # ground truths, every 2nd iter
            x = torch.cat(self.x_pool[:train_batch_size]).cuda()
            y = torch.cat(self.y_pool[:train_batch_size]).cuda()
            with torch.no_grad():
                z_0 = torch.cat(self.z_pool[:train_batch_size]).cuda()
            was_sampled = True
        else:
            x, y = x.cuda(), y.cuda()
            p, p_s = masking_schedule(
                step,
                schedule_start=500,
                schedule_end=10000,
                max_prob=0.75,
                prob_stages=3,
                max_patch_shape=[4, 4],
                patch_shape_stages=3,
            )
            masked_x = random_mask(
                x,
                p=p,
                mask_type="dropout",
                patch_shape=p_s,
                device=self.device,
            )
            with torch.no_grad():
                z_0 = self.model.seed(rgb_in=masked_x, sz=train_size)

        # iterate nca
        T = np.random.randint(8, 32 + 1)
        z_T = self.model(
            z_0,
            step_n=T,
            update_rate=0.5,
            chkpt_segments=1,
            attn_size=[3, 3],
        )

        return {
            "output_cells": z_T,
            "output_img": self.model.get_rgb_out(z_T),
            "ground_truth": {"x": x, "y": y},
            "was_sampled": was_sampled,
        }


if __name__ == "__main__":
    # run_vitca_train()
    model = LitViTCA()

    dataset_root = "/home/john/repaper_qad/vitca.pytorch/reproduce/code/downloads/datasets"
    train_size = [32, 32]
    train_batch_size = 16
    transforms = T.Compose(
        [T.ToTensor(), GetBands(3), T.RandomCrop(train_size), T.RandomHorizontalFlip(), T.RandomVerticalFlip()])
    train_dataset = LandCoverRepDataset(root=dataset_root).get_subset("train", transform=transforms)
    sampler = torch.utils.data.RandomSampler(
        train_dataset,
        replacement=True,
        num_samples= 100000 * train_batch_size,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True
    )

    trainer = pl.Trainer(
        precision=16, accelerator="gpu", min_steps=100000, max_epochs=1
    )
    trainer.fit(model=model, train_dataloaders=train_loader)
