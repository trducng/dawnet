"""
Credits:
    - Local attention: @mattie https://openreview.net/forum?id=9t24EBSlZOa
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from scipy import signal


def moore_neighborhood_filters(neighborhood_size: Tuple[int, int]):
    """Provide convolution filters to retrieve Moore neighborhood

    Args:
        neighborhood_size: height and width of the surrounding region
    """
    height, width = neighborhood_size
    impulses = []
    for i in range(height):
        for j in range(width):
            impulse = signal.unit_impulse((height, width), idx=(i, j), dtype=np.float32)
            impulses.append(impulse)
    filters = torch.tensor(np.stack(impulses))

    return filters


class LocalExpansion(nn.Module):
    """Extract the 2D local neighborhood of self-attention into a dedicated channel

    - To explicitly perform local self-attention (without attention masking)
    - Reduce computational effort

    Args:
        neighborhood_size: height and width of the surrounding region
    """

    def __init__(self, attn_neighborhood_size) -> None:
        super().__init__()
        self.attn_filters = nn.parameter.Parameter(
            moore_neighborhood_filters(attn_neighborhood_size),
            requires_grad=False
        )

    def forward(self, x, height, width):
        """
        Args:
            x: tensor of shape [batch x num heads x num pixels x dims]
            height: feature map height
            width: feature map width
        """
        b, h, _, d = x.shape
        y = rearrange(x, "b h (i j) d -> (b h d) 1 i j", i=height, j=width)
        y = F.conv2d(y, self.attn_filters[:, None], padding="same")
        _x = rearrange(y, "(b h d) filter_n i j -> b h (i j) filter_n d", b=b, h=h, d=d)
        return _x


class LocalAttention(nn.Module):
    """Self-attention that only pays attention to the surrounding neighborhood

    Even though normal input to self-attention layer is 3D, this self-attention layer
    assumes that the input is implicitly 4D, where [batch x steps x dim] is just
    reshaped from [batch x height x width x dim].

    Args:
        dim: the input dim to the layer
        attn_neighborhood_size: restrict the steps to attend to other steps within
            this boundary
        heads: the number of heads
        head_dim: the dimension of each head
        dropout: the dropped out percentage
    """

    def __init__(
        self,
        dim: int,
        attn_neighborhood_size: Tuple[int, int],
        heads: int,
        head_dim: int,
        dropout: float,
    ) -> None:

        super().__init__()
        inner_dim = head_dim * heads
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.to_qkv = torch.nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = torch.nn.Softmax(dim=-1)

        self.attn_map = None

        self.to_out = (
            torch.nn.Sequential(
                torch.nn.Linear(inner_dim, dim), torch.nn.Dropout(dropout)
            )
            if project_out
            else torch.nn.Identity()
        )

        # extension
        self.local = LocalExpansion(attn_neighborhood_size)

    def forward(self, x, h=None, w=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        q = rearrange(q, "b h n d -> b h n 1 d")
        k = self.local(k, h, w)  # b h n (attn_height attn_width) d
        v = self.local(v, h, w)  # b h n (attn_height attn_width) d

        dots = (
            torch.matmul(q, k.transpose(-1, -2)) * self.scale
        )  # b h n 1 (attn_height attn_width)

        attn = self.attend(dots)  # b h n 1 (attn_height attn_width)

        self.attn_maps = attn

        out = torch.matmul(attn, v)  # b h n 1 d
        out = rearrange(out, "b h n 1 d -> b n (h d)")
        return self.to_out(out)


class MaskedMultiHeadAttention(nn.Module):
    """Multi-head attention block

    Args:
        input_shape: dimension of the input
        n_heads: the number of attention heads
    """

    def __init__(self, input_shape: int, n_heads: int, dropout: float):
        if input_shape % n_heads:
            raise AttributeError(f"{input_shape=} must be divisible by {n_heads=}")

        super().__init__()
        self.input_shape = input_shape
        self.n_heads = n_heads
        self.drop_attn = nn.Dropout(p=dropout)
        self.kqv = nn.Linear(in_features=input_shape, out_features=input_shape * 3)

    def forward(self, x):
        """Forward through MultiHeadAttention block
        Args:
            x: the input with shape (batch size x sequence length x in dims)
        Returns:
            the output with shape (batch size x sequence length x in dims)
        """
        N, C, D = x.shape
        z = self.kqv(x)

        k = z[:, :, :D].reshape(N, C, self.n_heads, D // self.n_heads).transpose(1, 2)
        q = (
            z[:, :, D : D * 2]
            .reshape(N, C, self.n_heads, D // self.n_heads)
            .transpose(1, 2)
        )
        v = (
            z[:, :, D * 2 :]
            .reshape(N, C, self.n_heads, D // self.n_heads)
            .transpose(1, 2)
        )

        z = torch.matmul(q, k.transpose(2, 3)) * D ** -0.5
        mask = torch.tril(torch.ones(C, C, device=x.device)).view(1, 1, C, C)
        z = z.masked_fill(mask == 0, float("-inf"))
        z = F.softmax(z, dim=-1)
        z = self.drop_attn(z)
        z = torch.matmul(z, v)
        return z.transpose(1, 2).reshape(N, C, D)
