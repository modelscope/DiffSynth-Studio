from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class ChannelLinear(nn.Linear):

    @property
    def in_channels(self) -> int:
        return self.in_features

    @property
    def out_channels(self) -> int:
        return self.out_features


class LinearPixelShuffleUpsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stride: tuple[int, int, int],
        out_channels_reduction_factor: int = 1,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.proj_out_channels = math.prod(stride) * in_channels // out_channels_reduction_factor
        self.out_channels = self.proj_out_channels // math.prod(stride)
        self.proj = nn.Linear(in_channels, self.proj_out_channels, bias=True)

    def forward(self, x: torch.Tensor, drop_leading_frame: bool = True) -> torch.Tensor:
        x = self.proj(x)
        x = rearrange(
            x,
            "b t h w (c p1 p2 p3) -> b (t p1) (h p2) (w p3) c",
            p1=self.stride[0],
            p2=self.stride[1],
            p3=self.stride[2],
        )
        if self.stride[0] == 2 and drop_leading_frame:
            x = x[:, 1:, :, :, :]
        return x


class AdaLNZero(nn.Module):
    NUM_CHUNKS: int = 7

    def __init__(self, dim: int, t_emb_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(t_emb_dim, self.NUM_CHUNKS * dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h = self.proj(F.silu(t_emb))
        chunks = h.chunk(self.NUM_CHUNKS, dim=-1)
        return tuple(c[:, None, None, None, :] for c in chunks)


def modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift
