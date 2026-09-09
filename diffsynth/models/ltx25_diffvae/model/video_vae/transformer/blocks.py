from __future__ import annotations

import torch
from torch import nn

from diffsynth.models.ltx25_diffvae.model.video_vae.transformer.attention import NeighborhoodAttention3D
from diffsynth.models.ltx25_diffvae.model.video_vae.transformer.layers import AdaLNZero
from diffsynth.models.ltx25_diffvae.model.video_vae.transformer.swiglu import SwiGLU, plain_mlp

__all__ = [
    "DiffusionNABlock",
    "NABlock",
]


class NABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        rope_dim_split: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim=head_dim, rope_dim_split=rope_dim_split)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        hidden = (int(dim * mlp_ratio) + 15) // 16 * 16
        self.mlp = SwiGLU(dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = plain_mlp(x, self.mlp, self.norm2)
        return x


class DiffusionNABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        context_channels: int,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        rope_dim_split: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        self.context_channels = context_channels
        self.context_proj = nn.Linear(context_channels, dim, bias=True)
        self.scale_shift_table = nn.Parameter(torch.zeros(AdaLNZero.NUM_CHUNKS, dim))

        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim=head_dim, rope_dim_split=rope_dim_split)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        hidden = (int(dim * mlp_ratio) + 15) // 16 * 16
        self.mlp = SwiGLU(dim, hidden)
        self.attn.proj.reset_parameters()

    def _modulation(
        self, modulation: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [
            modulation[i] + self.scale_shift_table[i].view(1, 1, 1, 1, -1) for i in range(AdaLNZero.NUM_CHUNKS)
        ]
        return scale_msa, shift_msa, scale_mlp, shift_mlp
