import torch
from torch import nn

from diffsynth.models.ltx25_diffvae.model.video_vae.transformer.layers import modulate


def residual_mlp(
    x: torch.Tensor,
    mlp: nn.Module,
    norm: nn.RMSNorm,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    return x + mlp(modulate(norm(x), scale, shift))
