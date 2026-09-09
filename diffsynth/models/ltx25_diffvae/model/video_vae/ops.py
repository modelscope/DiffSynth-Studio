import torch
from einops import rearrange
from torch import nn


def patchify(x: torch.Tensor, patch_size_hw: int, patch_size_t: int = 1) -> torch.Tensor:
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size_hw, r=patch_size_hw)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b c (f p) (h q) (w r) -> b (c p r q) f h w",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    return x


def unpatchify(x: torch.Tensor, patch_size_hw: int, patch_size_t: int = 1) -> torch.Tensor:
    if patch_size_hw == 1 and patch_size_t == 1:
        return x

    if x.dim() == 4:
        x = rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size_hw, r=patch_size_hw)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c p r q) f h w -> b c (f p) (h q) (w r)",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )

    return x


class PerChannelStatistics(nn.Module):
    def __init__(self, latent_channels: int = 128):
        super().__init__()
        self.register_buffer("std-of-means", torch.ones(latent_channels))
        self.register_buffer("mean-of-means", torch.zeros(latent_channels))

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x)) + self.get_buffer("mean-of-means").view(
            1, -1, 1, 1, 1
        ).to(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)) / self.get_buffer("std-of-means").view(
            1, -1, 1, 1, 1
        ).to(x)
