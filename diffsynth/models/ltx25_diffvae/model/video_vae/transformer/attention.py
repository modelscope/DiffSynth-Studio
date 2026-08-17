import torch
from torch import nn

from diffsynth.models.ltx25_diffvae.model.video_vae.transformer.det_attn_rope import det_qkv_rope
from diffsynth.models.ltx25_diffvae.model.video_vae.transformer.fallback_na import EagerSdpaAttention
from diffsynth.models.ltx25_diffvae.model.video_vae.transformer.qkv import QKVProjections
from diffsynth.models.ltx25_diffvae.model.video_vae.transformer.rope_math import (
    DEFAULT_ABS_ROPE_NUM_TILES,
    default_rope_dim_split,
    rope_inv_freqs,
)


class NeighborhoodAttention3D(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        head_dim: int = 64,
        rope_dim_split: tuple[int, int, int] | None = None,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim % head_dim:
            raise ValueError(f"dim={dim} must be divisible by head_dim={head_dim}")
        rope_dim_split = rope_dim_split or default_rope_dim_split(head_dim)
        if sum(rope_dim_split) != head_dim:
            raise ValueError(f"rope_dim_split={rope_dim_split} must sum to head_dim={head_dim}")

        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.kernel_size = tuple(kernel_size)
        self.scale = head_dim**-0.5
        self.rope_dim_split = rope_dim_split
        self.rope_num_tiles = DEFAULT_ABS_ROPE_NUM_TILES
        self.rope_compute_dtype = torch.float32
        self.attention_function = EagerSdpaAttention()
        self.register_buffer("rope_inv_t", rope_inv_freqs(rope_dim_split[0], rope_base), persistent=False)
        self.register_buffer("rope_inv_h", rope_inv_freqs(rope_dim_split[1], rope_base), persistent=False)
        self.register_buffer("rope_inv_w", rope_inv_freqs(rope_dim_split[2], rope_base), persistent=False)
        self.qkv = QKVProjections(dim)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)

    def project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, frames, height, width, _ = x.shape
        q, k, v = self.qkv(x)
        shape = (batch, frames, height, width, self.num_heads, self.head_dim)
        return q.view(shape), k.view(shape), v.view(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, height, width, _ = x.shape
        if any(size < kernel for size, kernel in zip((frames, height, width), self.kernel_size, strict=True)):
            raise ValueError(f"input {(frames, height, width)} is smaller than neighborhood kernel {self.kernel_size}")
        q, k, v = det_qkv_rope(self, x)
        output = self.attention_function(self, q.contiguous(), k.contiguous(), v.contiguous())
        return self.proj(output.reshape(batch, frames, height, width, self.dim))
