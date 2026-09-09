from __future__ import annotations

import numpy as np
import torch

DEFAULT_ABS_ROPE_NUM_TILES = 4


def t_positions(t: int, device: torch.device) -> torch.Tensor:
    return torch.arange(t, dtype=torch.float32, device=device)


def h_positions(h: int, device: torch.device) -> torch.Tensor:
    return torch.arange(h, dtype=torch.float32, device=device)


def default_rope_dim_split(head_dim: int) -> tuple[int, int, int]:
    assert head_dim % 8 == 0, f"head_dim={head_dim} must be a multiple of 8 for default split"
    d_t = (head_dim // 4) // 2 * 2
    d_hw = (head_dim - d_t) // 2
    if d_hw % 2 != 0:
        d_t -= 2
        d_hw = (head_dim - d_t) // 2
    assert d_t > 0
    assert d_hw > 0
    return (d_t, d_hw, d_hw)


def rope_inv_freqs(dim: int, base: float = 10000.0) -> torch.Tensor:
    assert dim % 2 == 0, f"RoPE dim must be even, got {dim}"
    exponents = np.arange(0, dim, 2, dtype=np.float64) / dim
    inv_freqs = 1.0 / np.power(float(base), exponents)
    return torch.from_numpy(inv_freqs).to(torch.float32)


def rot_abs_axis_impl(
    xc: torch.Tensor,
    pos: torch.Tensor,
    inv: torch.Tensor,
    axis: int,
    *,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    out_dtype = xc.dtype
    pairs = xc.reshape(*xc.shape[:-1], xc.shape[-1] // 2, 2)
    xe = pairs[..., 0].to(compute_dtype)
    xo = pairs[..., 1].to(compute_dtype)
    shape = [1, 1, 1, 1, 1, inv.shape[0]]
    shape[axis] = pos.shape[0]
    ang = (pos[:, None] * inv[None, :]).reshape(shape)
    c = ang.cos().to(compute_dtype)
    s = ang.sin().to(compute_dtype)
    re = xe * c - xo * s
    ro = xe * s + xo * c
    out = torch.stack([re, ro], dim=-1).reshape(xc.shape)
    return out.to(out_dtype) if out.dtype != out_dtype else out
