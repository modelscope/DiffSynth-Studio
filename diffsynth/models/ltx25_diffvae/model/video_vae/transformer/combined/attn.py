from __future__ import annotations

import torch
from torch import nn

from diffsynth.models.ltx25_diffvae.model.video_vae.transformer.attention import NeighborhoodAttention3D
from diffsynth.models.ltx25_diffvae.model.video_vae.transformer.rope_math import (
    h_positions,
    rot_abs_axis_impl,
    t_positions,
)


_rot_abs_axis = torch.compiler.nested_compile_region(rot_abs_axis_impl)


def _apply_nested_abs_rope_slab(
    x: torch.Tensor,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    w_pos: torch.Tensor,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    d_t, d_h, _ = rope_split
    inv_t, inv_h, inv_w = inv_freqs
    t = x.shape[1]
    h = x.shape[2]
    xt = _rot_abs_axis(x[..., :d_t], t_positions(t, x.device), inv_t, axis=1, compute_dtype=compute_dtype)
    xh = _rot_abs_axis(
        x[..., d_t : d_t + d_h],
        h_positions(h, x.device),
        inv_h,
        axis=2,
        compute_dtype=compute_dtype,
    )
    xw = _rot_abs_axis(x[..., d_t + d_h :], w_pos, inv_w, axis=3, compute_dtype=compute_dtype)
    return torch.cat([xt, xh, xw], dim=-1)


def _apply_nested_full_volume_rope(
    x: torch.Tensor,
    rope_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    num_tiles: int,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    slabs = torch.chunk(x, num_tiles, dim=3)
    w_off = 0
    parts: list[torch.Tensor] = []
    for slab in slabs:
        w_slab = slab.shape[3]
        w_pos = torch.arange(w_slab, dtype=torch.float32, device=x.device) + w_off
        parts.append(
            _apply_nested_abs_rope_slab(
                slab,
                rope_split,
                inv_freqs,
                w_pos=w_pos,
                compute_dtype=compute_dtype,
            )
        )
        w_off = w_off + w_slab
    return torch.cat(parts, dim=3)


def _qkv_nested_rope(attn: NeighborhoodAttention3D, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = attn.project_qkv(x)
    q = attn.q_norm(q) * attn.scale
    k = attn.k_norm(k)
    inv_freqs = (
        attn.rope_inv_t.to(device=x.device),
        attn.rope_inv_h.to(device=x.device),
        attn.rope_inv_w.to(device=x.device),
    )
    q = _apply_nested_full_volume_rope(
        q,
        attn.rope_dim_split,
        inv_freqs,
        num_tiles=attn.rope_num_tiles,
        compute_dtype=attn.rope_compute_dtype,
    )
    k = _apply_nested_full_volume_rope(
        k,
        attn.rope_dim_split,
        inv_freqs,
        num_tiles=attn.rope_num_tiles,
        compute_dtype=attn.rope_compute_dtype,
    )
    return q, k, v


def full(
    x: torch.Tensor,
    attn: NeighborhoodAttention3D,
    norm: nn.RMSNorm,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    y = norm(x) * (1.0 + scale) + shift
    batch, t, h, w, _ = y.shape
    kt, kh, kw = attn.kernel_size
    if t < kt or h < kh or w < kw:
        raise ValueError(
            f"3D neighborhood attention requires spatial dims >= kernel_size; "
            f"got (T,H,W)=({t},{h},{w}) vs kernel={attn.kernel_size}"
        )

    q, k, v = _qkv_nested_rope(attn, y)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = attn.attention_function(attn, q, k, v)
    out = out.reshape(batch, t, h, w, attn.dim)
    return x + attn.proj(out)
