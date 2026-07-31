# SPDX-License-Identifier: Apache-2.0
# MiniMax H3 video VAE (DiffSynth port of the checkpoint remote code).
# Asymmetric KL VAE: causal 3D-CNN encoder + ViT-3D decoder, 24-ch latent,
# 16x spatial / 4x temporal (17-frame -> 5-latent) compression.
# Consolidated from the original 15-file bundle into one module to follow
# DiffSynth's one-file-per-component convention. Section banners below mark
# the original source files; diffusers base classes are provided by the
# inlined _compat shim. Single-card only: sequence-parallel/distributed and
# torch.compile scaffolding have been removed.
from __future__ import annotations

from contextlib import nullcontext
import functools
import inspect
import math
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torchvision.transforms import Normalize

from ..core.attention import attention_forward



# ======================================================================
# ---- _compat.py ----
# ======================================================================
# Minimal diffusers-compat shim so the ported MiniMax H3 video-VAE remote code
# runs without depending on diffusers. Provides the base classes / decorator
# the code imports from diffusers, backed by torch.nn only.




class _Config(dict):
    """dict that also supports attribute access (self.config.x and self.config['x'])."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


class ModelMixin(nn.Module):
    def __init__(self):
        super().__init__()
        if "config" not in self.__dict__:
            object.__setattr__(self, "config", _Config())


def register_to_config(init):
    """Populate ``self.config`` from the __init__ signature (positional args,
    keyword args, and defaults) BEFORE running the body, so the wrapped __init__
    can freely read and mutate self.config (matching diffusers semantics)."""

    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        cfg = _Config()
        params = list(inspect.signature(init).parameters.values())[1:]  # skip self
        positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        for p, val in zip(positional, args):
            cfg[p.name] = val
        for p in params:
            if p.name in cfg:
                continue
            if p.name in kwargs:
                cfg[p.name] = kwargs[p.name]
            elif p.default is not inspect.Parameter.empty:
                cfg[p.name] = p.default
        object.__setattr__(self, "config", cfg)
        init(self, *args, **kwargs)

    return wrapper


def maybe_allow_in_graph(cls):
    return cls


# ======================================================================
# ---- parallel.py ----
# ======================================================================
# Parallel state and collective helpers for the MiniMax H3 visual VAE.


# ======================================================================
# ---- flash.py ----
# ======================================================================
# Torch-native attention implemented with PyTorch SDPA instead of FA4/CUTLASS.



_BLOCK_CAUSAL_MASK_MOD_CACHE = {}




def _as_bool_mask(mask, *, device):
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask, device=device)
    return mask.to(device=device, dtype=torch.bool)


def _ensure_nonempty_rows(mask):
    if mask.numel() == 0 or mask.shape[-1] == 0:
        return mask
    empty = ~mask.any(dim=-1)
    if empty.any():
        mask = mask.clone()
        mask[..., 0] |= empty
    return mask


def _sdpa_attention(query, key, value, causal=False, attn_mask=None):
    # query/key/value arrive as [B, S, H, D] ("b s n d"); delegate to
    # DiffSynth's unified attention_forward (flash/sage/xformers when available,
    # else torch SDPA). It rearranges to the backend layout internally.
    if attn_mask is not None and attn_mask.dim() == 3:
        attn_mask = attn_mask.unsqueeze(0)
    out = attention_forward(
        query,
        key,
        value,
        q_pattern="b s n d",
        k_pattern="b s n d",
        v_pattern="b s n d",
        out_pattern="b s n d",
        attn_mask=attn_mask,
        is_causal=causal,
    )
    return out.nan_to_num(0.0)


def _mask_mod_to_dense(mask_mod, batch, heads, q_len, kv_len, device, aux_tensors=None):
    q_idx = torch.arange(q_len, device=device).view(q_len, 1)
    kv_idx = torch.arange(kv_len, device=device).view(1, kv_len)
    dense = torch.empty((batch, heads, q_len, kv_len), dtype=torch.bool, device=device)
    for b in range(batch):
        b_idx = torch.tensor(b, device=device)
        for h in range(heads):
            h_idx = torch.tensor(h, device=device)
            mask = mask_mod(b_idx, h_idx, q_idx, kv_idx, None, aux_tensors)
            dense[b, h] = _as_bool_mask(mask, device=device)
    return _ensure_nonempty_rows(dense)


#########################################################
# Block causal attention
#########################################################


def make_block_causal_mask_mod(num_tokens, block_size, num_special=0, suffix=False):
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if num_special < 0:
        raise ValueError(f"num_special must be non-negative, got {num_special}")

    cache_key = (num_tokens, block_size, num_special, suffix)
    if cache_key in _BLOCK_CAUSAL_MASK_MOD_CACHE:
        return _BLOCK_CAUSAL_MASK_MOD_CACHE[cache_key]

    if suffix:

        def mask_mod(b, h, q_idx, kv_idx, seqlen_info, aux_tensors):
            del b, h, seqlen_info, aux_tensors
            q_is_special = q_idx >= num_tokens
            kv_is_special = kv_idx >= num_tokens
            return q_is_special | kv_is_special | (
                q_idx // block_size >= kv_idx // block_size
            )

    else:

        def mask_mod(b, h, q_idx, kv_idx, seqlen_info, aux_tensors):
            del b, h, seqlen_info, aux_tensors
            q_is_special = q_idx < num_special
            kv_is_special = kv_idx < num_special
            q_block_idx = (q_idx - num_special) // block_size
            kv_block_idx = (kv_idx - num_special) // block_size
            return q_is_special | kv_is_special | (q_block_idx >= kv_block_idx)

    mask_mod.block_sparse_cache_key = (
        "block_causal",
        num_tokens,
        block_size,
        num_special,
        suffix,
    )
    _BLOCK_CAUSAL_MASK_MOD_CACHE[cache_key] = mask_mod
    return mask_mod






#########################################################
# Public entry point
#########################################################


@torch.compiler.disable
def flash_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool = False,
    mask_mod=None,
    block_sparse=None,
    aux_tensors=None,
) -> torch.Tensor:
    use_masked = mask_mod is not None or block_sparse is not None

    if block_sparse is not None and mask_mod is None:
        raise ValueError("block_sparse requires mask_mod")
    if causal and mask_mod is not None:
        raise ValueError("causal must be encoded in mask_mod when using masked attention")
    if aux_tensors is not None and not use_masked:
        raise ValueError("aux_tensors is only supported with masked attention")

    if use_masked:
        batch, q_len, heads, _ = query.shape
        kv_len = key.shape[1]
        dense_mask = _mask_mod_to_dense(
            mask_mod,
            batch,
            heads,
            q_len,
            kv_len,
            query.device,
            aux_tensors=aux_tensors,
        )
        return _sdpa_attention(query, key, value, attn_mask=dense_mask)

    return _sdpa_attention(query, key, value, causal=causal)


# ======================================================================
# ---- func.py ----
# ======================================================================
# Token-id and rotary-embedding helpers for the MiniMax H3 visual VAE.




def create_token_ids(patch_dims, device, dtype, id_type="length_normalized", flatten=True):
    coords_list = []

    if isinstance(id_type, str):
        id_type_list = [id_type] * len(patch_dims)
    elif isinstance(id_type, list):
        id_type_list = id_type
        if len(id_type_list) != len(patch_dims):
            raise ValueError("id_type list must match patch_dims")
    else:
        raise ValueError("id_type must be a string or a list")

    if "area_normalized" in id_type_list or id_type == "area_normalized":
        raise NotImplementedError(
            "area_normalized id_type is not supported in this inference-only bundle"
        )

    for _dim_size, _id_type in zip(patch_dims, id_type_list):
        if isinstance(_dim_size, torch.Tensor):
            coords_list.append(_dim_size.to(device=device, dtype=dtype))
            continue

        if _id_type == "length_normalized":
            coords = torch.arange(0.5, _dim_size, dtype=dtype, device=device)
            coords = coords / _dim_size
            coords = 2.0 * coords - 1.0
        else:
            coords = torch.arange(_dim_size, dtype=dtype, device=device)

        coords_list.append(coords)

    coords = torch.stack(torch.meshgrid(*coords_list, indexing="ij"), dim=-1)
    if flatten:
        coords = coords.flatten(0, len(patch_dims) - 1)

    return coords.unsqueeze(0)


def _env_flag(name, default="0"):
    value = os.environ.get(name, default)
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_impl(
    t: torch.Tensor, rotary_pos_emb: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    cos, sin = rotary_pos_emb

    if cos.dim() != 4:
        raise ValueError(f"cos must be [B, N, 1, D], got {cos.shape}")

    cos = cos.to(t.dtype)
    sin = sin.to(t.dtype)

    rot_dim = cos.shape[-1]
    t_dim = t.shape[-1]

    if rot_dim < t_dim:
        t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]
        t_rot = (t_rot * cos) + (_rotate_half(t_rot) * sin)
        t = torch.cat((t_rot, t_pass), dim=-1)
    else:
        t = (t * cos) + (_rotate_half(t) * sin)

    return t


def apply_rotary_pos_emb(
    t: torch.Tensor, rotary_pos_emb: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    return _apply_rotary_pos_emb_impl(t, rotary_pos_emb)


# ======================================================================
# ---- normalize.py ----
# ======================================================================
# Pixel normalization transforms for the MiniMax H3 visual VAE.


NORM_CONFIGS = {
    "imagenet": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "simple": {
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
    },
    "raw": {
        "mean": (0.0, 0.0, 0.0),
        "std": (1.0, 1.0, 1.0),
    },
}


def get_norm_constants(norm_type: str = "imagenet") -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if norm_type not in NORM_CONFIGS:
        raise ValueError(f"Unknown norm_type: {norm_type}. Must be one of {list(NORM_CONFIGS.keys())}")
    config = NORM_CONFIGS[norm_type]
    return config["mean"], config["std"]


def get_normalize_transform(norm_type: str = "imagenet") -> Normalize:
    mean, std = get_norm_constants(norm_type)
    return Normalize(mean, std)


def get_denormalize_transform(norm_type: str = "imagenet") -> Normalize:
    mean, std = get_norm_constants(norm_type)
    inv_mean = tuple(-m / s for m, s in zip(mean, std))
    inv_std = tuple(1.0 / s for s in std)
    return Normalize(inv_mean, inv_std)


# ======================================================================
# ---- vae_module.py ----
# ======================================================================
# VAE distribution and aggregation helpers for the MiniMax H3 visual VAE.


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, upcast_fp32=True):
        if upcast_fp32:
            parameters = parameters.to(dtype=torch.float32)

        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    @torch.compiler.disable
    def sample(self, generator=None):
        noise = torch.randn(self.mean.shape, generator=generator)
        x = self.mean + self.std * noise.to(device=self.parameters.device)
        return x


class ClsTokenAggregator:
    def __init__(self, vae_model):
        self.vae = vae_model
        self.cls_tokens = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cls_tokens and hasattr(self.vae.encoder, "loss_info"):
            self.vae.encoder.loss_info["cls_token"] = torch.stack(
                self.cls_tokens, dim=0
            ).mean(dim=0)
        return False

    def collect(self):
        if (
            hasattr(self.vae.encoder, "loss_info")
            and "cls_token" in self.vae.encoder.loss_info
        ):
            self.cls_tokens.append(self.vae.encoder.loss_info["cls_token"].clone())

    def collect_stacked(self, num_tiles, sample_batch_size):
        if (
            hasattr(self.vae.encoder, "loss_info")
            and "cls_token" in self.vae.encoder.loss_info
        ):
            cls_token = self.vae.encoder.loss_info["cls_token"]
            cls_token = cls_token.unflatten(0, (num_tiles, sample_batch_size))
            self.cls_tokens.extend(token.clone() for token in cls_token)


# ======================================================================
# ---- conv.py ----
# ======================================================================
# Spatial-parallel 3D convolution for the MiniMax H3 visual VAE.





class BaseConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        padding_mode="zeros",
        padding_mode_t=None,
        causal=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode=padding_mode,
        )
        padding_mode = "constant" if padding_mode == "zeros" else padding_mode
        padding_mode_t = "constant" if padding_mode_t == "zeros" else padding_mode_t
        self.pad_mode = padding_mode
        self.pad_mode_t = padding_mode_t or ("constant" if causal else "replicate")
        self.causal = causal

    def _apply_temporal_padding(self, x):
        B, C, D, H, W = x.shape
        if D > 1:
            pad_size = (
                0,
                0,
                0,
                0,
                self.padding[0] * 2 if self.causal else self.padding[0],
                0 if self.causal else self.padding[0],
            )
            return F.pad(x, pad_size, mode=self.pad_mode_t)
        else:
            if self.pad_mode_t == "constant":
                assert self.causal, "Zeros padding is only supported for causal mode"
                zeros = torch.zeros_like(x[:, :, :1, :, :]).expand(
                    -1, -1, self.kernel_size[0] - 1, -1, -1
                )
                return torch.cat([zeros, x], dim=2)
            else:
                return x.expand(-1, -1, self.kernel_size[0], -1, -1)

    def _apply_padding(self, x):
        if sum(self.padding) == 0:
            return x

        x = F.pad(
            x,
            (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 0, 0),
            mode=self.pad_mode,
        )

        x = self._apply_temporal_padding(x)
        return x

    def forward(self, x):
        if sum(self.padding) == 0:
            return super().forward(x)

        x = self._apply_padding(x)
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
        )


# ======================================================================
# ---- norm.py ----
# ======================================================================
# Torch-native normalization for the MiniMax H3 visual VAE.




def _validate_activation(activation):
    valid_activations = {"identity", "silu", "relu"}
    if activation not in valid_activations:
        raise ValueError(
            f"Unsupported activation: {activation}. Supported: {valid_activations}"
        )


def _apply_activation(x, activation):
    _validate_activation(activation)
    if activation == "identity":
        return x
    if activation == "silu":
        return F.silu(x)
    return F.relu(x)


def _merge_time_to_batch(x):
    batch, channels, depth, height, width = x.shape
    return (
        x.permute(0, 2, 1, 3, 4)
        .contiguous()
        .view(batch * depth, channels, 1, height, width)
    )


def _split_time_from_batch(x, batch):
    batch_depth, channels, _, height, width = x.shape
    depth = batch_depth // batch
    return (
        x.view(batch, depth, channels, height, width)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )


def fused_group_norm(x, num_groups, weight, bias, eps=1e-5, activation="silu"):
    out = F.group_norm(x, num_groups, weight=weight, bias=bias, eps=eps)
    return _apply_activation(out, activation)


def fused_spatial_norm(
    f,
    num_groups,
    norm_weight,
    norm_bias,
    dynamic_scale,
    dynamic_bias,
    eps=1e-5,
    activation="silu",
):
    norm_f = F.group_norm(
        f,
        num_groups,
        weight=norm_weight,
        bias=norm_bias,
        eps=eps,
    )
    out = norm_f * dynamic_scale + dynamic_bias
    return _apply_activation(out, activation)


class DummyAffine(torch.nn.Module):
    def __init__(self, num_channels, affine=True):
        super().__init__()
        if affine:
            self.weight = torch.nn.Parameter(torch.ones(num_channels))
            self.bias = torch.nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input):
        if self.weight is None:
            return input
        shape = [1, -1] + [1] * (input.dim() - 2)
        return input * self.weight.view(*shape) + self.bias.view(*shape)


class FusedGroupNorm3D(torch.nn.Module):
    """Compatibility wrapper implemented with native PyTorch ops."""

    def __init__(
        self,
        num_groups,
        num_channels,
        eps=1e-5,
        affine=True,
        activation="silu",
        cond_channels=None,
        use_t_isolated_gn=False,
        padding_mode="zeros",
        padding_mode_t=None,
        causal=True,
    ):
        super().__init__()
        _validate_activation(activation)
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.activation = activation
        self.use_t_isolated_gn = use_t_isolated_gn

        if cond_channels is not None:
            self.use_spatial_affine = True
            self.norm_layer = DummyAffine(num_channels, affine=affine)
            self.conv_y = BaseConv3d(
                cond_channels,
                num_channels,
                kernel_size=1,
                padding_mode=padding_mode,
                padding_mode_t=padding_mode_t,
                causal=causal,
            )
            self.conv_b = BaseConv3d(
                cond_channels,
                num_channels,
                kernel_size=1,
                padding_mode=padding_mode,
                padding_mode_t=padding_mode_t,
                causal=causal,
            )
        else:
            self.use_spatial_affine = False
            if self.affine:
                self.weight = torch.nn.Parameter(torch.ones(num_channels))
                self.bias = torch.nn.Parameter(torch.zeros(num_channels))
            else:
                self.register_parameter("weight", None)
                self.register_parameter("bias", None)

    def forward(self, f, cond=None):
        need_reshape = self.use_t_isolated_gn and f.dim() == 5
        batch = f.shape[0] if need_reshape else None
        f_size = f.shape[-3:]
        if need_reshape:
            f = _merge_time_to_batch(f)

        if self.use_spatial_affine:
            scale = self.conv_y(cond)
            bias = self.conv_b(cond)
            if math.prod(scale.shape[-3:]) * math.prod(bias.shape[-3:]) > 1:
                scale = F.interpolate(scale, size=f_size, mode="nearest")
                bias = F.interpolate(bias, size=f_size, mode="nearest")
            if need_reshape:
                scale = _merge_time_to_batch(scale)
                bias = _merge_time_to_batch(bias)
            out = fused_spatial_norm(
                f,
                self.num_groups,
                self.norm_layer.weight,
                self.norm_layer.bias,
                scale,
                bias,
                self.eps,
                self.activation,
            )
        else:
            if cond is not None:
                raise NotImplementedError("Dynamic affine is not defined")
            weight = self.weight if self.affine else None
            bias = self.bias if self.affine else None
            out = fused_group_norm(
                f, self.num_groups, weight, bias, self.eps, self.activation
            )

        if need_reshape:
            out = _split_time_from_batch(out, batch)
        return out


class SpatialParallelGroupNorm(nn.GroupNorm):
    pass


class TemporalIsolatedSpatialParallelGroupNorm(SpatialParallelGroupNorm):
    def forward(self, input):
        if input.dim() == 5:
            batch = input.shape[0]
            input = _merge_time_to_batch(input)
            output = super().forward(input)
            return _split_time_from_batch(output, batch)
        return super().forward(input)








class SpatialNorm3D(nn.Module):
    def __init__(
        self,
        f_channels,
        zq_channels,
        padding_mode="zeros",
        padding_mode_t=None,
        causal=True,
        use_t_isolated_gn=False,
    ):
        super().__init__()
        norm_cls = (
            TemporalIsolatedSpatialParallelGroupNorm
            if use_t_isolated_gn
            else SpatialParallelGroupNorm
        )
        self.norm_layer = norm_cls(
            num_groups=32, num_channels=f_channels, eps=1e-6, affine=True
        )

        self.conv_y = BaseConv3d(
            zq_channels,
            f_channels,
            kernel_size=1,
            padding_mode=padding_mode,
            padding_mode_t=padding_mode_t,
            causal=causal,
        )
        self.conv_b = BaseConv3d(
            zq_channels,
            f_channels,
            kernel_size=1,
            padding_mode=padding_mode,
            padding_mode_t=padding_mode_t,
            causal=causal,
        )

    def forward(self, f, zq):
        f_size = f.shape[-3:]
        norm_f = self.norm_layer(f)
        scale = self.conv_y(zq)
        bias = self.conv_b(zq)

        if math.prod(scale.shape[-3:]) * math.prod(bias.shape[-3:]) > 1:
            scale = F.interpolate(scale, size=f_size, mode="nearest")
            bias = F.interpolate(bias, size=f_size, mode="nearest")

        return norm_f * scale + bias


def get_spatial_norm_3d(
    num_channels,
    cond_channels,
    *,
    padding_mode="zeros",
    padding_mode_t=None,
    causal=True,
    use_t_isolated_gn=False,
):
    if os.environ.get("MINIMAX_H3_USE_FUSED_NORM", "false").lower() == "true":
        return FusedGroupNorm3D(
            num_groups=32,
            num_channels=num_channels,
            eps=1e-6,
            affine=True,
            cond_channels=cond_channels,
            use_t_isolated_gn=use_t_isolated_gn,
            padding_mode=padding_mode,
            padding_mode_t=padding_mode_t,
            causal=causal,
        )
    return SpatialNorm3D(
        num_channels,
        cond_channels,
        padding_mode=padding_mode,
        padding_mode_t=padding_mode_t,
        causal=causal,
        use_t_isolated_gn=use_t_isolated_gn,
    )


def get_group_norm_3d(num_channels, use_t_isolated_gn=False):
    if os.environ.get("MINIMAX_H3_USE_FUSED_NORM", "false").lower() == "true":
        return FusedGroupNorm3D(
            num_groups=32,
            num_channels=num_channels,
            eps=1e-6,
            affine=True,
            use_t_isolated_gn=use_t_isolated_gn,
        )

    norm_cls = (
        TemporalIsolatedSpatialParallelGroupNorm
        if use_t_isolated_gn
        else SpatialParallelGroupNorm
    )
    return norm_cls(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)


# ======================================================================
# ---- attention.py ----
# ======================================================================
# Attention module for the MiniMax H3 visual VAE (inference-only bundle).




def _vit_norm_input(module, hidden_states):
    if _env_flag("MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM", "1"):
        return hidden_states.float()
    weight = getattr(module, "weight", None)
    return hidden_states.to(getattr(weight, "dtype", hidden_states.dtype))


def maybe_checkpoint(owner, function, *args):
    if owner.training and getattr(owner, "gradient_checkpointing", False):
        raise NotImplementedError(
            "gradient checkpointing is not supported in this inference-only bundle"
        )
    return function(*args)


class Attention(nn.Module):
    def __init__(
        self,
        heads,
        dim_head,
        embed_dim: Optional[int] = None,
        qk_norm_type: Optional[str] = None,
        qk_norm_affine: bool = False,
        bias: bool = True,
        out_bias: Optional[bool] = None,
        eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.attn_inner_dim = dim_head * heads
        self.embed_dim = embed_dim if embed_dim is not None else self.attn_inner_dim

        out_bias = out_bias if out_bias is not None else bias

        if qk_norm_type is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm_type == "layer_norm":
            self.norm_q = nn.LayerNorm(
                dim_head, eps=eps, elementwise_affine=qk_norm_affine
            )
            self.norm_k = nn.LayerNorm(
                dim_head, eps=eps, elementwise_affine=qk_norm_affine
            )
        elif qk_norm_type == "rms_norm":
            self.norm_q = nn.RMSNorm(
                dim_head, eps=eps, elementwise_affine=qk_norm_affine
            )
            self.norm_k = nn.RMSNorm(
                dim_head, eps=eps, elementwise_affine=qk_norm_affine
            )
        else:
            raise ValueError(
                f"unknown qk_norm_type: {qk_norm_type}. Should be None,'layer_norm','rms_norm'"
            )

        self.to_qkv = nn.Linear(self.embed_dim, self.attn_inner_dim * 3, bias=bias)

        self.to_out = nn.Linear(self.attn_inner_dim, self.embed_dim, bias=out_bias)

    def _perform_attention(self, query, key, value, pack_info):
        cu_seqlens = pack_info.get("cu_seqlens", None)
        mask_mod = pack_info.get("mask_mod", None)
        block_sparse = pack_info.get("block_sparse", None)

        if cu_seqlens is not None:
            raise NotImplementedError(
                "varlen attention is not supported in this inference-only bundle"
            )

        if mask_mod is not None:
            hidden_states = flash_attn(
                query,
                key,
                value,
                mask_mod=mask_mod,
                block_sparse=block_sparse,
            )
        else:
            hidden_states = flash_attn(
                query,
                key,
                value,
            )

        return hidden_states

    def perform_attention(self, query, key, value, pack_info={}):
        return self._perform_attention(query, key, value, pack_info)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        pack_info: dict = {},
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.to_qkv(hidden_states)
        qkv = qkv.view(batch_size, seq_len, -1, 3 * self.dim_head)
        query, key, value = torch.chunk(qkv, 3, dim=-1)

        if self.norm_q is not None:
            query = self.norm_q(_vit_norm_input(self.norm_q, query)).to(query.dtype)
        if self.norm_k is not None:
            key = self.norm_k(_vit_norm_input(self.norm_k, key)).to(key.dtype)

        if rotary_pos_emb is not None:
            query = apply_rotary_pos_emb(query, rotary_pos_emb)
            key = apply_rotary_pos_emb(key, rotary_pos_emb)

        hidden_states = self.perform_attention(query, key, value, pack_info)

        hidden_states = hidden_states.reshape(batch_size, seq_len, -1)
        hidden_states = self.to_out(hidden_states)

        return hidden_states


# ======================================================================
# ---- base_module.py ----
# ======================================================================
# Transformer building blocks for the MiniMax H3 visual VAE ViT decoder.




class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        activation_fn: str = "silu",
        bias: bool = True,
        use_gated: bool = True,
        glu_balanced: bool = False,
    ):
        super().__init__()
        ratio = 2 / 3 if (use_gated and glu_balanced) else 1
        inner_dim = round(dim * mult * ratio)
        dim_out = dim_out if dim_out is not None else dim
        self.use_gated = use_gated

        if use_gated:
            self.w1 = nn.Linear(dim, inner_dim * 2, bias=bias)
        else:
            self.w1 = nn.Linear(dim, inner_dim, bias=bias)

        if activation_fn == "silu":
            self.act_fn = nn.SiLU()
        elif activation_fn == "gelu":
            self.act_fn = nn.GELU()
        elif activation_fn == "gelu-approximate":
            self.act_fn = nn.GELU(approximate="tanh")
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

        self.w2 = nn.Linear(inner_dim, dim_out, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.w1(hidden_states)

        if self.use_gated:
            gate, hidden_states = hidden_states.chunk(2, dim=-1)
            hidden_states = self.act_fn(gate) * hidden_states
        else:
            hidden_states = self.act_fn(hidden_states)

        hidden_states = self.w2(hidden_states)
        return hidden_states


class RotaryEmbeddingND(nn.Module):
    def __init__(self, dim, rotary_base=10000, n_dim=3, use_angle=False):
        super().__init__()
        self.dim = dim
        self.n_dim = n_dim

        if dim % (2 * n_dim) != 0:
            raise ValueError(
                f"head_dim {dim} must be divisible by 2 * n_dim {2 * n_dim}"
            )

        if use_angle:
            self.angle_scale = 2.0 * math.pi
        else:
            self.angle_scale = 1.0

        inv_freq = 1 / rotary_base ** torch.arange(
            0, 1, 2 * n_dim / dim, dtype=torch.float32
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, img_ids):
        B, N, D = img_ids.shape
        if D != self.n_dim:
            raise ValueError(f"Expected {self.n_dim} dimensions, got {D}")

        with torch.autocast("cuda", enabled=False):
            angles = (
                self.angle_scale
                * img_ids[:, :, :, None]
                * self.inv_freq.to(img_ids.device)[None, None, None, :]
            )
            angles = angles.flatten(2, 3)
            angles = angles.tile(2)
            angles = angles.unsqueeze(2)

            cos = torch.cos(angles)
            sin = torch.sin(angles)

        return cos.to(dtype=img_ids.dtype), sin.to(dtype=img_ids.dtype)


@maybe_allow_in_graph
class TransformerBlock(nn.Module):
    def __init__(
        self,
        heads: int,
        dim_head: int,
        embed_dim: Optional[int] = None,
        ffn_glu_balanced: bool = False,
        norm_type: str = "layer_norm",
        norm_affine: bool = True,
        qk_norm_type: str = "rms_norm",
        qk_norm_affine: bool = False,
        ffn_activation_fn: str = "silu",
        ffn_use_gated: bool = True,
        use_scale: bool = True,
        bias: bool = True,
        eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        dim = embed_dim if embed_dim is not None else dim_head * heads
        self.use_scale = use_scale

        if norm_type == "layer_norm":
            norm_class = nn.LayerNorm
        elif norm_type == "rms_norm":
            norm_class = nn.RMSNorm
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

        self.norm1 = norm_class(
            dim,
            elementwise_affine=norm_affine,
            eps=eps,
        )
        self.attn = Attention(
            heads=heads,
            dim_head=dim_head,
            embed_dim=dim,
            qk_norm_type=qk_norm_type,
            qk_norm_affine=qk_norm_affine,
            bias=bias,
            eps=eps,
            **kwargs,
        )
        if use_scale:
            self.scale1 = nn.Parameter(torch.zeros(dim))

        self.norm2 = norm_class(
            dim,
            elementwise_affine=norm_affine,
            eps=eps,
        )
        self.ff = FeedForward(
            dim=dim,
            activation_fn=ffn_activation_fn,
            bias=bias,
            use_gated=ffn_use_gated,
            glu_balanced=ffn_glu_balanced,
        )
        if use_scale:
            self.scale2 = nn.Parameter(torch.zeros(dim))

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        rotary_pos_emb: Optional[torch.FloatTensor] = None,
        pack_info: dict = {},
    ):
        norm_hidden_states = self.norm1(_vit_norm_input(self.norm1, hidden_states)).to(hidden_states.dtype)
        attn_output = self.attn(norm_hidden_states, rotary_pos_emb, pack_info)
        if self.use_scale:
            hidden_states = hidden_states + attn_output * self.scale1
        else:
            hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(_vit_norm_input(self.norm2, hidden_states)).to(hidden_states.dtype)
        ff_output = self.ff(norm_hidden_states)
        if self.use_scale:
            hidden_states = hidden_states + ff_output * self.scale2
        else:
            hidden_states = hidden_states + ff_output

        return hidden_states


# ======================================================================
# ---- vae_cnn.py ----
# ======================================================================
# 3D causal CNN encoder for the MiniMax H3 visual VAE (inference-only bundle).











# ============================================================================
# 3D CNN Components
# ============================================================================


def norm_silu(x, norm, cond=None):
    if cond is None:
        return F.silu(norm(x))
    else:
        return F.silu(norm(x, cond))


class Downsample3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_stride=1,
        space_stride=2,
        padding_mode="zeros",
        padding_mode_t=None,
        causal=True,
    ):
        super().__init__()
        self.time_stride = time_stride
        self.space_stride = space_stride

        assert time_stride in [1, 2]
        assert space_stride in [1, 2, 3]

        self.conv = BaseConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=(1, 0, 0),
            stride=(time_stride, space_stride, space_stride),
            padding_mode=padding_mode,
            padding_mode_t=padding_mode_t,
            causal=causal,
        )
        self.causal = self.conv.causal
        self.pad_mode = self.conv.pad_mode

    def forward(self, x):
        if self.space_stride == 2:
            pad = (0, 1, 0, 1, 0, 0)
            x = F.pad(x, pad, mode=self.pad_mode)
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        zq_ch=None,
        padding_mode="zeros",
        padding_mode_t=None,
        causal=True,
        use_t_isolated_gn=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.use_fused_norm = (
            os.environ.get("MINIMAX_H3_USE_FUSED_NORM", "false").lower() == "true"
        )

        if zq_ch is None:
            self.norm1 = get_group_norm_3d(in_channels, use_t_isolated_gn=use_t_isolated_gn)
            self.norm2 = get_group_norm_3d(out_channels, use_t_isolated_gn=use_t_isolated_gn)
        else:
            self.norm1 = get_spatial_norm_3d(
                in_channels,
                zq_ch,
                padding_mode=padding_mode,
                padding_mode_t=padding_mode_t,
                causal=causal,
                use_t_isolated_gn=use_t_isolated_gn,
            )
            self.norm2 = get_spatial_norm_3d(
                out_channels,
                zq_ch,
                padding_mode=padding_mode,
                padding_mode_t=padding_mode_t,
                causal=causal,
                use_t_isolated_gn=use_t_isolated_gn,
            )

        self.conv1 = BaseConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
            padding_mode_t=padding_mode_t,
            causal=causal,
        )

        self.conv2 = BaseConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
            padding_mode_t=padding_mode_t,
            causal=causal,
        )

        if self.in_channels != self.out_channels:
            self.nin_shortcut = BaseConv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding_mode=padding_mode,
                padding_mode_t=padding_mode_t,
                causal=causal,
            )

    def forward(self, x, zq=None):
        h = x

        if self.use_fused_norm:
            h = self.norm1(h, zq)
        else:
            h = norm_silu(h, self.norm1, zq)

        h = self.conv1(h)

        if self.use_fused_norm:
            h = self.norm2(h, zq)
        else:
            h = norm_silu(h, self.norm2, zq)

        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class EncoderFCN3D(nn.Module):
    def __init__(
        self,
        ch,
        ch_mult,
        space_down,
        time_down,
        num_res_blocks,
        in_channels,
        z_channels,
        double_z=False,
        zq_ch=None,
        padding_mode="zeros",
        padding_mode_t=None,
        causal=True,
        use_t_isolated_gn=False,
    ):
        super().__init__()
        self.ch = ch
        self.num_levels = len(ch_mult)

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = [num_res_blocks] * self.num_levels
        else:
            self.num_res_blocks = num_res_blocks

        self.space_down_factors = space_down
        self.time_down_factors = time_down
        self.in_channels = in_channels

        self.use_fused_norm = (
            os.environ.get("MINIMAX_H3_USE_FUSED_NORM", "false").lower() == "true"
        )

        block_mid = [ch * ch_mult[i] for i in range(self.num_levels)]
        block_in = [block_mid[0]] + block_mid[:-1]
        block_out = block_mid

        conv_kwargs = dict(
            padding_mode=padding_mode,
            padding_mode_t=padding_mode_t,
            causal=causal,
        )

        self.conv_in = BaseConv3d(
            in_channels, block_in[0], kernel_size=3, padding=1, **conv_kwargs
        )

        self.down = nn.ModuleList()
        for i_level in range(self.num_levels):
            down = nn.Module()

            down.block = nn.ModuleList()
            for i in range(self.num_res_blocks[i_level]):
                down.block.append(
                    ResnetBlock3D(
                        in_channels=block_in[i_level] if i == 0 else block_mid[i_level],
                        out_channels=block_mid[i_level],
                        zq_ch=zq_ch,
                        use_t_isolated_gn=use_t_isolated_gn,
                        **conv_kwargs,
                    )
                )

            if space_down[i_level] * time_down[i_level] > 1:
                down.downsample = Downsample3D(
                    block_mid[i_level],
                    block_out[i_level],
                    time_stride=time_down[i_level],
                    space_stride=space_down[i_level],
                    **conv_kwargs,
                )
            else:
                if block_out[i_level] != block_mid[i_level]:
                    down.downsample = BaseConv3d(
                        block_mid[i_level],
                        block_out[i_level],
                        kernel_size=1,
                        **conv_kwargs,
                    )

            self.down.append(down)

        if zq_ch is None:
            self.norm_out = get_group_norm_3d(
                block_out[-1], use_t_isolated_gn=use_t_isolated_gn
            )
        else:
            self.norm_out = get_spatial_norm_3d(
                block_out[-1],
                zq_ch,
                use_t_isolated_gn=use_t_isolated_gn,
                **conv_kwargs,
            )

        self.conv_out = BaseConv3d(
            block_out[-1],
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            padding=1,
            **conv_kwargs,
        )

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(self, x, zq=None):
        h = self.conv_in(x)
        for i_level in range(self.num_levels):
            for i_block in range(self.num_res_blocks[i_level]):
                h = maybe_checkpoint(self, self.down[i_level].block[i_block], h, zq)
            if hasattr(self.down[i_level], "downsample"):
                h = self.down[i_level].downsample(h)

        if self.use_fused_norm:
            h = self.norm_out(h, zq)
        else:
            h = norm_silu(h, self.norm_out, zq)

        h = self.conv_out(h)
        return h


# ======================================================================
# ---- vae_vit.py ----
# ======================================================================
# ViT3D decoder for the MiniMax H3 visual VAE (inference-only bundle).




def _linear_with_module_dtype(linear, tensor, out_dtype=None):
    weight = getattr(linear, "weight", None)
    target_dtype = getattr(weight, "dtype", tensor.dtype)
    output = linear(tensor.to(target_dtype))
    if out_dtype is not None and output.dtype != out_dtype:
        output = output.to(out_dtype)
    return output


def _pack_tensors_3d(tensors, patch_size, patch_size_t):
    batch_size, num_channels_tensors, temporal, height, width = tensors.shape

    tensors = tensors.view(
        batch_size,
        num_channels_tensors,
        temporal // patch_size_t,
        patch_size_t,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    tensors = tensors.permute(0, 2, 4, 6, 1, 3, 5, 7)
    tensors = tensors.reshape(
        batch_size,
        (temporal // patch_size_t) * (height // patch_size) * (width // patch_size),
        num_channels_tensors * patch_size_t * patch_size * patch_size,
    )
    return tensors


def _unpack_tensors_3d(tensors, patch_size, patch_size_t, temporal, height, width):
    batch_size, num_patches, channels = tensors.shape
    num_channels_tensors = channels // (patch_size_t * patch_size * patch_size)

    tensors = tensors.view(
        batch_size,
        temporal // patch_size_t,
        height // patch_size,
        width // patch_size,
        num_channels_tensors,
        patch_size_t,
        patch_size,
        patch_size,
    )
    tensors = tensors.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    tensors = tensors.reshape(batch_size, num_channels_tensors, temporal, height, width)
    return tensors


class ViTBase(ModelMixin):
    """Base class for ViT Encoder and Decoder with common functionality."""

    _supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]
    gradient_checkpointing_mode = "full"

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def _init_weights(self):
        def basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(basic_init)

    def init_mask_config(self, dim, is_3d=False):
        self._mask_dim = dim
        self._mask_is_3d = is_3d
        self.register_buffer("mask_token", torch.zeros(1, 1, dim))

    def set_mask_config(self, mask_config):
        self.mask_prob = mask_config.get("mask_prob", 0.0)
        self.mask_enabled = self.mask_prob > 0
        self.mask_style = mask_config.get("mask_style", "replace")
        if self.mask_enabled and self.mask_style == "drop" and self.mask_prob < 1.0:
            pass
        if self._mask_is_3d:
            self.temporal_scale_range = mask_config.get("temporal_scale_range", (0.3, 0.5))
            self.spatial_scale_range = mask_config.get("spatial_scale_range", (0.1, 0.25))
            self.min_mask_ratio = mask_config.get("min_mask_ratio", 0.75)
            self.max_mask_ratio = mask_config.get("max_mask_ratio", 0.95)
        else:
            self.spatial_scale_range = mask_config.get("spatial_scale_range", (0.15, 0.15))
            self.min_mask_ratio = mask_config.get("min_mask_ratio", 0.5)
            self.max_mask_ratio = mask_config.get("max_mask_ratio", 0.75)
        self.aspect_ratio_range = mask_config.get("aspect_ratio_range", (0.75, 1.5))
        self.max_retries = mask_config.get("max_retries", 100)
        if self.mask_enabled and self.mask_style == "drop" and getattr(self, "t_causal", False):
            pass
        if self.mask_enabled and "mask_token" in self._buffers:
            del self._buffers["mask_token"]
            self.mask_token = nn.Parameter(torch.randn(1, 1, self._mask_dim) * 0.02)

    def init_suffix_tokens(self, dim, num_register_tokens, has_cls_token=True):
        self.num_register_tokens = num_register_tokens
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.randn(1, num_register_tokens, dim) * 0.02)
        else:
            self.register_tokens = None
        if has_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def apply_mask_preprocess(self, hidden_states, img_ids, patch_dims, num_suffix):
        if self.training and self.mask_enabled:
            raise NotImplementedError(
                "mask modeling is not supported in this inference-only bundle"
            )
        return hidden_states, img_ids

    def forward_transformer_blocks(self, hidden_states, rotary_pos_emb, pack_info=None):
        if pack_info is None:
            pack_info = {}
        for block in self.transformer_blocks:
            hidden_states = maybe_checkpoint(
                self, block, hidden_states, rotary_pos_emb, pack_info
            )
        return hidden_states

    def apply_mask_postprocess(self, hidden_states, num_patches):
        if self.training and self.mask_enabled and self.mask_style == "drop":
            raise NotImplementedError(
                "mask modeling is not supported in this inference-only bundle"
            )
        return hidden_states








class ViT3DDecoder(ViTBase):
    """Vision Transformer Video Decoder using TransformerBlock."""

    @register_to_config
    def __init__(
        self,
        patch_size: int = 16,
        patch_size_t: int = 4,
        t_causal: bool = False,
        in_channels: int = 16,
        out_channels: int = 3,
        num_layers: int = 24,
        heads: int = 16,
        dim_head: int = 64,
        norm_type: str = "layer_norm",
        norm_affine: bool = True,
        qk_norm_type: str = None,
        qk_norm_affine: bool = False,
        ffn_activation_fn: str = "gelu",
        ffn_use_gated: bool = False,
        rope_theta: float = 100.0,
        rope_dim_ratio: float = 1.0,
        bias: bool = True,
        eps: float = 1e-5,
        num_register_tokens: int = 4,
        mask_config: dict = {},
        **kwargs,
    ):
        super().__init__()

        dim = heads * dim_head
        rope_apply_dim = int(dim_head * rope_dim_ratio)

        self.pos_embed = RotaryEmbeddingND(rope_apply_dim, rope_theta, n_dim=3, use_angle=True)

        self.x_embedder = nn.Linear(in_channels, dim)

        self.init_suffix_tokens(dim, num_register_tokens, has_cls_token=False)

        self.t_causal = t_causal

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    heads=heads,
                    dim_head=dim_head,
                    norm_type=norm_type,
                    norm_affine=norm_affine,
                    qk_norm_type=qk_norm_type,
                    qk_norm_affine=qk_norm_affine,
                    ffn_activation_fn=ffn_activation_fn,
                    ffn_use_gated=ffn_use_gated,
                    bias=bias,
                    eps=eps,
                    **kwargs,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(dim, elementwise_affine=norm_affine, eps=eps)
        patch_dim = out_channels * patch_size_t * patch_size * patch_size
        self.proj_out = nn.Linear(dim, patch_dim)

        self.init_mask_config(dim, is_3d=True)
        self.set_mask_config(mask_config)

        self._init_weights()
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.loss_info = {}

        B, C, latent_T, latent_H, latent_W = x.shape
        patch_size = self.config.patch_size
        patch_size_t = self.config.patch_size_t
        num_suffix = 1 + self.num_register_tokens

        hidden_states = _pack_tensors_3d(x, 1, 1)
        latent_size = (latent_T, latent_H, latent_W)

        with torch.autocast("cuda", enabled=False):
            hidden_states = _linear_with_module_dtype(self.x_embedder, hidden_states, hidden_states.dtype)

        num_patches = hidden_states.shape[1]

        tokens = [hidden_states]

        if self.register_tokens is not None:
            register_tokens = self.register_tokens.expand(B, -1, -1)
            tokens.append(register_tokens)

        cls_token = torch.zeros_like(hidden_states[:, 0:1, :])
        tokens.append(cls_token)
        hidden_states = torch.cat(tokens, dim=1)

        patch_dims = [latent_T, latent_H, latent_W]
        img_ids = create_token_ids(latent_size, x.device, x.dtype).expand(B, -1, -1)
        suffix_ids = torch.zeros((B, num_suffix, 3), device=x.device, dtype=img_ids.dtype)
        img_ids = torch.cat([img_ids, suffix_ids], dim=1)

        hidden_states, img_ids = self.apply_mask_preprocess(hidden_states, img_ids, patch_dims, num_suffix)

        pack_info = {}
        if self.t_causal:
            spatial_size = latent_H * latent_W
            mask_mod = make_block_causal_mask_mod(
                num_tokens=num_patches,
                block_size=spatial_size,
                suffix=True,
            )
            pack_info["mask_mod"] = mask_mod

        rotary_pos_emb = self.pos_embed(img_ids)

        for block in self.transformer_blocks:
            hidden_states = maybe_checkpoint(
                self, block, hidden_states, rotary_pos_emb, pack_info
            )

        hidden_states = self.norm_out(hidden_states)

        hidden_states = self.apply_mask_postprocess(hidden_states, num_patches)

        with torch.autocast("cuda", enabled=False):
            output = _linear_with_module_dtype(self.proj_out, hidden_states, hidden_states.dtype)

        output = output[:, :num_patches, :]

        video_t = latent_size[0] * patch_size_t
        video_h = latent_size[1] * patch_size
        video_w = latent_size[2] * patch_size
        output = _unpack_tensors_3d(output, patch_size, patch_size_t, video_t, video_h, video_w)

        return output


# ======================================================================
# ---- vae_processor.py ----
# ======================================================================
# Tensor pre/post-processing for the MiniMax H3 visual VAE.




class VAEProcessor:

    def __init__(
        self,
        *,
        vae_ratio,
        vae_ratio_t,
        clip_length,
        frame_overlap,
        token_overlap,
        tokens_chunk_size,
        isolated_last_frame,
        latent_patch_size,
        crop_mode,
        pixel_norm_type="imagenet",
        transform=None,
        transform_rev=None,
        use_3d_conv=False,
    ):
        self.vae_ratio = vae_ratio
        self.vae_ratio_t = vae_ratio_t
        self.clip_length = clip_length
        self.frame_overlap = frame_overlap
        self.token_overlap = token_overlap
        self.tokens_chunk_size = tokens_chunk_size
        self.isolated_last_frame = isolated_last_frame
        self.latent_patch_size = latent_patch_size
        self.crop_mode = crop_mode
        self.transform = transform or get_normalize_transform(pixel_norm_type)
        self.transform_rev = transform_rev or get_denormalize_transform(pixel_norm_type)
        self.use_3d_conv = use_3d_conv

    def _ensure_list(self, data):
        return data if isinstance(data, list) else [data]

    def _align_to_total_patch_size(self, h, w):
        total_patch_size = self.latent_patch_size * self.vae_ratio
        new_h = (h // total_patch_size) * total_patch_size
        new_w = (w // total_patch_size) * total_patch_size
        return new_h, new_w

    def _crop_to_align(self, tensor, new_h, new_w, is_video=False):
        if is_video:
            _, _, _, h, w = tensor.shape
        else:
            _, _, h, w = tensor.shape

        if self.crop_mode == "center":
            top = (h - new_h) // 2
            left = (w - new_w) // 2
        else:
            top = 0
            left = 0

        if is_video:
            return tensor[:, :, :, top : top + new_h, left : left + new_w]
        else:
            return tensor[:, :, top : top + new_h, left : left + new_w]

    def _align_target_token(self, T, mode):
        intra_tail = self.clip_length % self.vae_ratio_t
        min_frames = intra_tail or self.vae_ratio_t
        full_chunks = T // self.clip_length
        remainder = T % self.clip_length

        if remainder == 0:
            return max(T, min_frames)

        if mode == "pad":
            aligned_r = (
                math.ceil((remainder - intra_tail) / self.vae_ratio_t) * self.vae_ratio_t
                + intra_tail
            )
            if aligned_r > self.clip_length:
                return (full_chunks + 1) * self.clip_length + intra_tail
            return full_chunks * self.clip_length + aligned_r
        else:  # trim
            k = (remainder - intra_tail) // self.vae_ratio_t
            if k >= 0:
                target = full_chunks * self.clip_length + k * self.vae_ratio_t + intra_tail
                return max(target, min_frames)
            elif full_chunks > 0:
                return full_chunks * self.clip_length
            else:
                return min_frames

    def _align_target(self, T, mode, granularity):
        if granularity == "chunk":
            step = self.clip_length
            tail = self.frame_overlap
            if self.isolated_last_frame:
                tail += 1

            k = math.ceil((T - tail) / step) if mode == "pad" else (T - tail) // step
            return max(k, 1) * step + tail

        isolated_extra = 1 if self.isolated_last_frame else 0
        return self._align_target_token(T - isolated_extra, mode) + isolated_extra

    def align_video_length(self, video_length, mode="pad", granularity="chunk"):
        target = self._align_target(video_length, mode, granularity)
        delta = target - video_length
        if delta > 0 and mode == "trim":
            raise ValueError(
                f"Cannot trim {video_length} frames to valid length {target}: "
                f"not enough frames (granularity={granularity})"
            )
        return delta

    def align_video_length_2pass(self, video_length):
        """Return the leading/trailing frame pads and trailing latent drop.

        This is the continuation-prefix (2-pass) alignment.  The caller temporarily disables the model's normal token
        drop and keeps these mirrored processor fields at zero.
        """
        if self.isolated_last_frame:
            raise ValueError(
                "align_video_length_2pass does not support isolated_last_frame"
            )
        if self.token_overlap != 0 or self.frame_overlap != 0:
            raise ValueError(
                "align_video_length_2pass requires token_drop=0 alignment"
            )

        leading = self.align_video_length(
            video_length, mode="pad", granularity="token"
        )
        token_aligned = video_length + leading
        trailing = self.align_video_length(
            token_aligned, mode="pad", granularity="chunk"
        )

        if trailing > 0:
            intra_tail = self.clip_length % self.vae_ratio_t
            full_chunks = token_aligned // self.clip_length
            remainder = token_aligned % self.clip_length
            real_tokens = full_chunks * self.tokens_chunk_size
            if remainder > 0:
                real_tokens += (
                    (remainder - intra_tail) // self.vae_ratio_t + 1
                )
            drop_tokens = (
                self.get_latent_length(token_aligned + trailing) - real_tokens
            )
        else:
            drop_tokens = 0

        return leading, trailing, drop_tokens

    def get_suitable_video_length(self, video_length, verbose=False):
        used_frame_length = video_length + self.align_video_length(
            video_length, mode="trim", granularity="chunk"
        )
        if verbose:
            pass
        return used_frame_length

    def get_latent_length(self, video_length):
        tail_frame = self.frame_overlap
        tail_token = self.token_overlap
        if self.isolated_last_frame:
            tail_frame += 1
            tail_token += 1

        video_length = self.get_suitable_video_length(video_length)
        latent_length = (
            int((video_length - tail_frame) // self.clip_length)
            * self.tokens_chunk_size
            + tail_token
        )
        return latent_length



    def transform_tensor(self, tensor):
        B, T = None, None
        if tensor.ndim == 5:
            if tensor.shape[2] == 3:
                tensor = tensor.transpose(1, 2)
            B, _, T, _, _ = tensor.shape
            tensor = rearrange(tensor, "b c t h w -> (b t) c h w")
        elif tensor.ndim == 4:
            if tensor.shape[0] == 3:
                tensor = tensor.transpose(0, 1)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

        tensor = self.transform(tensor)

        if B is not None and T is not None:
            tensor = rearrange(tensor, "(b t) c h w -> b c t h w", b=B, t=T)

        return tensor.contiguous()

    def revert_tensor(self, tensor):
        B, T = None, None
        if self.use_3d_conv:
            tensor = tensor.unsqueeze(2) if tensor.ndim == 4 else tensor
            B, _, T, _, _ = tensor.shape
            tensor = rearrange(tensor, "b c t h w -> (b t) c h w")
        tensor_rev = self.transform_rev(tensor).clamp(0, 1)
        if B is not None:
            tensor_rev = rearrange(tensor_rev, "(b t) c h w -> b c t h w", b=B, t=T)
        return tensor_rev.contiguous()

    @staticmethod
    def convert_numpy_to_tensor(numpy_array, device=None):
        if isinstance(numpy_array, list):
            numpy_array = np.stack(numpy_array, axis=0)
        numpy_array = numpy_array.astype(np.float32)
        tensor = torch.from_numpy(numpy_array)
        tensor = tensor.permute(0, 3, 1, 2)
        tensor = tensor / 255.0
        if device is not None:
            tensor = tensor.to(device)
        return tensor


# ======================================================================
# ---- klvae.py ----
# ======================================================================
# MiniMax H3 visual VAE: 3D causal CNN encoder + ViT3D decoder (inference-only bundle).





def _resolve_temporal_cat_dtype():
    raw = os.environ.get("MINIMAX_H3_VAE_DECODER_TEMPORAL_CAT_DTYPE", "").strip().lower()
    if raw in ("", "0", "false", "no", "off", "none", "keep", "default"):
        return None
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if raw not in mapping:
        raise ValueError(
            "MINIMAX_H3_VAE_DECODER_TEMPORAL_CAT_DTYPE must be one of "
            "fp16|bf16|fp32|keep, got %r" % raw
        )
    return mapping[raw]


def _resolve_temporal_stream_cat():
    raw = os.environ.get("MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT", "1").strip().lower()
    return raw not in ("0", "false", "no", "off", "disable", "disabled")


class AutoencoderKL(ModelMixin):
    r"""
    Abstract shared base for the MiniMax H3 visual VAE.

    This class only carries the shared inference machinery (temporal
    chunking, tiling, encode/decode entry points). Instantiate the concrete
    subclass ``AutoencoderKLLegacy`` via ``from_pretrained`` instead.
    """

    _supports_gradient_checkpointing = True
    _compilable_modules = ["encoder", "decoder"]
    _deprecated_kwargs = [
        "clip_length",
        "token_drop",
        "isolated_first_frame",
        "isolated_last_frame",
        "isolated_key_frame",
        "encoder_tiling",
        "decoder_tiling",
        "parallel_tiling",
        "stack_tiling",
        "tile_size",
        "tile_overlap_min",
        "decoder_tile_size",
        "decoder_tile_overlap_min",
        "latent_patch_size",
        "crop_mode",
        "encoder_parallel",
        "decoder_parallel",
        "chunk_dim",
    ]  # legacy config keys accepted by from_pretrained for checkpoint compatibility


    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def _freeze_nested_module(self, module_path):
        parts = module_path.split(".")
        module = self
        for part in parts:
            module = getattr(module, part)
        module.requires_grad_(False)

    def setup_forward(self, **kwargs):
        self.clip_length = kwargs.get("clip_length", 17)
        self.token_drop = kwargs.get("token_drop", 0)
        self.frame_drop = self.token_drop * self.vae_ratio_t
        self.frame_pre_padding = (-self.clip_length) % self.vae_ratio_t
        self.tokens_chunk_size = math.ceil(self.clip_length / self.vae_ratio_t)
        self.token_overlap = (-self.token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(self.token_overlap * self.vae_ratio_t - self.frame_pre_padding, 0)
        self.isolated_first_frame = kwargs.get("isolated_first_frame", False)
        self.isolated_last_frame = kwargs.get("isolated_last_frame", False)
        self.isolated_key_frame = kwargs.get("isolated_key_frame", False)

        self.encoder_tiling = kwargs.get("encoder_tiling", False)
        self.decoder_tiling = kwargs.get("decoder_tiling", False)
        self.stack_tiling = kwargs.get("stack_tiling", False)
        self.tile_size = kwargs.get("tile_size", 256)
        self.tile_overlap_min = kwargs.get("tile_overlap_min", 64)
        self.decoder_tile_size = kwargs.get("decoder_tile_size", self.tile_size)
        self.decoder_tile_overlap_min = kwargs.get("decoder_tile_overlap_min", self.tile_overlap_min)
        self.latent_patch_size = kwargs.get("latent_patch_size", 1)
        self.crop_mode = kwargs.get("crop_mode", "top_left")
        self.pixel_norm_type = kwargs.get("pixel_norm_type", "imagenet")

        # spatial parallel mode
        if hasattr(self, "_sp_initialized"):
            if (
                kwargs.get("chunk_dim", -1) != self.chunk_dim
                or kwargs.get("encoder_parallel", False) != self.encoder_parallel
                or kwargs.get("decoder_parallel", False) != self.decoder_parallel
                or kwargs.get("parallel_tiling", False) != self.parallel_tiling
            ):
                pass
        else:
            self.chunk_dim = kwargs.get("chunk_dim", -1)
            self.encoder_parallel = kwargs.get("encoder_parallel", False)
            self.decoder_parallel = kwargs.get("decoder_parallel", False)
            self.parallel_tiling = kwargs.get("parallel_tiling", False)
            self._sp_initialized = True

        processor_kwargs = {
            "vae_ratio": self.vae_ratio,
            "vae_ratio_t": self.vae_ratio_t,
            "clip_length": self.clip_length,
            "frame_overlap": self.frame_overlap,
            "token_overlap": self.token_overlap,
            "tokens_chunk_size": self.tokens_chunk_size,
            "isolated_last_frame": self.isolated_last_frame,
            "latent_patch_size": self.latent_patch_size,
            "crop_mode": self.crop_mode,
            "pixel_norm_type": self.pixel_norm_type,
            "transform": self.transform,
            "transform_rev": self.transform_rev,
            "use_3d_conv": self.use_3d_conv,
        }
        if hasattr(self, "processor"):
            for key, value in processor_kwargs.items():
                setattr(self.processor, key, value)
        else:
            self.processor = VAEProcessor(**processor_kwargs)

    def split_tiles(self, input_len, is_decoder=False):
        tile_size = self.decoder_tile_size if is_decoder else self.tile_size
        tile_overlap_min = self.decoder_tile_overlap_min if is_decoder else self.tile_overlap_min

        if tile_size >= input_len:
            return [0], [input_len], []

        N = math.ceil(input_len / tile_size)
        while True:
            overlaps = [tile_overlap_min] * (N - 1)
            remaining = tile_size * N - sum(overlaps) - input_len

            if remaining < 0:
                N += 1
            else:
                break

        remaining_units = remaining // self.vae_ratio
        for i in range(remaining_units):
            overlaps[i % (N - 1)] += self.vae_ratio

        tile_start_idx = [0]
        for i in range(N - 1):
            tile_start_idx.append(tile_start_idx[-1] + tile_size - overlaps[i])

        tile_len = [tile_size] * N
        return tile_start_idx, tile_len, overlaps

    def blend(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int, dim: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)

        positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
        weight_a = 1 - positions / blend_extent
        weight_b = positions / blend_extent

        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = weight_a.view(shape)
        weight_b = weight_b.view(shape)

        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        a_overlap = a[tuple(slice_a)]

        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)
        b_overlap = b[tuple(slice_b)]

        blended = a_overlap * weight_a + b_overlap * weight_b

        if blend_extent < b.shape[dim]:
            slice_b_rest = [slice(None)] * b.ndim
            slice_b_rest[dim] = slice(blend_extent, None)
            b_rest = b[tuple(slice_b_rest)]
            return torch.cat([blended, b_rest], dim=dim)
        else:
            return blended

    def _run_tile_tasks(self, tiles, tile_indices, forward_fn, stack_tiling, cls_agg=None):
        if stack_tiling and tile_indices:
            sample_batch_size = tiles[0].shape[0]
            tile_batch = torch.cat([tiles[idx] for idx in tile_indices], dim=0)
            output_batch = forward_fn(tile_batch)
            output_tiles = output_batch.unflatten(
                0, (len(tile_indices), sample_batch_size)
            ).unbind(dim=0)
            if cls_agg is not None:
                cls_agg.collect_stacked(len(tile_indices), sample_batch_size)
            return list(output_tiles)

        tasks = []
        for idx in tile_indices:
            tasks.append(forward_fn(tiles[idx]))
            if cls_agg is not None:
                cls_agg.collect()
        return tasks

    def tiled_encode(self, x):
        height, width = x.shape[-2], x.shape[-1]
        y_idx, y_len, y_overlap = self.split_tiles(height, False)
        x_idx, x_len, x_overlap = self.split_tiles(width, False)

        i_max, j_max = len(y_idx), len(x_idx)
        num_tiles = i_max * j_max

        x_tiles = []
        for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
            for j, (j_pos, j_len) in enumerate(zip(x_idx, x_len)):
                tile = x[..., i_pos : i_pos + i_len, j_pos : j_pos + j_len]
                x_tiles.append(tile)

        with ClsTokenAggregator(self) as agg:
            local_tile_indices = list(range(num_tiles))
            stack_tiling = self.stack_tiling and not (
                self.training and getattr(self.encoder, "mask_enabled", False)
            )
            encoded_tasks = self._run_tile_tasks(
                x_tiles, local_tile_indices, self.encode, stack_tiling, agg
            )
            all_encoded = encoded_tasks

        rows = [[None for _ in range(j_max)] for _ in range(i_max)]
        for idx, encoded in enumerate(all_encoded):
            i, j = idx // j_max, idx % j_max
            rows[i][j] = encoded.to(x.device)

        latent_y_overlap = [
            tile_overlap // self.vae_ratio for tile_overlap in y_overlap
        ]
        latent_x_overlap = [
            tile_overlap // self.vae_ratio for tile_overlap in x_overlap
        ]

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend(rows[i - 1][j], tile, latent_y_overlap[i - 1], dim=-2)
                if j > 0:
                    tile = self.blend(row[j - 1], tile, latent_x_overlap[j - 1], dim=-1)
                if i < len(rows) - 1:
                    tile = tile[..., : -latent_y_overlap[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, : -latent_x_overlap[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        z = torch.cat(result_rows, dim=-2)

        return z

    def tiled_decode(self, z):
        height, width = (
            z.shape[-2] * self.vae_ratio,
            z.shape[-1] * self.vae_ratio,
        )
        y_idx, y_len, y_overlap = self.split_tiles(height, True)
        x_idx, x_len, x_overlap = self.split_tiles(width, True)

        i_max, j_max = len(y_idx), len(x_idx)
        num_tiles = i_max * j_max

        z_tiles = []
        for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
            i_pos, i_len = (
                i_pos // self.vae_ratio,
                i_len // self.vae_ratio,
            )
            for j, (j_pos, j_len) in enumerate(zip(x_idx, x_len)):
                j_pos, j_len = (j_pos // self.vae_ratio, j_len // self.vae_ratio)
                tile = z[..., i_pos : i_pos + i_len, j_pos : j_pos + j_len]
                z_tiles.append(tile)

        local_tile_indices = list(range(num_tiles))
        stack_tiling = self.stack_tiling and not (
            self.training and getattr(self.decoder, "mask_enabled", False)
        )
        decoded_tasks = self._run_tile_tasks(
            z_tiles, local_tile_indices, self.decode, stack_tiling
        )
        all_decoded = decoded_tasks


        rows = [[None for _ in range(j_max)] for _ in range(i_max)]
        for idx, decoded in enumerate(all_decoded):
            i, j = idx // j_max, idx % j_max
            rows[i][j] = decoded.to(z.device)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend(rows[i - 1][j], tile, y_overlap[i - 1], dim=-2)
                if j > 0:
                    tile = self.blend(row[j - 1], tile, x_overlap[j - 1], dim=-1)
                if i < len(rows) - 1:
                    tile = tile[..., : -y_overlap[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, : -x_overlap[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        dec = torch.cat(result_rows, dim=-2)
        return dec

    def _adaptive_encode(self, x):
        if self.encoder_tiling:
            return self.tiled_encode(x)
        else:
            return self.encode(x)

    def _adaptive_decode(self, z):
        if self.decoder_tiling:
            return self.tiled_decode(z)
        else:
            return self.decode(z)

    def trim_code(self, z, target_codes):
        if target_codes < z.shape[2]:
            if self.causal_encoder:
                z = z[:, :, -target_codes:, :, :]
            else:
                start_frame = (z.shape[2] - target_codes) // 2
                z = z[:, :, start_frame : start_frame + target_codes, :, :]
        return z

    def trim_output(self, dec, target_frames):
        if target_frames < dec.shape[2]:
            if self.causal_encoder:  # This is defined by encoder, not decoder
                dec = dec[:, :, -target_frames:, :, :]
            else:
                start_frame = (dec.shape[2] - target_frames) // 2
                dec = dec[:, :, start_frame : start_frame + target_frames, :, :]
        return dec

    def encode_temporal(self, x):
        offset_frame = 1 if self.isolated_first_frame and self.frame_pre_padding == 0 else 0

        if x.shape[2] % self.clip_length != offset_frame:
            pad_size = (offset_frame - x.shape[2]) % self.clip_length
            pad_frames = x[:, :, -1:].repeat(1, 1, pad_size, 1, 1)
            x = torch.cat([x, pad_frames], dim=2)

        num_chunks = (x.shape[2] - offset_frame) // self.clip_length

        z_list = []
        for i in range(num_chunks):
            start_idx = i * self.clip_length + offset_frame
            end_idx = (i + 1) * self.clip_length + offset_frame
            clip_x = x[:, :, start_idx:end_idx, :, :]

            if self.isolated_key_frame:
                key_frame = clip_x[:, :, :1, :, :]
                z_key = self._adaptive_encode(key_frame)

                if clip_x.shape[2] > 1:
                    video_frames = clip_x[:, :, 1:, :, :]
                    z_video = self._adaptive_encode(video_frames)
                    z = torch.cat([z_key, z_video], dim=2)
                else:
                    z = z_key
            else:
                z = self._adaptive_encode(clip_x)

            z_list.append(z)

        z = torch.cat(z_list, dim=2)
        if self.token_drop > 0:
            z = z[:, :, : -self.token_drop]

        if self.isolated_first_frame:
            input_first_frame = x[:, :, :1, :, :]
            z_first_frame = self._adaptive_encode(input_first_frame)

            if self.frame_pre_padding == 0:
                z = torch.cat([z_first_frame, z], dim=2)
            else:
                z = torch.cat([z_first_frame, z[:, :, 1:, :, :]], dim=2)

        if self.isolated_last_frame:
            frame_num = x.shape[2]
            last_frame_idx = frame_num - self.frame_drop + offset_frame
            input_last_frame = x[:, :, last_frame_idx : last_frame_idx + 1, :, :]
            z_last_frame = self._adaptive_encode(input_last_frame)
            z = torch.cat([z, z_last_frame], dim=2)

        return z

    def _decode_temporal_pad_frames(self, z, pad_tokens):
        if pad_tokens <= 0:
            return 0
        intra_tail = self.clip_length % self.vae_ratio_t
        if intra_tail == 0:
            return int(pad_tokens) * int(self.vae_ratio_t)

        z_len_before_pad = z.shape[2] - pad_tokens
        return sum(
            (
                intra_tail
                if (z_len_before_pad + k) % self.tokens_chunk_size == 0
                else self.vae_ratio_t
            )
            for k in range(pad_tokens)
        )

    def _decode_temporal_output_frame_plan(self, z, z_head, z_tail, num_chunks, pad_tokens):
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        split_count = int(self.token_drop > 0) + 1
        total_frames = 0
        final_overlap_frames = 0

        if z_head is not None:
            total_frames += 1

        for i in range(num_chunks):
            t_start_idx = i * self.tokens_chunk_size
            t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap
            clip_token_len = max(0, min(t_end_idx, z.shape[2]) - min(t_start_idx, z.shape[2]))
            if i == 0 and z_head is not None:
                clip_token_len += z_head.shape[2]
            if i == num_chunks - 1 and z_tail is not None:
                clip_token_len += z_tail.shape[2]

            clip_frame_len = clip_token_len * self.vae_ratio_t
            if i == 0 and z_head is not None:
                clip_frame_len = max(0, clip_frame_len - self.vae_ratio_t)
            if i == num_chunks - 1 and z_tail is not None:
                clip_frame_len = max(0, clip_frame_len - self.vae_ratio_t)

            for j in range(split_count):
                f_start_idx = j * chunk_dec
                f_end_idx = min(f_start_idx + chunk_dec, clip_frame_len)
                chunk_frames = max(0, f_end_idx - f_start_idx - self.frame_pre_padding)
                if j == 0:
                    total_frames += chunk_frames
                else:
                    final_overlap_frames = chunk_frames

        total_frames += final_overlap_frames
        if z_tail is not None:
            total_frames += 1

        pad_frames = self._decode_temporal_pad_frames(z, pad_tokens)
        return int(total_frames), int(pad_frames), int(total_frames - pad_frames)

    def _decode_temporal_streaming(self, z, z_head, z_tail, num_chunks, pad_tokens, temporal_cat_dtype):
        total_frames, pad_frames, output_frames = self._decode_temporal_output_frame_plan(
            z, z_head, z_tail, num_chunks, pad_tokens
        )
        if output_frames <= 0:
            raise ValueError(
                f"decode_temporal streaming planned non-positive output_frames={output_frames} "
                f"total_frames={total_frames} pad_frames={pad_frames}"
            )


        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        split_count = int(self.token_drop > 0) + 1
        dec = None
        dec_overlap = None
        write_pos = 0
        logical_frames = 0
        dropped_frames = 0
        decoded_count = 0

        def write_part(part):
            nonlocal dec, write_pos, logical_frames, dropped_frames
            part_frames = int(part.shape[2])
            if part_frames <= 0:
                return
            logical_frames += part_frames
            if dec is None:
                out_shape = list(part.shape)
                out_shape[2] = output_frames
                dec = torch.empty(out_shape, dtype=part.dtype, device=part.device)

            remaining = int(dec.shape[2]) - write_pos
            copy_frames = min(part_frames, max(0, remaining))
            if copy_frames > 0:
                dec[:, :, write_pos : write_pos + copy_frames, :, :].copy_(
                    part[:, :, :copy_frames, :, :]
                )
                write_pos += copy_frames
            dropped_frames += part_frames - copy_frames

        for i in range(num_chunks):
            t_start_idx = i * self.tokens_chunk_size
            t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap
            clip_z = z[:, :, t_start_idx:t_end_idx, :, :]

            if i == 0 and z_head is not None:
                clip_z = torch.cat([z_head, clip_z], dim=2)

            if i == num_chunks - 1 and z_tail is not None:
                clip_z = torch.cat([clip_z, z_tail], dim=2)

            clip_dec = self._adaptive_decode(clip_z)
            decoded_count += 1
            if temporal_cat_dtype is not None and clip_dec.dtype != temporal_cat_dtype:
                clip_dec = clip_dec.to(temporal_cat_dtype)
            if clip_dec.device != z.device:
                clip_dec = clip_dec.to(z.device)


            dec_tail = None
            if i == 0 and z_head is not None:
                write_part(clip_dec[:, :, self.vae_ratio_t - 1 : self.vae_ratio_t, :, :])
                clip_dec = clip_dec[:, :, self.vae_ratio_t :, :, :]

            if i == num_chunks - 1 and z_tail is not None:
                dec_tail = clip_dec[:, :, -1:, :, :]
                clip_dec = clip_dec[:, :, : -self.vae_ratio_t, :, :]

            for j in range(split_count):
                f_start_idx = j * chunk_dec
                f_end_idx = min(f_start_idx + chunk_dec, clip_dec.shape[2])
                clip_dec_chunk = clip_dec[:, :, f_start_idx:f_end_idx, :, :]
                clip_dec_chunk = clip_dec_chunk[:, :, self.frame_pre_padding :, :, :]

                if j == 0:
                    if dec_overlap is not None:
                        clip_dec_chunk = self.blend(
                            dec_overlap, clip_dec_chunk, self.frame_overlap, dim=-3
                        )
                        dec_overlap = None
                    write_part(clip_dec_chunk)
                else:
                    # Break the view's reference to the full decoded clip so earlier
                    # temporal chunks can be released before the final output exists.
                    dec_overlap = clip_dec_chunk.contiguous()

            if i == num_chunks - 1:
                if dec_overlap is not None:
                    write_part(dec_overlap)
                    dec_overlap = None
                if dec_tail is not None:
                    write_part(dec_tail)

            del clip_dec, clip_z

        if dec is None:
            raise RuntimeError("decode_temporal streaming produced no output tensor")
        if logical_frames != total_frames or dropped_frames != pad_frames or write_pos != output_frames:
            raise RuntimeError(
                "decode_temporal streaming frame plan mismatch: "
                f"logical_frames={logical_frames} total_frames={total_frames} "
                f"dropped_frames={dropped_frames} pad_frames={pad_frames} "
                f"write_pos={write_pos} output_frames={output_frames}"
            )

        return dec

    def decode_temporal(self, z):
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t

        isolated_token_num = 0
        if self.isolated_first_frame and self.frame_pre_padding == 0:
            isolated_token_num = isolated_token_num + 1
        if self.isolated_last_frame:
            isolated_token_num = isolated_token_num + 1

        pseudo_total_tokens = z.shape[2] - isolated_token_num + self.token_drop

        pad_tokens = 0
        remainder = pseudo_total_tokens % self.tokens_chunk_size
        if remainder != 0:
            if self.training:
                raise ValueError(f"Temporal token length {z.shape[2]} is wrong!")
            else:
                pad_tokens = self.tokens_chunk_size - remainder
                pseudo_total_tokens = pseudo_total_tokens + pad_tokens

        pseudo_num_chunks = pseudo_total_tokens // self.tokens_chunk_size
        num_chunks = pseudo_num_chunks - int(self.token_drop > 0)

        z_head = None
        if self.isolated_first_frame and self.frame_pre_padding == 0:
            z_head = z[:, :, :1, :, :]
            z = z[:, :, 1:, :, :]

        z_tail = None
        if self.isolated_last_frame:
            z_tail = z[:, :, -1:, :, :]
            z = z[:, :, :-1, :, :]

        if pad_tokens > 0:
            pad_z = z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)
            z = torch.cat([z, pad_z], dim=2)

        temporal_cat_dtype = _resolve_temporal_cat_dtype()
        if not self.training and _resolve_temporal_stream_cat():
            return self._decode_temporal_streaming(
                z, z_head, z_tail, num_chunks, pad_tokens, temporal_cat_dtype
            )

        decoded_tasks = []
        for i in range(num_chunks):
            t_start_idx = i * self.tokens_chunk_size
            t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap
            clip_z = z[:, :, t_start_idx:t_end_idx, :, :]

            if i == 0 and z_head is not None:
                clip_z = torch.cat([z_head, clip_z], dim=2)

            if i == num_chunks - 1 and z_tail is not None:
                clip_z = torch.cat([clip_z, z_tail], dim=2)

            clip_dec = self._adaptive_decode(clip_z)
            if temporal_cat_dtype is not None and clip_dec.dtype != temporal_cat_dtype:
                clip_dec = clip_dec.to(temporal_cat_dtype)

            decoded_tasks.append((i, clip_dec))

        clip_dec_list = [clip_dec.to(z.device) for _, clip_dec in decoded_tasks]

        dec_list = []
        dec_overlap = None

        dec_head = None
        if z_head is not None:
            dec_head = clip_dec_list[0][:, :, self.vae_ratio_t - 1 : self.vae_ratio_t, :, :]
            clip_dec_list[0] = clip_dec_list[0][:, :, self.vae_ratio_t :, :, :]

        dec_tail = None
        if z_tail is not None:
            dec_tail = clip_dec_list[-1][:, :, -1:, :, :]
            clip_dec_list[-1] = clip_dec_list[-1][:, :, : -self.vae_ratio_t, :, :]

        if dec_head is not None:
            dec_list.append(dec_head)

        for i in range(num_chunks):
            for j in range(int(self.token_drop > 0) + 1):
                clip_dec = clip_dec_list[i]

                f_start_idx = j * chunk_dec
                f_end_idx = min(f_start_idx + chunk_dec, clip_dec.shape[2])
                clip_dec_chunk = clip_dec[:, :, f_start_idx:f_end_idx, :, :]
                clip_dec_chunk = clip_dec_chunk[:, :, self.frame_pre_padding :, :, :]

                if j == 0:
                    if dec_overlap is not None:
                        clip_dec_chunk = self.blend(
                            dec_overlap, clip_dec_chunk, self.frame_overlap, dim=-3
                        )
                    dec_list.append(clip_dec_chunk)
                else:
                    dec_overlap = clip_dec_chunk

        if dec_overlap is not None:
            dec_list.append(dec_overlap)

        if dec_tail is not None:
            dec_list.append(dec_tail)


        dec = torch.cat(dec_list, dim=2)

        pad_frames = self._decode_temporal_pad_frames(z, pad_tokens)
        if pad_frames > 0:
            dec = dec[:, :, :-pad_frames, :, :]

        return dec

    def decode_base(self, z, frame_num=None, process_image=False):
        if process_image or not self.use_3d_conv:
            if not self.use_3d_conv and z.ndim == 5:
                z = z.squeeze(2)

            recon = self._adaptive_decode(z)
        else:
            recon = self.decode_temporal(z)

        if self.use_3d_conv:
            if frame_num is not None:
                target_frames = frame_num
            else:
                target_frames = recon.shape[2]

            recon = self.trim_output(recon, target_frames)
            if process_image:
                recon = recon.squeeze(2)

        return recon

    #########################################################
    # freeze_scope is retained from the training codebase: in this
    # inference-only bundle (self.training is always False) it simply
    # provides the no_grad() context used by encode()/decode().
    #########################################################


    def freeze_scope(self, module_name):
        if not self.training:
            return torch.no_grad()

        if_freeze = module_name in self.fix_modules
        if if_freeze:
            return torch.no_grad()
        else:
            return nullcontext()








    #########################################################
    # following methods are for inference
    #########################################################

    @torch.no_grad()
    def encode_images(
        self,
        images: Union[List[np.ndarray], List[torch.Tensor]],
        transform_input: bool = False,
        use_fp16_latent: bool = False,
        verbose: bool = False,
    ) -> List[torch.Tensor]:
        """encode images into latents

        Args:
            images (Union[List[np.ndarray], List[torch.Tensor]]):
                List of images, single input will be wrapped in a list.
                If input is a list of np.ndarray, it should be in shape B * (H, W, 3), dtype uint8.
                If input is a list of torch.Tensor, it should be in shape B * (3, H, W), dtype float32.
            transform_input (bool, optional):
                Whether to transform input using ImageNet std/mean. Defaults to False.
                If input is a list of np.ndarray, it will always be set to True.
            use_fp16_latent (bool, optional):
                Whether to use fp16 latent. Defaults to False.
            verbose (bool, optional):
                Whether to print debug information. Defaults to False.

        Returns:
            List[torch.Tensor]:
                List of image latents.
                If self.use_3d_conv is True, it should be in shape B * (D, 1, H', W').
                Otherwise, it should be in shape B * (D, H', W').
        """

        images = self.processor._ensure_list(images)

        if isinstance(images[0], Image.Image):
            images = [np.array(image) for image in images]

        if isinstance(images[0], np.ndarray):
            device = next(self.parameters()).device
            images = self.processor.convert_numpy_to_tensor(images, device)
            images = torch.split(images, 1, dim=0)
            transform_input = True

        if transform_input:
            images = [
                image.unsqueeze(0) if image.ndim == 3 else image for image in images
            ]
            images = [self.processor.transform_tensor(image) for image in images]

        prepared = []
        for image_tensor in images:
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            _, _, h, w = image_tensor.shape
            new_h, new_w = self.processor._align_to_total_patch_size(h, w)
            image_tensor = self.processor._crop_to_align(image_tensor, new_h, new_w)
            prepared.append(image_tensor)

        if len(prepared) > 1 and len(set(t.shape for t in prepared)) == 1:
            stacked = torch.cat(prepared, dim=0)
            if verbose:
                pass
            all_latents = self.encode_base(stacked, True)
            image_latents = [all_latents[i].contiguous() for i in range(all_latents.shape[0])]
        else:
            image_latents = []
            for image_tensor in prepared:
                if verbose:
                    pass
                image_latent = self.encode_base(image_tensor, True)
                image_latents.append(image_latent.squeeze(0).contiguous())

        if use_fp16_latent:
            image_latents = [lat.to(torch.float16) for lat in image_latents]

        if verbose:
            for lat in image_latents:
                pass

        return image_latents

    @torch.no_grad()
    def encode_videos(
        self,
        videos: Union[List[np.ndarray], List[torch.Tensor]],
        transform_input: bool = False,
        use_fp16_latent: bool = False,
        verbose: bool = False,
        encode_prefix: bool = False,
    ) -> List[torch.Tensor]:
        """encode videos into latents

        Args:
            videos (Union[List[np.ndarray], List[torch.Tensor]]):
                List of videos, single input will be wrapped in a list.
                If input is a list of np.ndarray, it should be in shape B * (T, H, W, 3), dtype uint8.
                If input is a list of torch.Tensor, it should be in shape B * (3, T, H, W), dtype float32.
            transform_input (bool, optional):
                Whether to transform input using ImageNet std/mean. Defaults to False.
                If input is a list of np.ndarray, it will always be set to True.
            use_fp16_latent (bool, optional):
                Whether to use fp16 latent. Defaults to False.
            verbose (bool, optional):
                Whether to print debug information. Defaults to False.
            encode_prefix (bool, optional):
                Continuation (prefix) mode: prepend normalized
                black frames to token alignment, append black frames to chunk
                alignment, encode with token_drop disabled, then discard only
                the trailing padding tokens. Returns both latents and leading
                pad-frame counts. Defaults to False.

        Returns:
            List[torch.Tensor]:
                List of video latents, shape B * (D, T', H', W').
                With encode_prefix=True, returns
                (List[torch.Tensor], List[int]).
        """

        videos = self.processor._ensure_list(videos)

        if isinstance(videos[0], np.ndarray):
            device = next(self.parameters()).device
            videos = [self.processor.convert_numpy_to_tensor(video, device) for video in videos]
            transform_input = True

        if transform_input:
            videos = [self.processor.transform_tensor(video) for video in videos]
            videos = [video.transpose(0, 1) for video in videos]

        if encode_prefix:
            if self.isolated_last_frame:
                raise ValueError(
                    "encode_prefix does not support isolated_last_frame"
                )

            video_latents = []
            prefix_pad_frames = []
            for video in videos:
                if video.ndim == 4:
                    video = video.unsqueeze(0)
                _, _, _, h, w = video.shape
                new_h, new_w = self.processor._align_to_total_patch_size(h, w)
                video = self.processor._crop_to_align(
                    video, new_h, new_w, is_video=True
                )

                model_alignment = (
                    self.token_drop,
                    self.frame_drop,
                    self.token_overlap,
                    self.frame_overlap,
                )
                processor_alignment = (
                    self.processor.token_overlap,
                    self.processor.frame_overlap,
                )
                self.token_drop = 0
                self.frame_drop = 0
                self.token_overlap = 0
                self.frame_overlap = 0
                self.processor.token_overlap = 0
                self.processor.frame_overlap = 0
                try:
                    orig_frames = video.shape[2]
                    leading, trailing, drop_tokens = (
                        self.processor.align_video_length_2pass(orig_frames)
                    )
                    _, _, _, cropped_h, cropped_w = video.shape
                    if leading > 0:
                        black = self.processor.transform(
                            video.new_zeros(leading, 3, cropped_h, cropped_w)
                        )
                        black = black.unsqueeze(0).permute(0, 2, 1, 3, 4)
                        video = torch.cat([black, video], dim=2)
                    if trailing > 0:
                        black = self.processor.transform(
                            video.new_zeros(trailing, 3, cropped_h, cropped_w)
                        )
                        black = black.unsqueeze(0).permute(0, 2, 1, 3, 4)
                        video = torch.cat([video, black], dim=2)

                    if verbose:
                        pass

                    video_latent = self.encode_base(video, False)
                    if drop_tokens > 0:
                        video_latent = video_latent[:, :, :-drop_tokens, :, :]
                    prefix_pad_frames.append(leading)
                finally:
                    (
                        self.token_drop,
                        self.frame_drop,
                        self.token_overlap,
                        self.frame_overlap,
                    ) = model_alignment
                    (
                        self.processor.token_overlap,
                        self.processor.frame_overlap,
                    ) = processor_alignment

                video_latents.append(video_latent.squeeze(0).contiguous())

            if use_fp16_latent:
                video_latents = [lat.to(torch.float16) for lat in video_latents]
            if verbose:
                for latent in video_latents:
                    pass
            return video_latents, prefix_pad_frames

        prepared = []
        for video in videos:
            if video.ndim == 4:
                video = video.unsqueeze(0)
            used_frame_length = self.processor.get_suitable_video_length(video.shape[2], verbose)
            _, _, _, h, w = video.shape
            new_h, new_w = self.processor._align_to_total_patch_size(h, w)
            video = video[:, :, :used_frame_length, :, :]
            video = self.processor._crop_to_align(video, new_h, new_w, is_video=True)
            prepared.append(video)

        if len(prepared) > 1 and len(set(t.shape for t in prepared)) == 1:
            stacked = torch.cat(prepared, dim=0)
            if verbose:
                pass
            all_latents = self.encode_base(stacked, False)
            video_latents = [all_latents[i].contiguous() for i in range(all_latents.shape[0])]
        else:
            video_latents = []
            for video in prepared:
                if verbose:
                    pass
                video_latent = self.encode_base(video, False)
                video_latents.append(video_latent.squeeze(0).contiguous())

        if use_fp16_latent:
            video_latents = [lat.to(torch.float16) for lat in video_latents]

        if verbose:
            for lat in video_latents:
                pass

        return video_latents




# ============================================================================
# Legacy CNN VAE
# ============================================================================


class AutoencoderKLLegacy(AutoencoderKL):
    r"""
    A VAE model (legacy CNN-based) for encoding pixels into latents and decoding latent representations into pixels.
    """

    @register_to_config
    def __init__(
        self,
        in_channels=3,
        out_ch=3,
        ch=128,
        embed_dim=16,
        z_channels=16,
        use_3d_conv=False,
        # cnn vae
        zq_ch_encoder=None,
        zq_ch_decoder=None,
        num_res_blocks=2,
        num_res_blocks_decoder=None,
        ch_mult=[1, 2, 2, 4, 4, 8],
        space_down=[2, 2, 2, 2, 1, 1],
        space_up=[1, 2, 2, 2, 2, 1],
        time_down=None,
        time_up=None,
        padding_mode="zeros",
        padding_mode_t=None,
        use_t_isolated_gn=False,
        causal_encoder=True,
        causal_decoder=True,
        use_vit_decoder=False,
        vit_decoder_kwargs=None,
        # stats
        shift_factor=0.0,
        scaling_factor=1.0,
        # pixel normalization
        pixel_norm_type="imagenet",
        # others
        **kwargs,
    ):
        ModelMixin.__init__(self)  # NOTE: avoid wrong @register_to_config

        if not use_3d_conv or not use_vit_decoder:
            raise NotImplementedError(
                "this release only supports use_3d_conv=True with use_vit_decoder=True"
            )

        self.transform = get_normalize_transform(pixel_norm_type)
        self.transform_rev = get_denormalize_transform(pixel_norm_type)

        self.use_3d_conv = use_3d_conv
        self.causal_encoder = causal_encoder
        self.causal_decoder = causal_decoder
        self.slidedec = self.causal_encoder and not self.causal_decoder

        # some registered parameters for simplicity
        self.vae_ratio = int(np.cumprod(space_down)[-1])
        self.vae_ratio_t = int(np.cumprod(time_down)[-1]) if time_down else 1
        self.config["vae_ratio"] = self.vae_ratio
        self.config["vae_ratio_t"] = self.vae_ratio_t

        # some registered parameters for inference and training
        self.setup_forward(**kwargs)
        self.setup_training(**kwargs)

        # init encoder
        encoder_config = {
            "double_z": True,
            "z_channels": z_channels,
            "zq_ch": zq_ch_encoder,
            "in_channels": in_channels,
            "ch": ch,
            "num_res_blocks": num_res_blocks,
            "ch_mult": ch_mult,
            "space_down": space_down,
            "time_down": time_down,
            "padding_mode": padding_mode,
            "padding_mode_t": padding_mode_t,
            "causal": causal_encoder,
            "use_t_isolated_gn": use_t_isolated_gn,
        }
        self.encoder = EncoderFCN3D(**encoder_config)

        # init pointwise quant/post_quant conv
        self.quant_conv = nn.Conv3d(z_channels * 2, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv3d(embed_dim, z_channels, 1)

        self.use_vit_decoder = use_vit_decoder

        # init decoder
        vit_kwargs = {
            "patch_size": self.vae_ratio,
            "in_channels": z_channels,
            "out_channels": out_ch,
            **(vit_decoder_kwargs or {}),
        }
        vit_kwargs.setdefault("patch_size_t", self.vae_ratio_t)
        vit_kwargs.setdefault("t_causal", causal_decoder)
        self.decoder = ViT3DDecoder(**vit_kwargs)

        for module in set(self.fix_modules + self.frozen_modules):
            self._freeze_nested_module(module)

        self.gradient_checkpointing = False

    def encode(self, x):
        with self.freeze_scope("encoder"):
            h = self.encoder(x)

        with self.freeze_scope("quant_conv"):
            moments = self.quant_conv(h)

        return moments

    def decode(self, z):
        with self.freeze_scope("post_quant_conv"):
            z2 = self.post_quant_conv(z)

        with self.freeze_scope("decoder"):
            if self.use_vit_decoder:
                dec = self.decoder(z2)
            else:
                dec = self.decoder(z2, z)

        return dec

    def encode_base(self, input, process_image=False):
        if self.use_3d_conv and input.ndim == 4:
            input = input.unsqueeze(2)

        if process_image or not self.use_3d_conv:
            moments = self._adaptive_encode(input)
        else:
            moments = self.encode_temporal(input)

        z = DiagonalGaussianDistribution(moments).sample()

        if process_image and self.use_3d_conv:
            z = self.trim_code(z, 1)

        return z

    #########################################################
    # training-related knobs kept only for checkpoint/config compatibility
    #########################################################

    def setup_training(self, **kwargs):
        self.fix_modules = kwargs.get("fix_modules", [])
        self.frozen_modules = kwargs.get("frozen_modules", [])


# ======================================================================
# ---- __init__.py ----
# ======================================================================
# MiniMax H3 video VAE (DiffSynth port of the checkpoint remote code).
# Asymmetric KL VAE: causal 3D-CNN encoder + ViT-3D decoder, 24-ch latent,
# 16x spatial / 4x temporal (17-frame -> 5-latent) compression. Single-card
# only (sequence-parallel/distributed code removed). Defaults below come from
# the checkpoint video_vae/source/config.json + video_vae/config.json.


class MiniMaxH3VideoVAE(AutoencoderKLLegacy):
    """Top-level video VAE. state_dict keys match the checkpoint 1:1
    (encoder./decoder./quant_conv./post_quant_conv.)."""

    def __init__(
        self,
        in_channels=3,
        out_ch=3,
        ch=128,
        embed_dim=24,
        z_channels=24,
        use_3d_conv=True,
        zq_ch_encoder=None,
        zq_ch_decoder=None,
        num_res_blocks=2,
        num_res_blocks_decoder=None,
        ch_mult=(1, 2, 2, 4, 4, 8),
        space_down=(2, 2, 2, 2, 1, 1),
        space_up=(1, 2, 2, 2, 2, 1),
        time_down=(1, 2, 2, 1, 1, 1),
        time_up=None,
        padding_mode="reflect",
        padding_mode_t=None,
        use_t_isolated_gn=True,
        causal_encoder=True,
        causal_decoder=False,
        use_vit_decoder=True,
        vit_decoder_kwargs=None,
        shift_factor=0.0,
        scaling_factor=1.0,
        pixel_norm_type="imagenet",
        clip_length=17,
        token_drop=3,
        encoder_tiling=1,
        decoder_tiling=1,
        parallel_tiling=1,
        tile_size=256,
        tile_overlap_min=64,
        encoder_parallel=0,
        decoder_parallel=0,
        chunk_dim=-1,
        **kwargs,
    ):
        if vit_decoder_kwargs is None:
            vit_decoder_kwargs = {
                "dim_head": 64,
                "ffn_activation_fn": "silu",
                "ffn_use_gated": True,
                "heads": 32,
                "norm_affine": True,
                "norm_type": "rms_norm",
                "num_layers": 36,
                "qk_norm_affine": False,
                "qk_norm_type": "rms_norm",
                "rope_dim_ratio": 0.75,
                "rope_theta": 100.0,
            }
        super().__init__(
            in_channels=in_channels,
            out_ch=out_ch,
            ch=ch,
            embed_dim=embed_dim,
            z_channels=z_channels,
            use_3d_conv=use_3d_conv,
            zq_ch_encoder=zq_ch_encoder,
            zq_ch_decoder=zq_ch_decoder,
            num_res_blocks=num_res_blocks,
            num_res_blocks_decoder=num_res_blocks_decoder,
            ch_mult=list(ch_mult),
            space_down=list(space_down),
            space_up=list(space_up),
            time_down=list(time_down),
            time_up=time_up,
            padding_mode=padding_mode,
            padding_mode_t=padding_mode_t,
            use_t_isolated_gn=use_t_isolated_gn,
            causal_encoder=causal_encoder,
            causal_decoder=causal_decoder,
            use_vit_decoder=use_vit_decoder,
            vit_decoder_kwargs=vit_decoder_kwargs,
            shift_factor=shift_factor,
            scaling_factor=scaling_factor,
            pixel_norm_type=pixel_norm_type,
            clip_length=clip_length,
            token_drop=token_drop,
            encoder_tiling=encoder_tiling,
            decoder_tiling=decoder_tiling,
            parallel_tiling=parallel_tiling,
            tile_size=tile_size,
            tile_overlap_min=tile_overlap_min,
            encoder_parallel=encoder_parallel,
            decoder_parallel=decoder_parallel,
            chunk_dim=chunk_dim,
            **kwargs,
        )


__all__ = ["MiniMaxH3VideoVAE"]
