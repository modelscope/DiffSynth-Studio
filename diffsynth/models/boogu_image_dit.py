import inspect
import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import RMSNorm
import math
import torch.nn.functional as F
from einops import repeat
from dataclasses import dataclass


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor=None,
        out_dim: int = None,
        out_context_dim: int = None,
        context_pre_only=None,
        pre_only=False,
        elementwise_affine: bool = True,
        is_causal: bool = False,
    ):
        super().__init__()

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.is_causal = is_causal

        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head ** -0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None."
            )

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None

        if spatial_norm_dim is not None:
            self.spatial_norm = None
        else:
            self.spatial_norm = None

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "layer_norm":
            self.norm_q = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            self.norm_k = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        elif qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            self.norm_k = RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        elif qk_norm == "rms_norm_across_heads":
            self.norm_q = RMSNorm(dim_head * heads, eps=eps)
            self.norm_k = RMSNorm(dim_head * kv_heads, eps=eps)
        else:
            raise ValueError(f"unknown qk_norm: {qk_norm}.")

        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = nn.LayerNorm(self.cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            if self.added_kv_proj_dim is not None:
                norm_cross_num_channels = added_kv_proj_dim
            else:
                norm_cross_num_channels = self.cross_attention_dim
            self.norm_cross = nn.GroupNorm(
                num_channels=norm_cross_num_channels, num_groups=cross_attention_norm_num_groups, eps=1e-5, affine=True
            )
        else:
            raise ValueError(f"unknown cross_attention_norm: {cross_attention_norm}.")

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)

        if not self.only_cross_attention:
            self.to_k = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
            self.to_v = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        self.added_proj_bias = added_proj_bias
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
            if self.context_pre_only is not None:
                self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
        else:
            self.add_q_proj = None
            self.add_k_proj = None
            self.add_v_proj = None

        if not self.pre_only:
            self.to_out = nn.ModuleList([])
            self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
            self.to_out.append(nn.Dropout(dropout))
        else:
            self.to_out = None

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out = nn.Linear(self.inner_dim, self.out_context_dim, bias=out_bias)
        else:
            self.to_add_out = None

        if qk_norm is not None and added_kv_proj_dim is not None:
            if qk_norm == "layer_norm":
                self.norm_added_q = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
                self.norm_added_k = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            elif qk_norm == "rms_norm":
                self.norm_added_q = RMSNorm(dim_head, eps=eps)
                self.norm_added_k = RMSNorm(dim_head, eps=eps)
            elif qk_norm == "rms_norm_across_heads":
                self.norm_added_q = None
                self.norm_added_k = RMSNorm(dim_head * kv_heads, eps=eps)
            else:
                raise ValueError(f"unknown qk_norm: {qk_norm}.")
        else:
            self.norm_added_q = None
            self.norm_added_k = None

        if processor is None:
            processor = AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
        self.set_processor(processor)

    def set_processor(self, processor) -> None:
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)
        ):
            pass
        self.processor = processor

    def get_processor(self, return_deprecated_lora: bool = False):
        if not return_deprecated_lora:
            return self.processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states"}
        unused_kwargs = [
            k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def norm_encoder_hidden_states(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.norm_cross is not None, "self.norm_cross must be defined to call self.norm_encoder_hidden_states"

        if isinstance(self.norm_cross, nn.LayerNorm):
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
        elif isinstance(self.norm_cross, nn.GroupNorm):
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
        else:
            assert False

        return encoder_hidden_states


class AttnProcessor:
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        if attn.to_out is not None:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class AttnProcessor2_0:
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=attn.is_causal, scale=attn.scale
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if attn.to_out is not None:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def _get_taylor_cache_entry(
    cache_dic: Dict, current: Dict, create: bool = False
) -> Dict:
    cache_root = cache_dic["cache"][-1]
    stream = current["stream"]
    layer = current["layer"]
    module = current["module"]

    if create:
        return (
            cache_root.setdefault(stream, {})
            .setdefault(layer, {})
            .setdefault(module, {})
        )
    return cache_root[stream][layer][module]


def _tree_sub(lhs, rhs):
    if isinstance(lhs, tuple):
        return tuple(_tree_sub(x, y) for x, y in zip(lhs, rhs))
    return lhs - rhs


def _tree_div(value, divisor):
    if isinstance(value, tuple):
        return tuple(_tree_div(x, divisor) for x in value)
    return value / divisor


def _tree_add(lhs, rhs):
    if lhs is None:
        return rhs
    if isinstance(lhs, tuple):
        return tuple(_tree_add(x, y) for x, y in zip(lhs, rhs))
    return lhs + rhs


def _tree_mul(value, scalar):
    if isinstance(value, tuple):
        return tuple(_tree_mul(x, scalar) for x in value)
    return value * scalar


def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Build/update Taylor coefficients from the latest feature tensor.

    Args:
        cache_dic: Global cache dict storing per-stream/layer/module states.
        current: Current execution state with keys like `stream`, `layer`,
            `module`, and `step`.
        feature: Current feature tensor to use as 0-th order term.
    """
    difference_distance = (
        current["activated_steps"][-1] - current["activated_steps"][-2]
    )

    cache_entry = _get_taylor_cache_entry(cache_dic, current, create=True)
    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic["max_order"]):
        if (cache_entry.get(i, None) is not None) and (
            current["step"] > cache_dic["first_enhance"] - 2
        ):
            updated_taylor_factors[i + 1] = (
                updated_taylor_factors[i] - cache_entry[i]
            ) / difference_distance
        else:
            break

    cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]] = (
        updated_taylor_factors
    )


def derivative_approximation_4_double_stream(
    cache_dic: Dict, current: Dict, feature: tuple
):
    """
    Build/update Taylor coefficients for double-stream outputs.
    """
    difference_distance = (
        current["activated_steps"][-1] - current["activated_steps"][-2]
    )

    cache_entry = _get_taylor_cache_entry(cache_dic, current, create=True)
    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic["max_order"]):
        if (cache_entry.get(i, None) is not None) and (
            current["step"] > cache_dic["first_enhance"] - 2
        ):
            updated_taylor_factors[i + 1] = _tree_div(
                _tree_sub(updated_taylor_factors[i], cache_entry[i]),
                difference_distance,
            )
        else:
            break

    cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]] = (
        updated_taylor_factors
    )


def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Reconstruct feature estimate using cached Taylor coefficients.

    Returns:
        A tensor with the same shape as cached feature tensors for the
        current stream/layer/module.
    """
    x = current["step"] - current["activated_steps"][-1]
    output = 0
    cache_entry = _get_taylor_cache_entry(cache_dic, current)

    for i in range(len(cache_entry)):
        output += (1 / math.factorial(i)) * cache_entry[i] * (x**i)

    return output


def taylor_formula_4_double_stream(cache_dic: Dict, current: Dict) -> tuple:
    """
    Reconstruct double-stream outputs using cached Taylor coefficients.
    """
    x = current["step"] - current["activated_steps"][-1]
    output = None
    cache_entry = _get_taylor_cache_entry(cache_dic, current)

    for i in range(len(cache_entry)):
        output = _tree_add(
            output,
            _tree_mul(cache_entry[i], (1 / math.factorial(i)) * (x**i)),
        )

    return output


def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor storage for the first step/module access.

    The target location is
    `cache_dic['cache'][-1][stream][layer][module]`.
    """
    if (current["step"] == 0) and (cache_dic["taylor_cache"]):
        cache_root = cache_dic["cache"][-1]
        cache_root.setdefault(current["stream"], {}).setdefault(current["layer"], {})[
            current["module"]
        ] = {}


class BooguImagePromptTuningRotaryPosEmbed(nn.Module):
    """
    Rotary Position Embedding for Prompt Tuning tokens.

    This class generates rotary position embeddings specifically for prompt tuning tokens.
    Since prompt tokens are treated as text tokens, we use text-style position encoding
    with a fixed sequence length equal to num_trainable_prompt_tokens.

    Args:
        theta: Base frequency for rotary embeddings
        axes_dim: Dimensions for each axis (tuple like (32, 32, 32))
        num_trainable_prompt_tokens: Number of trainable prompt tokens
    """

    def __init__(self, theta: int, dim: int, num_trainable_prompt_tokens: int):
        super().__init__()
        self.theta = theta
        self.num_trainable_prompt_tokens = num_trainable_prompt_tokens
        # For text tokens, only use the first dimension (text/temporal dimension)
        self.dim = dim  # Extract text dimension from tuple

    def forward(
        self, batch_size: int, device: torch.device, use_causal_mask: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate rotary position embeddings and attention mask for prompt tuning.

        Args:
            batch_size: Batch size
            device: Target device for tensors
            use_causal_mask: Whether to use causal attention mask

        Returns:
            Tuple of (rotary_embeddings, attention_mask)
            - rotary_embeddings: [B, num_tokens, instruction_dim//2] - RoPE embeddings for prompt tokens (complex form)
            - attention_mask: [B, num_tokens] or [B, num_tokens, num_tokens] - Attention mask
        """
        # Generate 1D rotary embeddings for text-style tokens
        freqs_dtype = (
            torch.float32 if torch.backends.mps.is_available() else torch.float64
        )

        # get_1d_rotary_pos_embed(dim, seq_len) returns [seq_len, dim//2]
        # Because RoPE uses complex representation, each dimension is split into sin/cos pairs
        text_freqs_cis = get_1d_rotary_pos_embed(
            self.dim,  # This should be 32 (text dimension)
            self.num_trainable_prompt_tokens,  # Sequence length
            theta=self.theta,
            freqs_dtype=freqs_dtype,
        )

        # For prompt tuning, we create simple sequential position embeddings
        # Each prompt token gets a unique position ID: 0, 1, 2, ..., num_tokens-1
        position_indices = torch.arange(
            self.num_trainable_prompt_tokens,
            dtype=torch.int64,
            device=text_freqs_cis.device,
        )

        # Select the appropriate rotary embeddings for each position
        # text_freqs_cis is [num_tokens, instruction_dim//2], we want [num_tokens, instruction_dim//2]
        rotary_emb = text_freqs_cis[
            position_indices
        ]  # [num_tokens, instruction_dim//2]

        # Expand to batch size and move to target device
        rotary_emb = (
            rotary_emb.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        )  # [B, num_tokens, instruction_dim//2]

        # Create attention mask based on use_causal_mask parameter
        if use_causal_mask:
            # Create causal mask: only future tokens can attend to past tokens
            # Lower triangular matrix where mask[i, j] = True if i >= j
            causal_mask = torch.tril(
                torch.ones(
                    self.num_trainable_prompt_tokens,
                    self.num_trainable_prompt_tokens,
                    dtype=torch.bool,
                    device=device,
                )
            )  # [num_tokens, num_tokens]

            # Expand to batch size [B, num_tokens, num_tokens]
            attention_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # Non-causal mask: all tokens can attend to each other (all True)
            attention_mask = torch.ones(
                batch_size,
                self.num_trainable_prompt_tokens,
                dtype=torch.bool,
                device=device,
            )  # [B, num_tokens]

        return rotary_emb, attention_mask


def get_1d_rotary_pos_embed(
    dim: int,
    pos: np.ndarray | int,
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    theta = theta * ntk_factor
    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)) / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    is_npu = freqs.device.type == "npu"
    if is_npu:
        freqs = freqs.float()
    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio, allegro
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis


class BooguImageDoubleStreamRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        theta: int,
        axes_dim: Tuple[int, int, int],
        axes_lens: Tuple[int, int, int] = (300, 512, 512),
        patch_size: int = 2,
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.axes_lens = axes_lens
        self.patch_size = patch_size

    @staticmethod
    def get_freqs_cis(
        axes_dim: Tuple[int, int, int], axes_lens: Tuple[int, int, int], theta: int
    ) -> List[torch.Tensor]:
        freqs_cis = []
        freqs_dtype = (
            torch.float32 if torch.backends.mps.is_available() else torch.float64
        )
        for i, (d, e) in enumerate(zip(axes_dim, axes_lens)):
            emb = get_1d_rotary_pos_embed(d, e, theta=theta, freqs_dtype=freqs_dtype)
            freqs_cis.append(emb)
        return freqs_cis

    def _get_freqs_cis(self, freqs_cis, ids: torch.Tensor) -> torch.Tensor:
        device = ids.device
        if ids.device.type == "mps":
            ids = ids.to("cpu")

        result = []
        for i in range(len(self.axes_dim)):
            freqs = freqs_cis[i].to(ids.device)
            index = ids[:, :, i : i + 1].repeat(1, 1, freqs.shape[-1]).to(torch.int64)
            result.append(
                torch.gather(
                    freqs.unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index
                )
            )
        return torch.cat(result, dim=-1).to(device)

    def forward(
        self,
        freqs_cis,
        attention_mask,
        l_effective_ref_img_len,
        l_effective_img_len,
        ref_img_sizes,
        img_sizes,
        device,
    ):
        batch_size = len(attention_mask)
        p = self.patch_size

        encoder_seq_len = attention_mask.shape[1]
        l_effective_cap_len = attention_mask.sum(dim=1).tolist()

        seq_lengths = [
            cap_len + sum(ref_img_len) + img_len
            for cap_len, ref_img_len, img_len in zip(
                l_effective_cap_len, l_effective_ref_img_len, l_effective_img_len
            )
        ]

        max_seq_len = max(seq_lengths)
        max_ref_img_len = max(
            [sum(ref_img_len) for ref_img_len in l_effective_ref_img_len]
        )
        max_img_len = max(l_effective_img_len)

        # Create position IDs
        position_ids = torch.zeros(
            batch_size, max_seq_len, 3, dtype=torch.int32, device=device
        )

        for i, (cap_seq_len, seq_len) in enumerate(
            zip(l_effective_cap_len, seq_lengths)
        ):
            # add text position ids
            position_ids[i, :cap_seq_len] = repeat(
                torch.arange(cap_seq_len, dtype=torch.int32, device=device), "l -> l 3"
            )

            pe_shift = cap_seq_len
            pe_shift_len = cap_seq_len

            if ref_img_sizes[i] is not None:
                for ref_img_size, ref_img_len in zip(
                    ref_img_sizes[i], l_effective_ref_img_len[i]
                ):
                    H, W = ref_img_size
                    ref_H_tokens, ref_W_tokens = H // p, W // p
                    assert ref_H_tokens * ref_W_tokens == ref_img_len
                    # add image position ids

                    row_ids = repeat(
                        torch.arange(ref_H_tokens, dtype=torch.int32, device=device),
                        "h -> h w",
                        w=ref_W_tokens,
                    ).flatten()
                    col_ids = repeat(
                        torch.arange(ref_W_tokens, dtype=torch.int32, device=device),
                        "w -> h w",
                        h=ref_H_tokens,
                    ).flatten()
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 0] = (
                        pe_shift
                    )
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 1] = (
                        row_ids
                    )
                    position_ids[i, pe_shift_len : pe_shift_len + ref_img_len, 2] = (
                        col_ids
                    )

                    pe_shift += max(ref_H_tokens, ref_W_tokens)
                    pe_shift_len += ref_img_len

            H, W = img_sizes[i]
            H_tokens, W_tokens = H // p, W // p
            assert H_tokens * W_tokens == l_effective_img_len[i]

            row_ids = repeat(
                torch.arange(H_tokens, dtype=torch.int32, device=device),
                "h -> h w",
                w=W_tokens,
            ).flatten()
            col_ids = repeat(
                torch.arange(W_tokens, dtype=torch.int32, device=device),
                "w -> h w",
                h=H_tokens,
            ).flatten()

            assert pe_shift_len + l_effective_img_len[i] == seq_len
            position_ids[i, pe_shift_len:seq_len, 0] = pe_shift
            position_ids[i, pe_shift_len:seq_len, 1] = row_ids
            position_ids[i, pe_shift_len:seq_len, 2] = col_ids

        # Get combined rotary embeddings
        freqs_cis = self._get_freqs_cis(freqs_cis, position_ids)

        # create separate rotary embeddings for captions and images
        cap_freqs_cis = torch.zeros(
            batch_size,
            encoder_seq_len,
            freqs_cis.shape[-1],
            device=device,
            dtype=freqs_cis.dtype,
        )
        ref_img_freqs_cis = torch.zeros(
            batch_size,
            max_ref_img_len,
            freqs_cis.shape[-1],
            device=device,
            dtype=freqs_cis.dtype,
        )
        img_freqs_cis = torch.zeros(
            batch_size,
            max_img_len,
            freqs_cis.shape[-1],
            device=device,
            dtype=freqs_cis.dtype,
        )

        # Calculate combined image sequence lengths (ref_img + img) for each sample
        combined_img_seq_lengths = [
            sum(ref_img_len) + img_len
            for ref_img_len, img_len in zip(
                l_effective_ref_img_len, l_effective_img_len
            )
        ]
        max_combined_img_len = max(combined_img_seq_lengths)

        # Create combined image rotary embeddings
        combined_img_freqs_cis = torch.zeros(
            batch_size,
            max_combined_img_len,
            freqs_cis.shape[-1],
            device=device,
            dtype=freqs_cis.dtype,
        )

        for i, (cap_seq_len, ref_img_len, img_len, seq_len) in enumerate(
            zip(
                l_effective_cap_len,
                l_effective_ref_img_len,
                l_effective_img_len,
                seq_lengths,
            )
        ):
            cap_freqs_cis[i, :cap_seq_len] = freqs_cis[i, :cap_seq_len]
            ref_img_freqs_cis[i, : sum(ref_img_len)] = freqs_cis[
                i, cap_seq_len : cap_seq_len + sum(ref_img_len)
            ]
            img_freqs_cis[i, :img_len] = freqs_cis[
                i,
                cap_seq_len + sum(ref_img_len) : cap_seq_len
                + sum(ref_img_len)
                + img_len,
            ]

            # Combined image rotary embeddings: ref_img + img (same order as img_patch_embed_and_refine)
            combined_img_freqs_cis[i, : sum(ref_img_len)] = freqs_cis[
                i, cap_seq_len : cap_seq_len + sum(ref_img_len)
            ]
            combined_img_freqs_cis[i, sum(ref_img_len) : sum(ref_img_len) + img_len] = (
                freqs_cis[
                    i,
                    cap_seq_len + sum(ref_img_len) : cap_seq_len
                    + sum(ref_img_len)
                    + img_len,
                ]
            )

        return (
            cap_freqs_cis,
            ref_img_freqs_cis,
            img_freqs_cis,
            freqs_cis,
            l_effective_cap_len,
            seq_lengths,
            combined_img_freqs_cis,
            combined_img_seq_lengths,
        )


class LuminaRMSNormZero(nn.Module):
    """
    Norm layer adaptive RMS normalization zero.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
    """

    def __init__(
        self,
        embedding_dim: int,
        norm_eps: float,
        norm_elementwise_affine: bool,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(
            min(embedding_dim, 1024),
            4 * embedding_dim,
            bias=True,
        )

        self.norm = RMSNorm(embedding_dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None])
        return x, gate_msa, scale_mlp, gate_mlp


class LuminaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
        out_dim: Optional[int] = None,
    ):
        super().__init__()

        # AdaLN
        self.silu = nn.SiLU()
        self.linear_1 = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=bias)

        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(
                embedding_dim, eps=eps, elementwise_affine=elementwise_affine
            )
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

        self.linear_2 = None
        if out_dim is not None:
            self.linear_2 = nn.Linear(embedding_dim, out_dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        conditioning_embedding: torch.Tensor,
    ) -> torch.Tensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        emb = self.linear_1(self.silu(conditioning_embedding).to(x.dtype))
        scale = emb
        x = self.norm(x) * (1 + scale)[:, None, :]

        if self.linear_2 is not None:
            x = self.linear_2(x)

        return x


def swiglu(x, y):
    return F.silu(x.float(), inplace=False).to(x.dtype) * y


class LuminaFeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        hidden_size (`int`):
            The dimensionality of the hidden layers in the model. This parameter determines the width of the model's
            hidden representations.
        intermediate_size (`int`): The intermediate dimension of the feedforward layer.
        multiple_of (`int`, *optional*): Value to ensure hidden dimension is a multiple
            of this value.
        ffn_dim_multiplier (float, *optional*): Custom multiplier for hidden
            dimension. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        multiple_of: Optional[int] = 256,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        self.swiglu = swiglu

        # custom hidden_size factor multiplier
        if ffn_dim_multiplier is not None:
            inner_dim = int(ffn_dim_multiplier * inner_dim)
        inner_dim = multiple_of * ((inner_dim + multiple_of - 1) // multiple_of)

        self.linear_1 = nn.Linear(
            dim,
            inner_dim,
            bias=False,
        )
        self.linear_2 = nn.Linear(
            inner_dim,
            dim,
            bias=False,
        )
        self.linear_3 = nn.Linear(
            dim,
            inner_dim,
            bias=False,
        )

    def forward(self, x):
        h1, h2 = self.linear_1(x), self.linear_3(x)
        swiglu_fn = self.swiglu
        return self.linear_2(swiglu_fn(h1, h2))


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = torch.nn.SiLU()

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = torch.nn.SiLU()

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.linear_1.weight, std=0.02)
        nn.init.zeros_(self.linear_1.bias)
        nn.init.normal_(self.linear_2.weight, std=0.02)
        nn.init.zeros_(self.linear_2.bias)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Lumina2CombinedTimestepCaptionEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        instruction_feat_dim: int = 2048,
        frequency_embedding_size: int = 256,
        norm_eps: float = 1e-5,
        timestep_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=frequency_embedding_size,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
            scale=timestep_scale,
        )

        self.timestep_embedder = TimestepEmbedding(
            in_channels=frequency_embedding_size, time_embed_dim=min(hidden_size, 1024)
        )

        self.caption_embedder = nn.Sequential(
            RMSNorm(instruction_feat_dim, eps=norm_eps),
            nn.Linear(instruction_feat_dim, hidden_size, bias=True),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.trunc_normal_(self.caption_embedder[1].weight, std=0.02)
        nn.init.zeros_(self.caption_embedder[1].bias)

    def forward(
        self,
        timestep: torch.Tensor,
        instruction_hidden_states: torch.Tensor,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        timestep_proj = self.time_proj(timestep).to(dtype=dtype)
        time_embed = self.timestep_embedder(timestep_proj)
        caption_embed = self.caption_embedder(instruction_hidden_states)
        return time_embed, caption_embed


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(
                -1
            )  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, Boogu and CogView4
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(
                -2
            )  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(
                f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2."
            )

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(
            x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        )
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class BooguImageAttnProcessor:
    """
    Processor for implementing scaled dot-product attention with flash attention and variable length sequences.

    This processor is optimized for PyTorch 2.0 and implements:
    - Flash attention with variable length sequences
    - Rotary position embeddings (RoPE)
    - Query-Key normalization
    - Proportional attention scaling

    Args:
        None

    Raises:
        ImportError: If PyTorch version is less than 2.0
    """

    def __init__(self) -> None:
        """Initialize the attention processor."""
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "BooguImageAttnProcessorFlash2Varlen requires PyTorch 2.0. "
                "Please upgrade PyTorch to version 2.0 or later."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        base_sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Process attention computation with flash attention.

        Args:
            attn: Attention module
            hidden_states: Hidden states tensor of shape (batch_size, seq_len, hidden_dim)
            encoder_hidden_states: Encoder hidden states tensor
            attention_mask: Optional attention mask tensor
            image_rotary_emb: Optional rotary embeddings for image tokens
            base_sequence_length: Optional base sequence length for proportional attention

        Returns:
            torch.Tensor: Processed hidden states after attention computation
        """
        batch_size, sequence_length, _ = hidden_states.shape

        # Get Query-Key-Value Pair
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype

        # Get key-value heads
        kv_heads = inner_dim // head_dim

        # Reshape tensors for attention computation
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        # Apply Query-Key normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply Rotary Position Embeddings
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=False)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=False)

        query, key = query.to(dtype), key.to(dtype)

        # Calculate attention scale
        if base_sequence_length is not None:
            softmax_scale = (
                math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
            )
        else:
            softmax_scale = attn.scale

        # sdpa expects attn_mask with shape (B, H, Q, K) as boolean (True keeps, False masks)
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            if attention_mask.dim() == 2:
                # Standard padding mask [B, L] -> [B, 1, 1, L]
                attention_mask = attention_mask.view(batch_size, 1, 1, -1)
            elif attention_mask.dim() == 3:
                # Robust causal + padding mask construction
                # Infer valid lengths from diagonal, then build lower-triangular mask within valid lengths
                B, L, _ = attention_mask.shape
                diag_valid = torch.diagonal(attention_mask, dim1=-2, dim2=-1)
                lengths = diag_valid.sum(dim=-1)  # [B]
                arange_L = torch.arange(L, device=attention_mask.device)
                # Padding masks for queries and keys: shape [B, L]
                q_valid = arange_L.unsqueeze(0) < lengths.unsqueeze(1)
                k_valid = q_valid  # same lengths assumed
                # Lower-triangular causal mask [L, L]
                causal = torch.tril(
                    torch.ones(L, L, dtype=torch.bool, device=attention_mask.device)
                )
                # Combine: [B, L, L]
                combined = causal & q_valid.unsqueeze(-1) & k_valid.unsqueeze(-2)
                attention_mask = combined.unsqueeze(1)  # [B, 1, L, L]
            else:
                raise ValueError(
                    f"Unsupported attention_mask shape: {attention_mask.shape}"
                )

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # explicitly repeat key and value to match query length, otherwise using enable_gqa=True results in MATH backend of sdpa in our test of pytorch2.6
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, scale=softmax_scale
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.type_as(query)

        # Apply output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class BooguImageDoubleStreamSelfAttnProcessor(nn.Module):
    """
    Double-stream self-attention processor without flash attention.

    This processor implements double-stream attention where:
    - Instruction and image features are processed separately to generate QKV
    - QKV are concatenated and processed together for cross-modal attention
    - Uses PyTorch's scaled_dot_product_attention for computation
    - Supports both standard and causal attention masks

    Args:
        head_dim: Dimension of each attention head
        num_attention_heads: Number of attention heads for queries
        num_kv_heads: Number of key-value heads
        qkv_bias: Whether to use bias in QKV linear layers
    """

    def __init__(
        self,
        head_dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        """Initialize the double-stream attention processor."""
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "BooguImageDoubleStreamSelfAttnProcessor requires PyTorch 2.0. "
                "Please upgrade PyTorch to version 2.0 or later."
            )

        # Calculate dimensions
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads

        query_dim = head_dim * num_attention_heads
        kv_dim = head_dim * num_kv_heads

        # Initialize separate Q, K, V linear layers for instruction and image
        # Query uses num_attention_heads, Key/Value use num_kv_heads
        self.img_to_q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.img_to_k = nn.Linear(query_dim, kv_dim, bias=qkv_bias)
        self.img_to_v = nn.Linear(query_dim, kv_dim, bias=qkv_bias)

        self.instruct_to_q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.instruct_to_k = nn.Linear(query_dim, kv_dim, bias=qkv_bias)
        self.instruct_to_v = nn.Linear(query_dim, kv_dim, bias=qkv_bias)

        # Additional output projection layers for instruction and image streams
        self.instruct_out = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.img_out = nn.Linear(query_dim, query_dim, bias=qkv_bias)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """
        Initialize the weights of the double-stream attention processor.

        Uses Xavier uniform initialization for linear layers and zero initialization for biases.
        """
        # Initialize image stream QKV projection layers
        nn.init.xavier_uniform_(self.img_to_q.weight)
        nn.init.xavier_uniform_(self.img_to_k.weight)
        nn.init.xavier_uniform_(self.img_to_v.weight)

        # Initialize instruction stream QKV projection layers
        nn.init.xavier_uniform_(self.instruct_to_q.weight)
        nn.init.xavier_uniform_(self.instruct_to_k.weight)
        nn.init.xavier_uniform_(self.instruct_to_v.weight)

        # Initialize separate output projection layers
        nn.init.xavier_uniform_(self.instruct_out.weight)
        nn.init.xavier_uniform_(self.img_out.weight)

        # Initialize biases if they exist
        if self.img_to_q.bias is not None:
            nn.init.zeros_(self.img_to_q.bias)
            nn.init.zeros_(self.img_to_k.bias)
            nn.init.zeros_(self.img_to_v.bias)
            nn.init.zeros_(self.instruct_to_q.bias)
            nn.init.zeros_(self.instruct_to_k.bias)
            nn.init.zeros_(self.instruct_to_v.bias)
            nn.init.zeros_(self.instruct_out.bias)
            nn.init.zeros_(self.img_out.bias)

    def _concat_instruction_image_features(
        self,
        img_hidden_states_list: List[torch.Tensor],
        instruct_hidden_states_list: List[torch.Tensor],
        encoder_seq_lengths: List[int],
        seq_lengths: List[int],
    ) -> List[torch.Tensor]:
        """
        Concatenate instruction (text & image) and reference image features (instruction first, then image).

        Args:
            img_hidden_states_list: List of image tensors [img_query, img_key, img_value]
            instruct_hidden_states_list: List of instruction tensors [instruct_query, instruct_key, instruct_value]
            encoder_seq_lengths: Instruction sequence lengths for each sample [B]
            seq_lengths: Total sequence lengths for each sample [B]

        Returns:
            List of concatenated tensors [query, key, value]
        """
        assert len(img_hidden_states_list) == len(instruct_hidden_states_list), (
            f"Length mismatch: img_list={len(img_hidden_states_list)}, instruct_list={len(instruct_hidden_states_list)}"
        )

        batch_size = img_hidden_states_list[0].shape[0]
        max_seq_len = max(seq_lengths)

        concatenated_list = []

        for img_tensor, instruct_tensor in zip(
            img_hidden_states_list, instruct_hidden_states_list
        ):
            # Ensure tensors are on the same device
            device = img_tensor.device
            if instruct_tensor.device != device:
                instruct_tensor = instruct_tensor.to(device)

            # Create output tensor with proper shape [B, max_seq_len, feature_dim]
            feature_dim = img_tensor.shape[-1]
            concatenated = img_tensor.new_zeros(batch_size, max_seq_len, feature_dim)

            # Concatenate instruction first, then image for each sample
            for i, (encoder_seq_len, seq_len) in enumerate(
                zip(encoder_seq_lengths, seq_lengths)
            ):
                # Place instruction tokens first
                concatenated[i, :encoder_seq_len] = instruct_tensor[i, :encoder_seq_len]
                # Place image tokens after instruction
                concatenated[i, encoder_seq_len:seq_len] = img_tensor[
                    i, : seq_len - encoder_seq_len
                ]

            concatenated_list.append(concatenated)

        return concatenated_list

    def _split_instruction_image_features(
        self,
        hidden_states_list: List[torch.Tensor],
        encoder_seq_lengths: List[int],
        seq_lengths: List[int],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Split concatenated features back to instruction and image features.
        Inverse operation of _concat_instruction_image_features.

        Args:
            hidden_states_list: List of concatenated tensors (usually just one element)
            encoder_seq_lengths: Instruction sequence lengths for each sample [B]
            seq_lengths: Total sequence lengths for each sample [B]

        Returns:
            List of tuples, each containing (instruct_hidden_states, img_hidden_states)
        """
        result_list = []

        for hidden_states in hidden_states_list:
            batch_size = hidden_states.shape[0]
            feature_dim = hidden_states.shape[-1]

            # Get maximum lengths for instruction and image
            max_instruct_len = max(encoder_seq_lengths)
            max_img_len = max(
                seq_len - encoder_seq_len
                for seq_len, encoder_seq_len in zip(seq_lengths, encoder_seq_lengths)
            )

            # Create output tensors [B, max_len, feature_dim]
            instruct_hidden_states = hidden_states.new_zeros(
                batch_size, max_instruct_len, feature_dim
            )
            img_hidden_states = hidden_states.new_zeros(
                batch_size, max_img_len, feature_dim
            )

            # Split each sample back to instruction and image
            for i, (encoder_seq_len, seq_len) in enumerate(
                zip(encoder_seq_lengths, seq_lengths)
            ):
                img_len = seq_len - encoder_seq_len

                # Extract instruction portion
                instruct_hidden_states[i, :encoder_seq_len] = hidden_states[
                    i, :encoder_seq_len
                ]
                # Extract image portion
                img_hidden_states[i, :img_len] = hidden_states[
                    i, encoder_seq_len:seq_len
                ]

            result_list.append((instruct_hidden_states, img_hidden_states))

        return result_list

    def __call__(
        self,
        attn: Attention,
        img_hidden_states: torch.Tensor,
        instruct_hidden_states: torch.Tensor,
        joint_attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        encoder_seq_lengths: List[
            int
        ] = None,  # [B] - Instruction sequence lengths for each sample
        seq_lengths: List[int] = None,  # [B] - Total sequence lengths for each sample
        base_sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Process double-stream self-attention computation with PyTorch's scaled_dot_product_attention.

        Args:
            attn: Attention module
            img_hidden_states: Image hidden states tensor [B, L_img, D]
            instruct_hidden_states: Instruction hidden states tensor [B, L_instruct, D]
            joint_attention_mask: Combined attention mask [B, L_total]
            rotary_emb: Rotary embeddings for the joint sequence
            encoder_seq_lengths: Instruction sequence lengths for each sample [B]
            seq_lengths: Total sequence lengths for each sample [B]
            base_sequence_length: Optional base sequence length for proportional attention

        Returns:
            torch.Tensor: Processed hidden states after attention computation
        """
        batch_size = img_hidden_states.shape[0]
        L_instruct = instruct_hidden_states.shape[1]
        L_img = img_hidden_states.shape[1]

        # Ensure Q, K, V linear layers are on the same device as input tensors
        device = img_hidden_states.device
        for layer in [
            self.img_to_q,
            self.img_to_k,
            self.img_to_v,
            self.instruct_to_q,
            self.instruct_to_k,
            self.instruct_to_v,
            self.instruct_out,
            self.img_out,
        ]:
            if (
                (layer.weight.device != device)
                and (str(layer.weight.device).lower() != "meta")
                and (str(device).lower() not in {"meta", "auto"})
            ):
                layer = layer.to(device)

        # Generate Q, K, V for image and instruction streams (NO head reshaping yet)
        img_query = self.img_to_q(img_hidden_states)  # [B, L_img, query_dim]
        img_key = self.img_to_k(img_hidden_states)  # [B, L_img, kv_dim]
        img_value = self.img_to_v(img_hidden_states)  # [B, L_img, kv_dim]

        instruct_query = self.instruct_to_q(
            instruct_hidden_states
        )  # [B, L_instruct, query_dim]
        instruct_key = self.instruct_to_k(
            instruct_hidden_states
        )  # [B, L_instruct, kv_dim]
        instruct_value = self.instruct_to_v(
            instruct_hidden_states
        )  # [B, L_instruct, kv_dim]

        # Use helper function to concatenate QKV (instruction first, then image)
        img_list = [img_query, img_key, img_value]  # [B, L_img, feature_dim] each
        instruct_list = [
            instruct_query,
            instruct_key,
            instruct_value,
        ]  # [B, L_instruct, feature_dim] each
        concatenated_list = self._concat_instruction_image_features(
            img_list, instruct_list, encoder_seq_lengths, seq_lengths
        )
        query, key, value = concatenated_list  # [B, max_seq_len, feature_dim] each

        # From here, follow exactly the same logic as BooguImageAttnProcessor
        sequence_length = max(seq_lengths)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype

        # Get key-value heads
        kv_heads = inner_dim // head_dim

        # Reshape tensors for attention computation
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        # Apply Query-Key normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply Rotary Position Embeddings
        if rotary_emb is not None:
            query = apply_rotary_emb(query, rotary_emb, use_real=False)
            key = apply_rotary_emb(key, rotary_emb, use_real=False)

        query, key = query.to(dtype), key.to(dtype)

        # Calculate attention scale
        if base_sequence_length is not None:
            softmax_scale = (
                math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
            )
        else:
            softmax_scale = attn.scale

        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        if joint_attention_mask is not None:
            joint_attention_mask = joint_attention_mask.bool()
            if joint_attention_mask.dim() == 2:
                # Standard mask [B, seq_len] -> [B, 1, 1, seq_len]
                joint_attention_mask = joint_attention_mask.view(batch_size, 1, 1, -1)
            elif joint_attention_mask.dim() == 3:
                # Causal mask [B, seq_len, seq_len] -> [B, 1, seq_len, seq_len]
                joint_attention_mask = joint_attention_mask.unsqueeze(1)
            else:
                raise ValueError(
                    f"Unsupported joint_attention_mask shape: {joint_attention_mask.shape}"
                )

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # explicitly repeat key and value to match query length, otherwise using enable_gqa=True results in MATH backend of sdpa in our test of pytorch2.6
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=joint_attention_mask, scale=softmax_scale
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.type_as(query)

        # Split hidden_states back to instruction and image, apply separate output projections, then merge
        split_results = self._split_instruction_image_features(
            [hidden_states], encoder_seq_lengths, seq_lengths
        )
        instruct_hidden_states, img_hidden_states = split_results[
            0
        ]  # [B, max_instruct_len, feature_dim], [B, max_img_len, feature_dim]

        # Apply separate output projections for instruction and image
        instruct_projected = self.instruct_out(
            instruct_hidden_states
        )  # [B, max_instruct_len, feature_dim]
        img_projected = self.img_out(img_hidden_states)  # [B, max_img_len, feature_dim]

        # Merge back to joint representation
        merged_list = self._concat_instruction_image_features(
            [img_projected], [instruct_projected], encoder_seq_lengths, seq_lengths
        )
        hidden_states = merged_list[0]  # [B, max_seq_len, feature_dim]

        # Apply final output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class PromptEmbedding(torch.nn.Module):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["BooguImageTransformerBlock"]
    _skip_layerwise_casting_patterns = ["prompt_token_embedding", "norm"]

    def __init__(self, prompt_tuning_configs):
        super().__init__()

        num_trainable_prompt_tokens = prompt_tuning_configs.get(
            "num_trainable_prompt_tokens", 32
        )
        hidden_size = prompt_tuning_configs.get("hidden_size", 2048)
        num_attention_heads = prompt_tuning_configs.get("num_attention_heads", 32)
        num_kv_heads = prompt_tuning_configs.get("num_kv_heads", 8)
        multiple_of = prompt_tuning_configs.get("multiple_of", 256)
        ffn_dim_multiplier = prompt_tuning_configs.get("ffn_dim_multiplier", None)
        norm_eps = prompt_tuning_configs.get("norm_eps", 1e-5)
        num_layers = prompt_tuning_configs.get("num_layers", 2)
        theta = prompt_tuning_configs.get("theta", 10000)

        self.register_to_config(
            num_trainable_prompt_tokens=num_trainable_prompt_tokens,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            norm_eps=norm_eps,
            num_layers=num_layers,
            theta=theta,
        )

        self.prompt_tuning_configs = prompt_tuning_configs

        prompt_emb_head_dim = self.config.hidden_size // self.config.num_attention_heads

        self.prompt_token_embedding = nn.Embedding(
            num_embeddings=self.config.num_trainable_prompt_tokens,
            embedding_dim=self.config.hidden_size,
        )

        # Rotary embedding for prompt tokens.
        self.prompt_rope_embedder = BooguImagePromptTuningRotaryPosEmbed(
            theta=self.config.theta,
            dim=prompt_emb_head_dim,
            num_trainable_prompt_tokens=self.config.num_trainable_prompt_tokens,
        )

        self.prompt_tuning_layers = nn.ModuleList(
            [
                BooguImageTransformerBlock(
                    dim=self.config.hidden_size,
                    num_attention_heads=self.config.num_attention_heads,
                    num_kv_heads=self.config.num_kv_heads,
                    multiple_of=self.config.multiple_of,
                    ffn_dim_multiplier=self.config.ffn_dim_multiplier,
                    norm_eps=self.config.norm_eps,
                    modulation=False,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self.gradient_checkpointing = False

        self.initialize_weights()

    def initialize_weights(self) -> None:
        # Small std keeps prompt tuning stable at init.
        nn.init.normal_(self.prompt_token_embedding.weight, mean=0.0, std=0.02)

    def forward(self, idx=None, batch_size=1, device=None, use_causal_mask=True):
        if idx is None:
            prompt_embeddings = self.prompt_token_embedding.weight
        else:
            prompt_embeddings = self.prompt_token_embedding(idx)

        # Expand to [B, num_tokens, hidden_dim].
        hidden_states = prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        rotary_emb, attention_mask = self.prompt_rope_embedder(
            batch_size, device, use_causal_mask
        )

        for i, layer in enumerate(self.prompt_tuning_layers):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    layer,
                    hidden_states,
                    attention_mask,
                    rotary_emb,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask,
                    rotary_emb,
                )
        return hidden_states

    @classmethod
    def from_config(cls, config, **kwargs):
        # `config` is loaded from config.json.
        instance = cls(prompt_tuning_configs=config)

        weight_dtype = kwargs.get("weight_dtype", None)
        if weight_dtype is not None:
            for p in instance.parameters():
                p.data = p.data.to(dtype=weight_dtype)

        return instance


class BooguImageTransformerBlock(nn.Module):
    """
    Basic Boogu-Image transformer block: attention + MLP + RMSNorm.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        modulation: bool = True,
    ) -> None:
        """Initialize the transformer block."""
        super().__init__()
        self.head_dim = dim // num_attention_heads
        self.modulation = modulation

        processor = BooguImageAttnProcessor()

        # Initialize attention layer
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // num_attention_heads,
            qk_norm="rms_norm",
            heads=num_attention_heads,
            kv_heads=num_kv_heads,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=processor,
        )

        # Initialize feed-forward network
        self.feed_forward = LuminaFeedForward(
            dim=dim,
            inner_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        # Initialize normalization layers
        if modulation:
            self.norm1 = LuminaRMSNormZero(
                embedding_dim=dim, norm_eps=norm_eps, norm_elementwise_affine=True
            )
        else:
            self.norm1 = RMSNorm(dim, eps=norm_eps)

        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize linear weights and modulation parameters."""
        nn.init.xavier_uniform_(self.attn.to_q.weight)
        nn.init.xavier_uniform_(self.attn.to_k.weight)
        nn.init.xavier_uniform_(self.attn.to_v.weight)
        nn.init.xavier_uniform_(self.attn.to_out[0].weight)

        nn.init.xavier_uniform_(self.feed_forward.linear_1.weight)
        nn.init.xavier_uniform_(self.feed_forward.linear_2.weight)
        nn.init.xavier_uniform_(self.feed_forward.linear_3.weight)

        if self.modulation:
            nn.init.zeros_(self.norm1.linear.weight)
            nn.init.zeros_(self.norm1.linear.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer block.

        Args:
            hidden_states: Input hidden states tensor
            attention_mask: Attention mask tensor
            image_rotary_emb: Rotary embeddings for image tokens
            temb: Optional timestep embedding tensor

        Returns:
            torch.Tensor: Output hidden states after transformer block processing
        """

        enable_taylorseer = getattr(self, "enable_taylorseer", False)

        if enable_taylorseer:
            if self.modulation:
                if temb is None:
                    raise ValueError("temb must be provided when modulation is enabled")

                if self.current["type"] == "full":
                    self.current["module"] = "total"
                    taylor_cache_init(cache_dic=self.cache_dic, current=self.current)

                    norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(
                        hidden_states, temb
                    )
                    attn_output = self.attn(
                        hidden_states=norm_hidden_states,
                        encoder_hidden_states=norm_hidden_states,
                        attention_mask=attention_mask,
                        image_rotary_emb=image_rotary_emb,
                    )
                    hidden_states = hidden_states + gate_msa.unsqueeze(
                        1
                    ).tanh() * self.norm2(attn_output)
                    mlp_output = self.feed_forward(
                        self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1))
                    )
                    hidden_states = hidden_states + gate_mlp.unsqueeze(
                        1
                    ).tanh() * self.ffn_norm2(mlp_output)

                    derivative_approximation(
                        cache_dic=self.cache_dic,
                        current=self.current,
                        feature=hidden_states,
                    )

                elif self.current["type"] == "Taylor":
                    self.current["module"] = "total"
                    hidden_states = taylor_formula(
                        cache_dic=self.cache_dic, current=self.current
                    )
            else:
                norm_hidden_states = self.norm1(hidden_states)
                attn_output = self.attn(
                    hidden_states=norm_hidden_states,
                    encoder_hidden_states=norm_hidden_states,
                    attention_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )
                hidden_states = hidden_states + self.norm2(attn_output)
                mlp_output = self.feed_forward(self.ffn_norm1(hidden_states))
                hidden_states = hidden_states + self.ffn_norm2(mlp_output)
        else:
            if self.modulation:
                if temb is None:
                    raise ValueError("temb must be provided when modulation is enabled")
                norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, temb
                )

                attn_output = self.attn(
                    hidden_states=norm_hidden_states,
                    encoder_hidden_states=norm_hidden_states,
                    attention_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )
                hidden_states = hidden_states + gate_msa.unsqueeze(
                    1
                ).tanh() * self.norm2(attn_output)
                mlp_output = self.feed_forward(
                    self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1))
                )
                hidden_states = hidden_states + gate_mlp.unsqueeze(
                    1
                ).tanh() * self.ffn_norm2(mlp_output)
            else:
                norm_hidden_states = self.norm1(hidden_states)
                attn_output = self.attn(
                    hidden_states=norm_hidden_states,
                    encoder_hidden_states=norm_hidden_states,
                    attention_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )
                hidden_states = hidden_states + self.norm2(attn_output)
                mlp_output = self.feed_forward(self.ffn_norm1(hidden_states))
                hidden_states = hidden_states + self.ffn_norm2(mlp_output)

        return hidden_states


class BooguImageDoubleStreamTransformerBlock(nn.Module):
    """
    Boogu-Image double-stream block.
    Here "double-stream" is the same idea as a "dual-stream" layer:
    instruction tokens and image tokens are processed in parallel streams.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        modulation: bool = True,
    ) -> None:
        """Initialize the double stream transformer block."""
        super().__init__()
        self.head_dim = dim // num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.modulation = modulation
        self.hidden_size = dim

        processor = BooguImageAttnProcessor()

        double_stream_processor = BooguImageDoubleStreamSelfAttnProcessor(
            head_dim=self.head_dim,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            qkv_bias=False,
        )

        # Image stream components.
        self.img_instruct_attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // num_attention_heads,
            qk_norm="rms_norm",
            heads=num_attention_heads,
            kv_heads=num_kv_heads,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=double_stream_processor,
        )

        self.img_self_attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // num_attention_heads,
            qk_norm="rms_norm",
            heads=num_attention_heads,
            kv_heads=num_kv_heads,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=processor,
        )

        self.img_feed_forward = LuminaFeedForward(
            dim=dim,
            inner_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        if modulation:
            # Image modulation terms: cross-attn, MLP, self-attn.
            self.img_norm1 = LuminaRMSNormZero(
                embedding_dim=dim, norm_eps=norm_eps, norm_elementwise_affine=True
            )
            self.img_norm2 = LuminaRMSNormZero(
                embedding_dim=dim, norm_eps=norm_eps, norm_elementwise_affine=True
            )
            self.img_norm3 = LuminaRMSNormZero(
                embedding_dim=dim, norm_eps=norm_eps, norm_elementwise_affine=True
            )
        else:
            self.img_norm1 = RMSNorm(dim, eps=norm_eps)
            self.img_norm2 = RMSNorm(dim, eps=norm_eps)
            self.img_norm3 = RMSNorm(dim, eps=norm_eps)

        self.img_ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.img_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.img_self_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.img_ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        # Instruction stream components.
        self.instruct_feed_forward = LuminaFeedForward(
            dim=dim,
            inner_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        if modulation:
            # Instruction modulation terms: cross-attn, MLP.
            self.instruct_norm1 = LuminaRMSNormZero(
                embedding_dim=dim, norm_eps=norm_eps, norm_elementwise_affine=True
            )
            self.instruct_norm2 = LuminaRMSNormZero(
                embedding_dim=dim, norm_eps=norm_eps, norm_elementwise_affine=True
            )
        else:
            self.instruct_norm1 = RMSNorm(dim, eps=norm_eps)
            self.instruct_norm2 = RMSNorm(dim, eps=norm_eps)

        self.instruct_ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.instruct_attn_norm = RMSNorm(dim, eps=norm_eps)
        self.instruct_ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.initialize_weights()

        # double_stream_processor owns its own q/k/v projections.
        for param in self.img_instruct_attn.to_q.parameters():
            param.requires_grad = False
        for param in self.img_instruct_attn.to_k.parameters():
            param.requires_grad = False
        for param in self.img_instruct_attn.to_v.parameters():
            param.requires_grad = False

        del self.img_instruct_attn.to_k
        del self.img_instruct_attn.to_v
        del self.img_instruct_attn.to_q

    def initialize_weights(self) -> None:
        """Initialize linear weights and modulation parameters."""
        nn.init.xavier_uniform_(self.img_instruct_attn.to_out[0].weight)

        # Keep Xavier init consistent across Boogu-Image blocks.
        nn.init.xavier_uniform_(self.img_self_attn.to_q.weight)
        nn.init.xavier_uniform_(self.img_self_attn.to_k.weight)
        nn.init.xavier_uniform_(self.img_self_attn.to_v.weight)
        nn.init.xavier_uniform_(self.img_self_attn.to_out[0].weight)

        nn.init.xavier_uniform_(self.img_feed_forward.linear_1.weight)
        nn.init.xavier_uniform_(self.img_feed_forward.linear_2.weight)
        nn.init.xavier_uniform_(self.img_feed_forward.linear_3.weight)

        nn.init.xavier_uniform_(self.instruct_feed_forward.linear_1.weight)
        nn.init.xavier_uniform_(self.instruct_feed_forward.linear_2.weight)
        nn.init.xavier_uniform_(self.instruct_feed_forward.linear_3.weight)

        # Initialize modulation parameters
        if self.modulation:
            nn.init.zeros_(self.img_norm1.linear.weight)
            nn.init.zeros_(self.img_norm1.linear.bias)
            nn.init.zeros_(self.img_norm2.linear.weight)
            nn.init.zeros_(self.img_norm2.linear.bias)
            nn.init.zeros_(self.img_norm3.linear.weight)
            nn.init.zeros_(self.img_norm3.linear.bias)

            nn.init.zeros_(self.instruct_norm1.linear.weight)
            nn.init.zeros_(self.instruct_norm1.linear.bias)
            nn.init.zeros_(self.instruct_norm2.linear.weight)
            nn.init.zeros_(self.instruct_norm2.linear.bias)

    def forward(
        self,
        img_hidden_states: torch.Tensor,  # [B, L_img, D] - Image tokens (ref_img + noise_img)
        instruct_hidden_states: torch.Tensor,  # [B, L_instruct, D] - Instruction tokens
        img_attention_mask: torch.Tensor,  # [B, L_img] - Attention mask for [ref_img + noise_img]
        joint_attention_mask: torch.Tensor,  # [B, L_total] - Combined attention mask for [instruct + img]
        image_rotary_emb: torch.Tensor,  # [B, L_img, head_dim] - Rotary embeddings for [ref_img + noise_img]
        rotary_emb: torch.Tensor,  # [B, L_total, head_dim] - Rotary embeddings for [instruct + img]
        temb: Optional[torch.Tensor] = None,  # [B, 1024] - Timestep embeddings
        encoder_seq_lengths: List[
            int
        ] = None,  # [B] - Instruction sequence lengths for each sample
        seq_lengths: List[int] = None,  # [B] - Total sequence lengths for each sample
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run one dual-stream (double-stream) block step.
        Returns updated `(img_hidden_states, instruct_hidden_states)`.
        """
        if self.modulation and temb is None:
            raise ValueError("temb must be provided when modulation is enabled")

        enable_taylorseer = getattr(self, "enable_taylorseer", False)
        if enable_taylorseer:
            self.current["module"] = "total"
            if self.current["type"] == "Taylor":
                return taylor_formula_4_double_stream(
                    cache_dic=self.cache_dic, current=self.current
                )
            if self.current["type"] == "full":
                taylor_cache_init(cache_dic=self.cache_dic, current=self.current)

        # Extract dimensions
        batch_size = img_hidden_states.shape[0]
        L_instruct = instruct_hidden_states.shape[1]  # Instruction sequence length
        L_img = img_hidden_states.shape[
            1
        ]  # Image sequence length (ref_img + noise_img)

        if self.modulation:
            # Step 1: modulation for both streams.
            img_norm1_out, img_gate_msa, img_scale_mlp, img_gate_mlp = self.img_norm1(
                img_hidden_states, temb
            )
            img_norm2_out, img_shift_mlp, _, _ = self.img_norm2(img_hidden_states, temb)
            img_norm3_out, img_gate_self, _, _ = self.img_norm3(img_hidden_states, temb)

            (
                instruct_norm1_out,
                instruct_gate_msa,
                instruct_scale_mlp,
                instruct_gate_mlp,
            ) = self.instruct_norm1(instruct_hidden_states, temb)
            instruct_norm2_out, instruct_shift_mlp, _, _ = self.instruct_norm2(
                instruct_hidden_states, temb
            )

            # Step 2: joint attention on [instruct + img].
            # Call processor directly because Attention.forward does not expose these dual-stream args.
            joint_attn_out = self.img_instruct_attn.processor(
                attn=self.img_instruct_attn,
                img_hidden_states=img_norm1_out,
                instruct_hidden_states=instruct_norm1_out,
                joint_attention_mask=joint_attention_mask,
                rotary_emb=rotary_emb,
                encoder_seq_lengths=encoder_seq_lengths,
                seq_lengths=seq_lengths,
            )

            # Split back into instruction/image segments.
            instruct_attn_out = instruct_hidden_states.new_zeros(
                batch_size, L_instruct, self.hidden_size
            )
            img_attn_out = img_hidden_states.new_zeros(
                batch_size, L_img, self.hidden_size
            )
            for i, (encoder_seq_len, seq_len) in enumerate(
                zip(encoder_seq_lengths, seq_lengths)
            ):
                instruct_attn_out[i, :encoder_seq_len] = joint_attn_out[
                    i, :encoder_seq_len
                ]
                img_attn_out[i, : seq_len - encoder_seq_len] = joint_attn_out[
                    i, encoder_seq_len:seq_len
                ]

            # Step 3: image self-attention.
            img_self_attn_out = self.img_self_attn(
                hidden_states=img_norm3_out,
                encoder_hidden_states=img_norm3_out,
                attention_mask=img_attention_mask,
                image_rotary_emb=image_rotary_emb,
            )

            # Step 4: residual updates.
            img_hidden_states = img_hidden_states + img_gate_msa.unsqueeze(
                1
            ).tanh() * self.img_attn_norm(img_attn_out)
            img_hidden_states = img_hidden_states + img_gate_self.unsqueeze(
                1
            ).tanh() * self.img_self_attn_norm(img_self_attn_out)

            img_mlp_input = (
                1 + img_scale_mlp.unsqueeze(1)
            ) * img_norm2_out + img_shift_mlp.unsqueeze(1)
            img_mlp_out = self.img_feed_forward(self.img_ffn_norm1(img_mlp_input))
            img_hidden_states = img_hidden_states + img_gate_mlp.unsqueeze(
                1
            ).tanh() * self.img_ffn_norm2(img_mlp_out)

            instruct_hidden_states = (
                instruct_hidden_states
                + instruct_gate_msa.unsqueeze(1).tanh()
                * self.instruct_attn_norm(instruct_attn_out)
            )

            instruct_mlp_input = (
                1 + instruct_scale_mlp.unsqueeze(1)
            ) * instruct_norm2_out + instruct_shift_mlp.unsqueeze(1)
            instruct_mlp_out = self.instruct_feed_forward(
                self.instruct_ffn_norm1(instruct_mlp_input)
            )
            instruct_hidden_states = (
                instruct_hidden_states
                + instruct_gate_mlp.unsqueeze(1).tanh()
                * self.instruct_ffn_norm2(instruct_mlp_out)
            )

        else:
            # Non-modulated branch used by context-style blocks.
            img_norm1_out = self.img_norm1(img_hidden_states)
            img_norm3_out = self.img_norm3(img_hidden_states)
            instruct_norm1_out = self.instruct_norm1(instruct_hidden_states)

            # Same processor path as above.
            joint_attn_out = self.img_instruct_attn.processor(
                attn=self.img_instruct_attn,
                img_hidden_states=img_norm1_out,
                instruct_hidden_states=instruct_norm1_out,
                joint_attention_mask=joint_attention_mask,
                rotary_emb=rotary_emb,
                encoder_seq_lengths=encoder_seq_lengths,
                seq_lengths=seq_lengths,
            )

            instruct_attn_out = instruct_hidden_states.new_zeros(
                batch_size, L_instruct, self.hidden_size
            )
            img_attn_out = img_hidden_states.new_zeros(
                batch_size, L_img, self.hidden_size
            )
            for i, (encoder_seq_len, seq_len) in enumerate(
                zip(encoder_seq_lengths, seq_lengths)
            ):
                instruct_attn_out[i, :encoder_seq_len] = joint_attn_out[
                    i, :encoder_seq_len
                ]
                img_attn_out[i, : seq_len - encoder_seq_len] = joint_attn_out[
                    i, encoder_seq_len:seq_len
                ]

            img_self_attn_out = self.img_self_attn(
                hidden_states=img_norm3_out,
                encoder_hidden_states=img_norm3_out,
                attention_mask=img_attention_mask,
                image_rotary_emb=image_rotary_emb,
            )

            img_hidden_states = img_hidden_states + self.img_attn_norm(img_attn_out)
            img_hidden_states = img_hidden_states + self.img_self_attn_norm(
                img_self_attn_out
            )
            img_norm2_out = self.img_norm2(img_hidden_states)
            img_mlp_out = self.img_feed_forward(self.img_ffn_norm1(img_norm2_out))
            img_hidden_states = img_hidden_states + self.img_ffn_norm2(img_mlp_out)

            instruct_hidden_states = instruct_hidden_states + self.instruct_attn_norm(
                instruct_attn_out
            )
            instruct_norm2_out = self.instruct_norm2(instruct_hidden_states)
            instruct_mlp_out = self.instruct_feed_forward(
                self.instruct_ffn_norm1(instruct_norm2_out)
            )
            instruct_hidden_states = instruct_hidden_states + self.instruct_ffn_norm2(
                instruct_mlp_out
            )

        if enable_taylorseer and self.current["type"] == "full":
            derivative_approximation_4_double_stream(
                cache_dic=self.cache_dic,
                current=self.current,
                feature=(img_hidden_states, instruct_hidden_states),
            )

        return img_hidden_states, instruct_hidden_states


BooguImageSingleStreamTransformerBlock = BooguImageTransformerBlock


@dataclass
class BooguImageDiTConfig:
    patch_size: int = 2
    in_channels: int = 16
    out_channels: Optional[int] = None
    hidden_size: int = 3360
    num_layers: int = 40
    num_double_stream_layers: int = 8
    num_refiner_layers: int = 2
    num_attention_heads: int = 28
    num_kv_heads: int = 7
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    axes_dim_rope: Tuple[int, int, int] = (40, 40, 40)
    axes_lens: Tuple[int, int, int] = (2048, 1664, 1664)
    instruction_feature_configs: Dict[str, Any] = None
    prompt_tuning_configs: Dict[str, Any] = None
    timestep_scale: float = 1000.0


class BooguImageDiT(torch.nn.Module):
    """
    Boogu-Image transformer with mixed stream topology.
    Early layers use double-stream (aka dual-stream) processing, then switch
    to single-stream joint processing.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = [
        "BooguImageTransformerBlock",
        "BooguImageSingleStreamTransformerBlock",
        "BooguImageDoubleStreamTransformerBlock",
        "PromptEmbedding",
        "nn.Embedding",
    ]
    _repeated_blocks = [
        "BooguImageTransformerBlock",
        "BooguImageSingleStreamTransformerBlock",
        "BooguImageDoubleStreamTransformerBlock",
    ]
    _skip_layerwise_casting_patterns = ["x_embedder", "norm", "embedding"]

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        out_channels: Optional[int] = None,
        hidden_size: int = 3360,
        num_layers: int = 40,
        num_double_stream_layers: int = 8,
        num_refiner_layers: int = 2,
        num_attention_heads: int = 28,
        num_kv_heads: int = 7,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        axes_dim_rope: Tuple[int, int, int] = (40, 40, 40),
        axes_lens: Tuple[int, int, int] = (2048, 1664, 1664),
        # instruction_feat_dim: int = 1024,
        instruction_feature_configs: Dict[str, Any] = {'instruction_feat_dim': 4096, 'num_instruction_feature_layers': 1, 'reduce_type': 'mean'},
        prompt_tuning_configs: Dict[str, Any] = {'ffn_dim_multiplier': None, 'hidden_size': 4096, 'multiple_of': 256, 'norm_eps': 1e-05, 'num_attention_heads': 32, 'num_kv_heads': 8, 'num_layers': 0, 'num_trainable_prompt_tokens': 0, 'use_causal_mask': True, 'use_prompt_tuning': False},
        timestep_scale: float = 1000.0,
    ) -> None:
        """Initialize the Boogu-Image mixed single-double stream transformer model."""
        super().__init__()
        self.config = BooguImageDiTConfig(
            patch_size,
            in_channels,
            out_channels,
            hidden_size,
            num_layers,
            num_double_stream_layers,
            num_refiner_layers,
            num_attention_heads,
            num_kv_heads,
            multiple_of,
            ffn_dim_multiplier,
            norm_eps,
            axes_dim_rope,
            axes_lens,
            instruction_feature_configs,
            prompt_tuning_configs,
            timestep_scale,
        )

        # Validate configuration
        if (hidden_size // num_attention_heads) != sum(axes_dim_rope):
            raise ValueError(
                f"hidden_size // num_attention_heads ({hidden_size // num_attention_heads}) "
                f"must equal sum(axes_dim_rope) ({sum(axes_dim_rope)})"
            )

        if num_double_stream_layers > num_layers:
            raise ValueError(
                f"num_double_stream_layers ({num_double_stream_layers}) cannot be greater than "
                f"num_layers ({num_layers})"
            )

        self.out_channels = out_channels or in_channels
        self.num_double_stream_layers = num_double_stream_layers
        self.num_single_stream_layers = num_layers - num_double_stream_layers
        self.instruction_feature_configs = instruction_feature_configs
        self.prompt_tuning_configs = prompt_tuning_configs
        self.preprocessed_instruction_feat_dim = (
            self.cal_preprocessed_instruction_feat_dim(instruction_feature_configs)
        )

        # Initialize embeddings
        self.rope_embedder = BooguImageDoubleStreamRotaryPosEmbed(
            theta=10000,
            axes_dim=axes_dim_rope,
            axes_lens=axes_lens,
            patch_size=patch_size,
        )

        self.x_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
        )

        self.ref_image_patch_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
        )

        self.time_caption_embed = Lumina2CombinedTimestepCaptionEmbedding(
            hidden_size=hidden_size,
            instruction_feat_dim=self.preprocessed_instruction_feat_dim,
            norm_eps=norm_eps,
            timestep_scale=timestep_scale,
        )

        # Refiner layers.
        self.noise_refiner = nn.ModuleList(
            [
                BooguImageTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        self.ref_image_refiner = nn.ModuleList(
            [
                BooguImageTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        self.context_refiner = nn.ModuleList(
            [
                BooguImageTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=False,
                )
                for _ in range(num_refiner_layers)
            ]
        )

        # Mixed architecture: dual-stream first, then single-stream.
        # Here "double-stream" and "dual-stream" mean the same thing.
        self.double_stream_layers = nn.ModuleList(
            [
                BooguImageDoubleStreamTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(num_double_stream_layers)
            ]
        )

        # Single-stream layers process the fused sequence.
        self.single_stream_layers = nn.ModuleList(
            [
                BooguImageSingleStreamTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    modulation=True,
                )
                for _ in range(self.num_single_stream_layers)
            ]
        )

        # Output norm and projection.
        self.norm_out = LuminaLayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_embedding_dim=min(hidden_size, 1024),
            elementwise_affine=False,
            eps=1e-6,
            bias=True,
            out_dim=patch_size * patch_size * self.out_channels,
        )

        # Distinguish multiple reference images.
        self.image_index_embedding = nn.Parameter(
            torch.randn(5, hidden_size)
        )  # support max 5 ref images

        self.gradient_checkpointing = False

        self.initialize_weights()

        # TeaCache settings
        self.enable_teacache = False
        self.enable_taylorseer = False
        self.enable_teacache_for_all_layers = False
        self.enable_taylorseer_for_all_layers = False
        self.teacache_rel_l1_thresh = 0.05
        self.teacache_params = None

        coefficients = [-5.48259225, 11.48772289, -4.47407401, 2.47730926, -0.03316487]
        self.rescale_func = np.poly1d(coefficients)

        self.layers = list(self.double_stream_layers) + list(self.single_stream_layers)

    def initialize_weights(self) -> None:
        """
        Initialize the weights of the model.

        Uses Xavier uniform initialization for linear layers.
        """
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.constant_(self.x_embedder.bias, 0.0)

        nn.init.xavier_uniform_(self.ref_image_patch_embedder.weight)
        nn.init.constant_(self.ref_image_patch_embedder.bias, 0.0)

        nn.init.zeros_(self.norm_out.linear_1.weight)
        nn.init.zeros_(self.norm_out.linear_1.bias)
        nn.init.zeros_(self.norm_out.linear_2.weight)
        nn.init.zeros_(self.norm_out.linear_2.bias)

        nn.init.normal_(self.image_index_embedding, std=0.02)

    def img_patch_embed_and_refine(
        self,
        hidden_states,
        ref_image_hidden_states,
        padded_img_mask,
        padded_ref_img_mask,
        noise_rotary_emb,
        ref_img_rotary_emb,
        l_effective_ref_img_len,
        l_effective_img_len,
        temb,
    ):
        """Embed image patches and run the refiner blocks."""
        batch_size = len(hidden_states)
        max_combined_img_len = max(
            [
                img_len + sum(ref_img_len)
                for img_len, ref_img_len in zip(
                    l_effective_img_len, l_effective_ref_img_len
                )
            ]
        )

        hidden_states = self.x_embedder(hidden_states)
        ref_image_hidden_states = self.ref_image_patch_embedder(ref_image_hidden_states)

        for i in range(batch_size):
            shift = 0
            for j, ref_img_len in enumerate(l_effective_ref_img_len[i]):
                ref_image_hidden_states[i, shift : shift + ref_img_len, :] = (
                    ref_image_hidden_states[i, shift : shift + ref_img_len, :]
                    + self.image_index_embedding[j]
                )
                shift += ref_img_len

        for layer in self.noise_refiner:
            hidden_states = layer(
                hidden_states, padded_img_mask, noise_rotary_emb, temb
            )

        flat_l_effective_ref_img_len = list(itertools.chain(*l_effective_ref_img_len))
        num_ref_images = len(flat_l_effective_ref_img_len)
        max_ref_img_len = max(flat_l_effective_ref_img_len)

        batch_ref_img_mask = ref_image_hidden_states.new_zeros(
            num_ref_images, max_ref_img_len, dtype=torch.bool
        )
        batch_ref_image_hidden_states = ref_image_hidden_states.new_zeros(
            num_ref_images, max_ref_img_len, self.config.hidden_size
        )
        batch_ref_img_rotary_emb = hidden_states.new_zeros(
            num_ref_images,
            max_ref_img_len,
            ref_img_rotary_emb.shape[-1],
            dtype=ref_img_rotary_emb.dtype,
        )
        batch_temb = temb.new_zeros(num_ref_images, *temb.shape[1:], dtype=temb.dtype)

        # Flatten reference images into a temporary batch.
        idx = 0
        for i in range(batch_size):
            shift = 0
            for ref_img_len in l_effective_ref_img_len[i]:
                batch_ref_img_mask[idx, :ref_img_len] = True
                batch_ref_image_hidden_states[idx, :ref_img_len] = (
                    ref_image_hidden_states[i, shift : shift + ref_img_len]
                )
                batch_ref_img_rotary_emb[idx, :ref_img_len] = ref_img_rotary_emb[
                    i, shift : shift + ref_img_len
                ]
                batch_temb[idx] = temb[i]
                shift += ref_img_len
                idx += 1

        # Refine each reference-image sample.
        for layer in self.ref_image_refiner:
            batch_ref_image_hidden_states = layer(
                batch_ref_image_hidden_states,
                batch_ref_img_mask,
                batch_ref_img_rotary_emb,
                batch_temb,
            )

        # Restore reference-image sequence layout.
        idx = 0
        for i in range(batch_size):
            shift = 0
            for ref_img_len in l_effective_ref_img_len[i]:
                ref_image_hidden_states[i, shift : shift + ref_img_len] = (
                    batch_ref_image_hidden_states[idx, :ref_img_len]
                )
                shift += ref_img_len
                idx += 1

        combined_img_hidden_states = hidden_states.new_zeros(
            batch_size, max_combined_img_len, self.config.hidden_size
        )
        for i, (ref_img_len, img_len) in enumerate(
            zip(l_effective_ref_img_len, l_effective_img_len)
        ):
            combined_img_hidden_states[i, : sum(ref_img_len)] = ref_image_hidden_states[
                i, : sum(ref_img_len)
            ]
            combined_img_hidden_states[
                i, sum(ref_img_len) : sum(ref_img_len) + img_len
            ] = hidden_states[i, :img_len]

        return combined_img_hidden_states

    def flat_and_pad_to_seq(self, hidden_states, ref_image_hidden_states):
        """Flatten patch tokens and pad to batched sequences."""
        batch_size = len(hidden_states)
        p = self.config.patch_size
        device = hidden_states[0].device

        img_sizes = [(img.size(1), img.size(2)) for img in hidden_states]
        l_effective_img_len = [(H // p) * (W // p) for (H, W) in img_sizes]

        if ref_image_hidden_states is not None:
            ref_img_sizes = [
                [(img.size(1), img.size(2)) for img in imgs]
                if imgs is not None
                else None
                for imgs in ref_image_hidden_states
            ]
            l_effective_ref_img_len = [
                [
                    (ref_img_size[0] // p) * (ref_img_size[1] // p)
                    for ref_img_size in _ref_img_sizes
                ]
                if _ref_img_sizes is not None
                else [0]
                for _ref_img_sizes in ref_img_sizes
            ]
        else:
            ref_img_sizes = [None for _ in range(batch_size)]
            l_effective_ref_img_len = [[0] for _ in range(batch_size)]

        max_ref_img_len = max(
            [sum(ref_img_len) for ref_img_len in l_effective_ref_img_len]
        )
        max_img_len = max(l_effective_img_len)

        # Reference-image patch embeddings.
        flat_ref_img_hidden_states = []
        for i in range(batch_size):
            if ref_img_sizes[i] is not None:
                imgs = []
                for ref_img in ref_image_hidden_states[i]:
                    C, H, W = ref_img.size()
                    ref_img = rearrange(
                        ref_img, "c (h p1) (w p2) -> (h w) (p1 p2 c)", p1=p, p2=p
                    )
                    imgs.append(ref_img)

                img = torch.cat(imgs, dim=0)
                flat_ref_img_hidden_states.append(img)
            else:
                flat_ref_img_hidden_states.append(None)

        # Noise-image patch embeddings.
        flat_hidden_states = []
        for i in range(batch_size):
            img = hidden_states[i]
            C, H, W = img.size()

            img = rearrange(img, "c (h p1) (w p2) -> (h w) (p1 p2 c)", p1=p, p2=p)
            flat_hidden_states.append(img)

        padded_ref_img_hidden_states = torch.zeros(
            batch_size,
            max_ref_img_len,
            flat_hidden_states[0].shape[-1],
            device=device,
            dtype=flat_hidden_states[0].dtype,
        )
        padded_ref_img_mask = torch.zeros(
            batch_size, max_ref_img_len, dtype=torch.bool, device=device
        )
        for i in range(batch_size):
            if ref_img_sizes[i] is not None:
                padded_ref_img_hidden_states[i, : sum(l_effective_ref_img_len[i])] = (
                    flat_ref_img_hidden_states[i]
                )
                padded_ref_img_mask[i, : sum(l_effective_ref_img_len[i])] = True

        padded_hidden_states = torch.zeros(
            batch_size,
            max_img_len,
            flat_hidden_states[0].shape[-1],
            device=device,
            dtype=flat_hidden_states[0].dtype,
        )
        padded_img_mask = torch.zeros(
            batch_size, max_img_len, dtype=torch.bool, device=device
        )
        for i in range(batch_size):
            padded_hidden_states[i, : l_effective_img_len[i]] = flat_hidden_states[i]
            padded_img_mask[i, : l_effective_img_len[i]] = True

        return (
            padded_hidden_states,
            padded_ref_img_hidden_states,
            padded_img_mask,
            padded_ref_img_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
        )

    def cal_preprocessed_instruction_feat_dim(
        self, instruction_feature_configs: Dict[str, Any]
    ):
        num_instruction_feat_layers = max(
            instruction_feature_configs.get("num_instruction_feat_layers", 1), 1
        )
        instruction_feat_dim = instruction_feature_configs.get(
            "instruction_feat_dim", 4096
        )
        reduce_type = instruction_feature_configs.get("reduce_type", "concat")
        if "cat" in reduce_type.lower():
            return num_instruction_feat_layers * instruction_feat_dim
        elif "mean" in reduce_type.lower():
            return instruction_feat_dim
        else:
            raise ValueError(f"Invalid reduce_type: {reduce_type}")

    def preprocess_instruction_hidden_states(
        self, raw_instruction_hidden_states, instruction_feature_configs: Dict[str, Any]
    ):
        num_instruction_feat_layers = max(
            instruction_feature_configs.get("num_instruction_feat_layers", 1), 1
        )
        instruction_feat_dim = instruction_feature_configs.get(
            "instruction_feat_dim", 4096
        )
        reduce_type = instruction_feature_configs.get("reduce_type", "concat")

        instruction_hidden_states = None
        if isinstance(raw_instruction_hidden_states, torch.Tensor):
            instruction_hidden_states = raw_instruction_hidden_states
        elif isinstance(raw_instruction_hidden_states, (list, tuple)):
            assert len(raw_instruction_hidden_states) == num_instruction_feat_layers
            if "cat" in reduce_type.lower():
                instruction_hidden_states = torch.cat(
                    raw_instruction_hidden_states, dim=-1
                )
            elif "mean" in reduce_type.lower():
                instruction_hidden_states = torch.mean(
                    torch.stack(raw_instruction_hidden_states), dim=0
                )
            else:
                raise ValueError(f"Invalid reduce_type: {reduce_type}")
        else:
            raise ValueError(
                f"Invalid type of raw_instruction_hidden_states, expected torch.Tensor or list, but got {type(raw_instruction_hidden_states)}"
            )

        assert (
            self.preprocessed_instruction_feat_dim
            == instruction_hidden_states.shape[-1]
        )

        return instruction_hidden_states

    def forward(
        self,
        hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        timestep: torch.Tensor,
        instruction_hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        instruction_attention_mask: torch.Tensor,
        ref_image_hidden_states: Optional[List[List[torch.Tensor]]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, torch.Tensor]:
        """
        Forward pass:
        context/refiner -> dual-stream (double-stream) -> fusion -> single-stream -> projection.
        """
        instruction_hidden_states = self.preprocess_instruction_hidden_states(
            instruction_hidden_states, self.instruction_feature_configs
        )

        enable_taylorseer = getattr(self, "enable_taylorseer", False)

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if False:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            # scale_lora_layers(self, lora_scale)
            pass
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                # logger.warning(
                #     "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                # )
                pass

        # === 1. Initial processing (same as original Boogu-Image) ===
        batch_size = len(hidden_states)
        is_hidden_states_tensor = isinstance(hidden_states, torch.Tensor)

        if is_hidden_states_tensor:
            assert hidden_states.ndim == 4
            hidden_states = [_hidden_states for _hidden_states in hidden_states]

        device = hidden_states[0].device

        # Timestep and instruction embedding.
        temb, instruction_hidden_states = self.time_caption_embed(
            timestep, instruction_hidden_states, hidden_states[0].dtype
        )

        # Flatten and pad token sequences.
        (
            hidden_states,
            ref_image_hidden_states,
            img_mask,
            ref_img_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
        ) = self.flat_and_pad_to_seq(hidden_states, ref_image_hidden_states)

        # Build rotary embeddings and sequence lengths.
        (
            context_rotary_emb,
            ref_img_rotary_emb,
            noise_rotary_emb,
            rotary_emb,
            encoder_seq_lengths,
            seq_lengths,
            combined_img_rotary_emb,
            combined_img_seq_lengths,
        ) = self.rope_embedder(
            freqs_cis,
            instruction_attention_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
            device,
        )

        # Context refinement.
        for layer in self.context_refiner:
            instruction_hidden_states = layer(
                instruction_hidden_states,
                instruction_attention_mask,
                context_rotary_emb,
            )

        # Image patch embedding and refinement.
        combined_img_hidden_states = self.img_patch_embed_and_refine(
            hidden_states,
            ref_image_hidden_states,
            img_mask,
            ref_img_mask,
            noise_rotary_emb,
            ref_img_rotary_emb,
            l_effective_ref_img_len,
            l_effective_img_len,
            temb,
        )

        # Dual-stream (double-stream) stage.
        instruct_hidden_states = instruction_hidden_states
        img_hidden_states = combined_img_hidden_states

        # Joint mask for [instruct + image].
        max_seq_len = max(seq_lengths)
        joint_attention_mask = hidden_states.new_zeros(
            batch_size, max_seq_len, dtype=torch.bool
        )
        for i, seq_len in enumerate(seq_lengths):
            joint_attention_mask[i, :seq_len] = True

        # Run dual-stream blocks.
        if self.num_double_stream_layers > 0:
            # Image-only mask for [ref + noise].
            max_img_len = max(combined_img_seq_lengths)
            img_attention_mask = hidden_states.new_zeros(
                batch_size, max_img_len, dtype=torch.bool
            )
            for i, img_seq_len in enumerate(combined_img_seq_lengths):
                img_attention_mask[i, :img_seq_len] = True

            enable_double_stream_taylorseer = (
                enable_taylorseer and self.enable_taylorseer_for_all_layers
            )
            enable_double_stream_teacache = (
                self.enable_teacache and self.enable_teacache_for_all_layers
            )

            if enable_double_stream_teacache:
                first_double_stream_layer = self.double_stream_layers[0]
                img_modulated_inp, _, _, _ = first_double_stream_layer.img_norm1(
                    img_hidden_states.clone(), temb
                )
                instruct_modulated_inp, _, _, _ = (
                    first_double_stream_layer.instruct_norm1(
                        instruct_hidden_states.clone(), temb
                    )
                )
                previous_double_modulated_inp = getattr(
                    self.teacache_params, "previous_double_modulated_inp", None
                )
                if (
                    self.teacache_params.is_first_or_last_step
                    or previous_double_modulated_inp is None
                ):
                    should_calc_double_stream = True
                    self.teacache_params.double_accumulated_rel_l1_distance = 0
                else:
                    img_rel_l1 = (
                        img_modulated_inp - previous_double_modulated_inp[0]
                    ).abs().mean() / previous_double_modulated_inp[0].abs().mean()
                    instruct_rel_l1 = (
                        instruct_modulated_inp - previous_double_modulated_inp[1]
                    ).abs().mean() / previous_double_modulated_inp[1].abs().mean()
                    rel_l1 = (img_rel_l1 + instruct_rel_l1) * 0.5
                    self.teacache_params.double_accumulated_rel_l1_distance += (
                        self.rescale_func(rel_l1.cpu().item())
                    )
                    if (
                        self.teacache_params.double_accumulated_rel_l1_distance
                        < self.teacache_rel_l1_thresh
                    ):
                        should_calc_double_stream = False
                    else:
                        should_calc_double_stream = True
                        self.teacache_params.double_accumulated_rel_l1_distance = 0
                self.teacache_params.previous_double_modulated_inp = (
                    img_modulated_inp,
                    instruct_modulated_inp,
                )
            else:
                should_calc_double_stream = True

            if enable_double_stream_teacache and not should_calc_double_stream:
                img_residual, instruct_residual = (
                    self.teacache_params.previous_double_residual
                )
                img_hidden_states = img_hidden_states + img_residual
                instruct_hidden_states = instruct_hidden_states + instruct_residual
            else:
                if enable_double_stream_taylorseer:
                    self.current["stream"] = "double_stream_layers"

                if enable_double_stream_teacache:
                    ori_img_hidden_states = img_hidden_states.clone()
                    ori_instruct_hidden_states = instruct_hidden_states.clone()

                for layer_idx, layer in enumerate(self.double_stream_layers):
                    if enable_double_stream_taylorseer:
                        layer.current = self.current
                        layer.cache_dic = self.cache_dic
                        layer.enable_taylorseer = True
                        self.current["layer"] = layer_idx
                    else:
                        layer.enable_taylorseer = False

                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        img_hidden_states, instruct_hidden_states = (
                            self._gradient_checkpointing_func(
                                layer,
                                img_hidden_states,
                                instruct_hidden_states,
                                img_attention_mask,
                                joint_attention_mask,
                                combined_img_rotary_emb,
                                rotary_emb,
                                temb,
                                encoder_seq_lengths,
                                seq_lengths,
                            )
                        )
                    else:
                        img_hidden_states, instruct_hidden_states = layer(
                            img_hidden_states,
                            instruct_hidden_states,
                            img_attention_mask,
                            joint_attention_mask,
                            combined_img_rotary_emb,
                            rotary_emb,
                            temb,
                            encoder_seq_lengths,
                            seq_lengths,
                        )

                if enable_double_stream_teacache:
                    self.teacache_params.previous_double_residual = (
                        img_hidden_states - ori_img_hidden_states,
                        instruct_hidden_states - ori_instruct_hidden_states,
                    )

        # Fuse streams to joint sequence.
        joint_hidden_states = hidden_states.new_zeros(
            batch_size, max(seq_lengths), self.config.hidden_size
        )
        for i, (encoder_seq_len, seq_len) in enumerate(
            zip(encoder_seq_lengths, seq_lengths)
        ):
            joint_hidden_states[i, :encoder_seq_len] = instruct_hidden_states[
                i, :encoder_seq_len
            ]
            joint_hidden_states[i, encoder_seq_len:seq_len] = img_hidden_states[
                i, : seq_len - encoder_seq_len
            ]

        # Single-stream stage.
        hidden_states = joint_hidden_states

        # TeaCache optimization.
        if self.enable_teacache and len(self.single_stream_layers) > 0:
            teacache_hidden_states = hidden_states.clone()
            teacache_temb = temb.clone()
            modulated_inp, _, _, _ = self.single_stream_layers[0].norm1(
                teacache_hidden_states, teacache_temb
            )
            if self.teacache_params.is_first_or_last_step:
                should_calc = True
                self.teacache_params.accumulated_rel_l1_distance = 0
            else:
                self.teacache_params.accumulated_rel_l1_distance += self.rescale_func(
                    (
                        (modulated_inp - self.teacache_params.previous_modulated_inp)
                        .abs()
                        .mean()
                        / self.teacache_params.previous_modulated_inp.abs().mean()
                    )
                    .cpu()
                    .item()
                )
                if (
                    self.teacache_params.accumulated_rel_l1_distance
                    < self.teacache_rel_l1_thresh
                ):
                    should_calc = False
                else:
                    should_calc = True
                    self.teacache_params.accumulated_rel_l1_distance = 0
            self.teacache_params.previous_modulated_inp = modulated_inp
        else:
            should_calc = True

        if self.enable_teacache and not should_calc:
            hidden_states += self.teacache_params.previous_residual
        else:
            if enable_taylorseer:
                self.current["stream"] = "single_stream_layers"

            if self.enable_teacache:
                ori_hidden_states = hidden_states.clone()

            for layer_idx, layer in enumerate(self.single_stream_layers):
                if enable_taylorseer:
                    layer.current = self.current
                    layer.cache_dic = self.cache_dic
                    layer.enable_taylorseer = True
                    self.current["layer"] = self.num_double_stream_layers + layer_idx

                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = self._gradient_checkpointing_func(
                        layer, hidden_states, joint_attention_mask, rotary_emb, temb
                    )
                else:
                    hidden_states = layer(
                        hidden_states, joint_attention_mask, rotary_emb, temb
                    )

            if self.enable_teacache:
                self.teacache_params.previous_residual = (
                    hidden_states - ori_hidden_states
                )

        # Output projection.
        hidden_states = self.norm_out(hidden_states, temb)

        # Reshape back to image format.
        p = self.config.patch_size
        output = []
        for i, (img_size, img_len, seq_len) in enumerate(
            zip(img_sizes, l_effective_img_len, seq_lengths)
        ):
            height, width = img_size
            img_tokens = hidden_states[i][seq_len - img_len : seq_len]
            img_output = rearrange(
                img_tokens,
                "(h w) (p1 p2 c) -> c (h p1) (w p2)",
                h=height // p,
                w=width // p,
                p1=p,
                p2=p,
            )
            output.append(img_output)

        if is_hidden_states_tensor:
            output = torch.stack(output, dim=0)

        # TaylorSeer step counter.
        if enable_taylorseer:
            self.current["step"] += 1

        return output
