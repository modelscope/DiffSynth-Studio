# Copyright 2025 The ACESTEO Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..core.attention.attention import attention_forward
from ..core import gradient_checkpoint_forward

from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutput
from transformers.processing_utils import Unpack
from transformers.utils import logging

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

logger = logging.get_logger(__name__)


def create_4d_mask(
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask: Optional[torch.Tensor] = None,  # [Batch, Seq_Len]
    sliding_window: Optional[int] = None,
    is_sliding_window: bool = False,
    is_causal: bool = True,
) -> torch.Tensor:
    """
    General 4D Attention Mask generator compatible with CPU/Mac/SDPA and Eager mode.
    Supports use cases:
    1. Causal Full: is_causal=True, is_sliding_window=False (standard GPT)
    2. Causal Sliding: is_causal=True, is_sliding_window=True (Mistral/Qwen local window)
    3. Bidirectional Full: is_causal=False, is_sliding_window=False (BERT/Encoder)
    4. Bidirectional Sliding: is_causal=False, is_sliding_window=True (Longformer local)

    Returns:
        [Batch, 1, Seq_Len, Seq_Len] additive mask (0.0 for keep, -inf for mask)
    """
    # ------------------------------------------------------
    # 1. Construct basic geometry mask [Seq_Len, Seq_Len]
    # ------------------------------------------------------

    # Build index matrices
    # i (Query): [0, 1, ..., L-1]
    # j (Key):   [0, 1, ..., L-1]
    indices = torch.arange(seq_len, device=device)
    # diff = i - j
    diff = indices.unsqueeze(1) - indices.unsqueeze(0)

    # Initialize all True (all positions visible)
    valid_mask = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool)

    # (A) Handle causality (Causal)
    if is_causal:
        # i >= j  =>  diff >= 0
        valid_mask = valid_mask & (diff >= 0)

    # (B) Handle sliding window
    if is_sliding_window and sliding_window is not None:
        if is_causal:
            # Causal sliding: only attend to past window steps
            # i - j <= window  =>  diff <= window
            # (diff >= 0 already handled above)
            valid_mask = valid_mask & (diff <= sliding_window)
        else:
            # Bidirectional sliding: attend past and future window steps
            # |i - j| <= window  =>  abs(diff) <= sliding_window
            valid_mask = valid_mask & (torch.abs(diff) <= sliding_window)

    # Expand dimensions to [1, 1, Seq_Len, Seq_Len] for broadcasting
    valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)

    # ------------------------------------------------------
    # 2. Apply padding mask (Key Masking)
    # ------------------------------------------------------
    if attention_mask is not None:
        # attention_mask shape: [Batch, Seq_Len] (1=valid, 0=padding)
        # We want to mask out invalid keys (columns)
        # Expand shape: [Batch, 1, 1, Seq_Len]
        padding_mask_4d = attention_mask.view(attention_mask.shape[0], 1, 1, seq_len).to(torch.bool)

        # Broadcasting: Geometry Mask [1, 1, L, L] & Padding Mask [B, 1, 1, L]
        # Result shape: [B, 1, L, L]
        valid_mask = valid_mask & padding_mask_4d

    # ------------------------------------------------------
    # 3. Convert to additive mask
    # ------------------------------------------------------
    # Get the minimal value for current dtype
    min_dtype = torch.finfo(dtype).min

    # Create result tensor filled with -inf by default
    mask_tensor = torch.full(valid_mask.shape, min_dtype, dtype=dtype, device=device)

    # Set valid positions to 0.0
    mask_tensor.masked_fill_(valid_mask, 0.0)

    return mask_tensor


def pack_sequences(hidden1: torch.Tensor, hidden2: torch.Tensor, mask1: torch.Tensor, mask2: torch.Tensor):
    """
    Pack two sequences by concatenating and sorting them based on mask values.

    Args:
        hidden1: First hidden states tensor of shape [B, L1, D]
        hidden2: Second hidden states tensor of shape [B, L2, D]
        mask1: First mask tensor of shape [B, L1]
        mask2: Second mask tensor of shape [B, L2]

    Returns:
        Tuple of (packed_hidden_states, new_mask) where:
        - packed_hidden_states: Packed hidden states with valid tokens (mask=1) first, shape [B, L1+L2, D]
        - new_mask: New mask tensor indicating valid positions, shape [B, L1+L2]
    """
    # Step 1: Concatenate hidden states and masks along sequence dimension
    hidden_cat = torch.cat([hidden1, hidden2], dim=1)  # [B, L, D]
    mask_cat = torch.cat([mask1, mask2], dim=1)  # [B, L]

    B, L, D = hidden_cat.shape

    # Step 2: Sort indices so that mask values of 1 come before 0
    sort_idx = mask_cat.argsort(dim=1, descending=True, stable=True)  # [B, L]

    # Step 3: Reorder hidden states using sorted indices
    hidden_left = torch.gather(hidden_cat, 1, sort_idx.unsqueeze(-1).expand(B, L, D))

    # Step 4: Create new mask based on valid sequence lengths
    lengths = mask_cat.sum(dim=1)  # [B]
    new_mask = (torch.arange(L, dtype=torch.long, device=hidden_cat.device).unsqueeze(0) < lengths.unsqueeze(1))

    return hidden_left, new_mask


class TimestepEmbedding(nn.Module):
    """
    Timestep embedding module for diffusion models.

    Converts timestep values into high-dimensional embeddings using sinusoidal
    positional encoding, followed by MLP layers. Used for conditioning diffusion
    models on timestep information.
    """
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        scale: float = 1,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act1 = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)
        self.in_channels = in_channels

        self.act2 = nn.SiLU()
        self.time_proj = nn.Linear(time_embed_dim, time_embed_dim * 6)
        self.scale = scale

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: A 1-D tensor of N indices, one per batch element. These may be fractional.
            dim: The dimension of the output embeddings.
            max_period: Controls the minimum frequency of the embeddings.

        Returns:
            An (N, D) tensor of positional embeddings.
        """
        t = t * self.scale
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.in_channels)
        temb = self.linear_1(t_freq.to(t.dtype))
        temb = self.act1(temb)
        temb = self.linear_2(temb)
        timestep_proj = self.time_proj(self.act2(temb)).unflatten(1, (6, -1))
        return temb, timestep_proj


class AceStepAttention(nn.Module):
    """
    Multi-headed attention module for AceStep model.

    Implements the attention mechanism from 'Attention Is All You Need' paper,
    with support for both self-attention and cross-attention modes. Uses RMSNorm
    for query and key normalization, and supports sliding window attention for
    efficient long-sequence processing.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rms_norm_eps: float,
        attention_bias: bool,
        attention_dropout: float,
        layer_types: list,
        head_dim: Optional[int] = None,
        sliding_window: Optional[int] = None,
        layer_idx: int = 0,
        is_cross_attention: bool = False,
        is_causal: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = attention_dropout
        if is_cross_attention:
            is_causal = False
        self.is_causal = is_causal
        self.is_cross_attention = is_cross_attention

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=attention_bias)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.attention_type = layer_types[layer_idx]
        self.sliding_window = sliding_window if layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Project and normalize query states
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

        # Determine if this is cross-attention (requires encoder_hidden_states)
        is_cross_attention = self.is_cross_attention and encoder_hidden_states is not None

        # Cross-attention path: attend to encoder hidden states
        if is_cross_attention:
            encoder_hidden_shape = (*encoder_hidden_states.shape[:-1], -1, self.head_dim)
            if past_key_value is not None:
                is_updated = past_key_value.is_updated.get(self.layer_idx)
                # After the first generated token, we can reuse all key/value states from cache
                curr_past_key_value = past_key_value.cross_attention_cache

                # Conditions for calculating key and value states
                if not is_updated:
                    # Compute and cache K/V for the first time
                    key_states = self.k_norm(self.k_proj(encoder_hidden_states).view(encoder_hidden_shape)).transpose(1, 2)
                    value_states = self.v_proj(encoder_hidden_states).view(encoder_hidden_shape).transpose(1, 2)
                    # Update cache: save all key/value states to cache for fast auto-regressive generation
                    key_states, value_states = curr_past_key_value.update(key_states, value_states, self.layer_idx)
                    # Set flag that this layer's cross-attention cache is updated
                    past_key_value.is_updated[self.layer_idx] = True
                else:
                    # Reuse cached key/value states for subsequent tokens
                    key_states = curr_past_key_value.layers[self.layer_idx].keys
                    value_states = curr_past_key_value.layers[self.layer_idx].values
            else:
                # No cache used, compute K/V directly
                key_states = self.k_norm(self.k_proj(encoder_hidden_states).view(encoder_hidden_shape)).transpose(1, 2)
                value_states = self.v_proj(encoder_hidden_states).view(encoder_hidden_shape).transpose(1, 2)

        # Self-attention path: attend to the same sequence
        else:
            # Project and normalize key/value states for self-attention
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            # Apply rotary position embeddings (RoPE) if provided
            if position_embeddings is not None:
                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # Update cache for auto-regressive generation
            if past_key_value is not None:
                # Sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # GGA expansion: if num_key_value_heads < num_attention_heads
        if self.num_key_value_groups > 1:
            key_states = key_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).flatten(1, 2)
            value_states = value_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).flatten(1, 2)

        # Use DiffSynth unified attention
        # Tensors are already in (batch, heads, seq, dim) format -> "b n s d"
        attn_output = attention_forward(
            query_states, key_states, value_states,
            q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d",
            attn_mask=attention_mask,
        )

        attn_weights = None  # attention_forward doesn't return weights

        # Flatten and project output: (B, n_heads, seq, dim) -> (B, seq, n_heads*dim)
        attn_output = attn_output.transpose(1, 2).flatten(2, 3).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class AceStepEncoderLayer(nn.Module):
    """
    Encoder layer for AceStep model.

    Consists of self-attention and MLP (feed-forward) sub-layers with residual connections.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int = 6144,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        layer_types: list = None,
        head_dim: Optional[int] = None,
        sliding_window: Optional[int] = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx

        self.self_attn = AceStepAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            layer_types=layer_types,
            head_dim=head_dim,
            sliding_window=sliding_window,
            layer_idx=layer_idx,
            is_cross_attention=False,
            is_causal=False,
        )
        self.input_layernorm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)

        # MLP (feed-forward) sub-layer
        self.mlp = Qwen3MLP(
            config=type('Config', (), {
                'hidden_size': hidden_size,
                'intermediate_size': intermediate_size,
                'hidden_act': 'silu',
            })()
        )
        self.attention_type = layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor,
        Optional[tuple[torch.FloatTensor, torch.FloatTensor]],
    ]:
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            # Encoders don't use cache
            use_cache=False,
            past_key_value=None,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class AceStepDiTLayer(nn.Module):
    """
    DiT (Diffusion Transformer) layer for AceStep model.

    Implements a transformer layer with three main components:
    1. Self-attention with adaptive layer norm (AdaLN)
    2. Cross-attention (optional) for conditioning on encoder outputs
    3. Feed-forward MLP with adaptive layer norm

    Uses scale-shift modulation from timestep embeddings for adaptive normalization.
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        rms_norm_eps: float,
        attention_bias: bool,
        attention_dropout: float,
        layer_types: list,
        head_dim: Optional[int] = None,
        sliding_window: Optional[int] = None,
        layer_idx: int = 0,
        use_cross_attention: bool = True,
    ):
        super().__init__()

        self.self_attn_norm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = AceStepAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            layer_types=layer_types,
            head_dim=head_dim,
            sliding_window=sliding_window,
            layer_idx=layer_idx,
        )

        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.cross_attn_norm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
            self.cross_attn = AceStepAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                rms_norm_eps=rms_norm_eps,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
                layer_types=layer_types,
                head_dim=head_dim,
                sliding_window=sliding_window,
                layer_idx=layer_idx,
                is_cross_attention=True,
            )

        self.mlp_norm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = Qwen3MLP(
            config=type('Config', (), {
                'hidden_size': hidden_size,
                'intermediate_size': intermediate_size,
                'hidden_act': 'silu',
            })()
        )

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, hidden_size) / hidden_size**0.5)
        self.attention_type = layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[EncoderDecoderCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        # Extract scale-shift parameters for adaptive layer norm from timestep embeddings
        # 6 values: (shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table.to(temb.device) + temb
        ).chunk(6, dim=1)

        # Step 1: Self-attention with adaptive layer norm (AdaLN)
        # Apply adaptive normalization: norm(x) * (1 + scale) + shift
        norm_hidden_states = (self.self_attn_norm(hidden_states) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output, self_attn_weights = self.self_attn(
            hidden_states=norm_hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            use_cache=False,
            past_key_value=None,
            **kwargs,
        )
        # Apply gated residual connection: x = x + attn_output * gate
        hidden_states = (hidden_states + attn_output * gate_msa).type_as(hidden_states)

        # Step 2: Cross-attention (if enabled) for conditioning on encoder outputs
        if self.use_cross_attention:
            norm_hidden_states = self.cross_attn_norm(hidden_states).type_as(hidden_states)
            attn_output, cross_attn_weights = self.cross_attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
            # Standard residual connection for cross-attention
            hidden_states = hidden_states + attn_output

        # Step 3: Feed-forward (MLP) with adaptive layer norm
        # Apply adaptive normalization for MLP: norm(x) * (1 + scale) + shift
        norm_hidden_states = (self.mlp_norm(hidden_states) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
        ff_output = self.mlp(norm_hidden_states)
        # Apply gated residual connection: x = x + mlp_output * gate
        hidden_states = (hidden_states + ff_output * c_gate_msa).type_as(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs



class Lambda(nn.Module):
    """
    Wrapper module for arbitrary lambda functions.

    Allows using lambda functions in nn.Sequential by wrapping them in a Module.
    Useful for simple transformations like transpose operations.
    """
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class AceStepDiTModel(nn.Module):
    """
    DiT (Diffusion Transformer) model for AceStep.

    Main diffusion model that generates audio latents conditioned on text, lyrics,
    and timbre. Uses patch-based processing with transformer layers, timestep
    conditioning, and cross-attention to encoder outputs.
    """
    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        layer_types: Optional[list] = None,
        head_dim: Optional[int] = None,
        sliding_window: Optional[int] = 128,
        use_sliding_window: bool = True,
        use_cache: bool = True,
        rope_theta: float = 1000000,
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        patch_size: int = 2,
        in_channels: int = 192,
        audio_acoustic_hidden_dim: int = 64,
        encoder_hidden_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        self.layer_types = layer_types or (["sliding_attention", "full_attention"] * (num_hidden_layers // 2))
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.use_cache = use_cache
        encoder_hidden_size = encoder_hidden_size or hidden_size

        # Rotary position embeddings for transformer layers
        rope_config = type('RopeConfig', (), {
            'hidden_size': hidden_size,
            'num_attention_heads': num_attention_heads,
            'num_key_value_heads': num_key_value_heads,
            'head_dim': head_dim,
            'max_position_embeddings': max_position_embeddings,
            'rope_theta': rope_theta,
            'rope_parameters': {'rope_type': 'default', 'rope_theta': rope_theta},
            'rms_norm_eps': rms_norm_eps,
            'attention_bias': attention_bias,
            'attention_dropout': attention_dropout,
            'hidden_act': 'silu',
            'intermediate_size': intermediate_size,
            'layer_types': self.layer_types,
            'sliding_window': sliding_window,
        })()
        self.rotary_emb = Qwen3RotaryEmbedding(rope_config)

        # Stack of DiT transformer layers
        self.layers = nn.ModuleList([
            AceStepDiTLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                intermediate_size=intermediate_size,
                rms_norm_eps=rms_norm_eps,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
                layer_types=self.layer_types,
                head_dim=head_dim,
                sliding_window=sliding_window,
                layer_idx=layer_idx,
            )
            for layer_idx in range(num_hidden_layers)
        ])

        self.patch_size = patch_size

        # Input projection: patch embedding using 1D convolution
        self.proj_in = nn.Sequential(
            Lambda(lambda x: x.transpose(1, 2)),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=hidden_size,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
            ),
            Lambda(lambda x: x.transpose(1, 2)),
        )

        # Timestep embeddings for diffusion conditioning
        self.time_embed = TimestepEmbedding(in_channels=256, time_embed_dim=hidden_size)
        self.time_embed_r = TimestepEmbedding(in_channels=256, time_embed_dim=hidden_size)

        # Project encoder hidden states to model dimension
        self.condition_embedder = nn.Linear(encoder_hidden_size, hidden_size, bias=True)

        # Output normalization and projection
        self.norm_out = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.proj_out = nn.Sequential(
            Lambda(lambda x: x.transpose(1, 2)),
            nn.ConvTranspose1d(
                in_channels=hidden_size,
                out_channels=audio_acoustic_hidden_dim,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
            ),
            Lambda(lambda x: x.transpose(1, 2)),
        )
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, hidden_size) / hidden_size**0.5)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_r: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        context_latents: torch.Tensor,
        use_cache: Optional[bool] = False,
        past_key_values: Optional[EncoderDecoderCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        return_hidden_states: int = None,
        custom_layers_config: Optional[dict] = None,
        enable_early_exit: bool = False,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):

        use_cache = use_cache if use_cache is not None else self.use_cache

        # Disable cache during training or when gradient checkpointing is enabled
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
        if self.training:
            use_cache = False

        # Initialize cache if needed (only during inference for auto-regressive generation)
        if not self.training and use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())

        # Compute timestep embeddings for diffusion conditioning
        # Two embeddings: one for timestep t, one for timestep difference (t - r)
        temb_t, timestep_proj_t = self.time_embed(timestep)
        temb_r, timestep_proj_r = self.time_embed_r(timestep - timestep_r)
        # Combine embeddings
        temb = temb_t + temb_r
        timestep_proj = timestep_proj_t + timestep_proj_r

        # Concatenate context latents (source latents + chunk masks) with hidden states
        hidden_states = torch.cat([context_latents, hidden_states], dim=-1)
        # Record original sequence length for later restoration after padding
        original_seq_len = hidden_states.shape[1]
        # Apply padding if sequence length is not divisible by patch_size
        # This ensures proper patch extraction
        pad_length = 0
        if hidden_states.shape[1] % self.patch_size != 0:
            pad_length = self.patch_size - (hidden_states.shape[1] % self.patch_size)
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_length), mode='constant', value=0)

        # Project input to patches and project encoder states
        hidden_states = self.proj_in(hidden_states)
        encoder_hidden_states = self.condition_embedder(encoder_hidden_states)

        # Cache positions
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )

        # Position IDs
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        seq_len = hidden_states.shape[1]
        encoder_seq_len = encoder_hidden_states.shape[1]
        dtype = hidden_states.dtype
        device = hidden_states.device

        # Initialize Mask variables
        full_attn_mask = None
        sliding_attn_mask = None
        encoder_attn_mask = None
        decoder_attn_mask = None
        # Target library discards the passed-in attention_mask for 4D mask
        # construction (line 1384: attention_mask = None)
        attention_mask = None

        # 1. Full Attention (Bidirectional, Global)
        full_attn_mask = create_4d_mask(
            seq_len=seq_len,
            dtype=dtype,
            device=device,
            attention_mask=attention_mask,
            sliding_window=None,
            is_sliding_window=False,
            is_causal=False
        )
        max_len = max(seq_len, encoder_seq_len)

        encoder_attn_mask = create_4d_mask(
            seq_len=max_len,
            dtype=dtype,
            device=device,
            attention_mask=attention_mask,
            sliding_window=None,
            is_sliding_window=False,
            is_causal=False
        )
        encoder_attn_mask = encoder_attn_mask[:, :, :seq_len, :encoder_seq_len]

        # 2. Sliding Attention (Bidirectional, Local)
        if self.use_sliding_window:
            sliding_attn_mask = create_4d_mask(
                seq_len=seq_len,
                dtype=dtype,
                device=device,
                attention_mask=attention_mask,
                sliding_window=self.sliding_window,
                is_sliding_window=True,
                is_causal=False
            )

        # Build mask mapping
        self_attn_mask_mapping = {
            "full_attention": full_attn_mask,
            "sliding_attention": sliding_attn_mask,
            "encoder_attention_mask": encoder_attn_mask,
        }

        # Create position embeddings to be shared across all decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_cross_attentions = () if output_attentions else None

        # Handle early exit for custom layer configurations
        max_needed_layer = float('inf')
        if custom_layers_config is not None and enable_early_exit:
            max_needed_layer = max(custom_layers_config.keys())
            output_attentions = True
            if all_cross_attentions is None:
                all_cross_attentions = ()

        # Process through transformer layers
        for index_block, layer_module in enumerate(self.layers):
            # Early exit optimization
            if index_block > max_needed_layer:
                break

            # Prepare layer arguments
            layer_args = (
                hidden_states,
                position_embeddings,
                timestep_proj,
                self_attn_mask_mapping[layer_module.attention_type],
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                encoder_hidden_states,
                self_attn_mask_mapping["encoder_attention_mask"],
            )
            layer_kwargs = flash_attn_kwargs

            # Use gradient checkpointing if enabled
            layer_outputs = gradient_checkpoint_forward(
                layer_module,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                *layer_args,
                **layer_kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions and self.layers[index_block].use_cross_attention:
                # layer_outputs structure: (hidden_states, self_attn_weights, cross_attn_weights)
                if len(layer_outputs) >= 3:
                    all_cross_attentions += (layer_outputs[2],)

        if return_hidden_states:
            return hidden_states

        # Extract scale-shift parameters for adaptive output normalization
        shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        # Apply adaptive layer norm: norm(x) * (1 + scale) + shift
        hidden_states = (self.norm_out(hidden_states) * (1 + scale) + shift).type_as(hidden_states)
        # Project output: de-patchify back to original sequence format
        hidden_states = self.proj_out(hidden_states)

        # Crop back to original sequence length to ensure exact length match (remove padding)
        hidden_states = hidden_states[:, :original_seq_len, :]

        outputs = (hidden_states, past_key_values)

        if output_attentions:
            outputs += (all_cross_attentions,)
        return outputs
