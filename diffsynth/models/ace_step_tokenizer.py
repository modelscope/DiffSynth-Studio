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
"""ACE-Step Audio Tokenizer — VAE latent discretization pathway.

Contains:
- AceStepAudioTokenizer: continuous VAE latent → discrete FSQ tokens
- AudioTokenDetokenizer: discrete tokens → continuous VAE-latent-shaped features

Only used in cover song mode (is_covers=True). Bypassed in text-to-music.
"""
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from ..core.attention import attention_forward
from ..core.gradient import gradient_checkpoint_forward

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutput
from transformers.processing_utils import Unpack
from transformers.utils import can_return_tuple, logging
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)
from vector_quantize_pytorch import ResidualFSQ

logger = logging.get_logger(__name__)


def create_4d_mask(
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    is_sliding_window: bool = False,
    is_causal: bool = True,
) -> torch.Tensor:
    indices = torch.arange(seq_len, device=device)
    diff = indices.unsqueeze(1) - indices.unsqueeze(0)
    valid_mask = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool)
    if is_causal:
        valid_mask = valid_mask & (diff >= 0)
    if is_sliding_window and sliding_window is not None:
        if is_causal:
            valid_mask = valid_mask & (diff <= sliding_window)
        else:
            valid_mask = valid_mask & (torch.abs(diff) <= sliding_window)
    valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)
    if attention_mask is not None:
        padding_mask_4d = attention_mask.view(attention_mask.shape[0], 1, 1, seq_len).to(torch.bool)
        valid_mask = valid_mask & padding_mask_4d
    min_dtype = torch.finfo(dtype).min
    mask_tensor = torch.full(valid_mask.shape, min_dtype, dtype=dtype, device=device)
    mask_tensor.masked_fill_(valid_mask, 0.0)
    return mask_tensor


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class AceStepAttention(nn.Module):
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
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

        is_cross_attention = self.is_cross_attention and encoder_hidden_states is not None

        if is_cross_attention:
            encoder_hidden_shape = (*encoder_hidden_states.shape[:-1], -1, self.head_dim)
            if past_key_value is not None:
                is_updated = past_key_value.is_updated.get(self.layer_idx)
                curr_past_key_value = past_key_value.cross_attention_cache
                if not is_updated:
                    key_states = self.k_norm(self.k_proj(encoder_hidden_states).view(encoder_hidden_shape)).transpose(1, 2)
                    value_states = self.v_proj(encoder_hidden_states).view(encoder_hidden_shape).transpose(1, 2)
                    key_states, value_states = curr_past_key_value.update(key_states, value_states, self.layer_idx)
                    past_key_value.is_updated[self.layer_idx] = True
                else:
                    key_states = curr_past_key_value.layers[self.layer_idx].keys
                    value_states = curr_past_key_value.layers[self.layer_idx].values
            else:
                key_states = self.k_norm(self.k_proj(encoder_hidden_states).view(encoder_hidden_shape)).transpose(1, 2)
                value_states = self.v_proj(encoder_hidden_states).view(encoder_hidden_shape).transpose(1, 2)

        else:
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            if position_embeddings is not None:
                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.num_key_value_groups > 1:
            key_states = key_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).flatten(1, 2)
            value_states = value_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).flatten(1, 2)

        attn_output = attention_forward(
            query_states, key_states, value_states,
            q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d",
            attn_mask=attention_mask,
        )
        attn_weights = None

        attn_output = attn_output.transpose(1, 2).flatten(2, 3).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class AceStepEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rms_norm_eps: float,
        attention_bias: bool,
        attention_dropout: float,
        layer_types: list,
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

        mlp_config = type('Config', (), {
            'hidden_size': hidden_size,
            'intermediate_size': intermediate_size,
            'hidden_act': 'silu',
        })()
        self.mlp = Qwen3MLP(mlp_config)
        self.attention_type = layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            use_cache=False,
            past_key_value=None,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


class AttentionPooler(nn.Module):
    """Pools every pool_window_size frames into 1 representation via transformer + CLS token."""

    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        layer_types: Optional[list] = None,
        head_dim: Optional[int] = None,
        sliding_window: Optional[int] = 128,
        use_sliding_window: bool = True,
        rope_theta: float = 1000000,
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        num_attention_pooler_hidden_layers: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Default matches target library config (24 alternating entries).
        self.layer_types = layer_types or (["sliding_attention", "full_attention"] * 12)
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.sliding_window = sliding_window
        self.use_sliding_window = use_sliding_window
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.num_attention_pooler_hidden_layers = num_attention_pooler_hidden_layers
        self._attn_implementation = kwargs.get("_attn_implementation", "sdpa")

        self.embed_tokens = nn.Linear(hidden_size, hidden_size)
        self.norm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        # Slice layer_types to our own layer count
        pooler_layer_types = self.layer_types[:num_attention_pooler_hidden_layers]
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
            'layer_types': pooler_layer_types,
            'sliding_window': sliding_window,
            '_attn_implementation': self._attn_implementation,
        })()
        self.rotary_emb = Qwen3RotaryEmbedding(rope_config)
        self.gradient_checkpointing = False
        self.special_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.layers = nn.ModuleList([
            AceStepEncoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                rms_norm_eps=rms_norm_eps,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
                layer_types=pooler_layer_types,
                head_dim=head_dim,
                sliding_window=sliding_window,
                layer_idx=layer_idx,
            )
            for layer_idx in range(num_attention_pooler_hidden_layers)
        ])

    @can_return_tuple
    def forward(
        self,
        x,
        attention_mask: Optional[torch.Tensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        B, T, P, D = x.shape
        x = self.embed_tokens(x)
        special_tokens = self.special_token.expand(B, T, 1, -1).to(x.device)
        x = torch.cat([special_tokens, x], dim=2)
        x = rearrange(x, "b t p c -> (b t) p c")

        cache_position = torch.arange(0, x.shape[1], device=x.device)
        position_ids = cache_position.unsqueeze(0)
        hidden_states = x
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        seq_len = x.shape[1]
        dtype = x.dtype
        device = x.device

        full_attn_mask = create_4d_mask(
            seq_len=seq_len, dtype=dtype, device=device,
            attention_mask=attention_mask, sliding_window=None,
            is_sliding_window=False, is_causal=False
        )
        sliding_attn_mask = None
        if self.use_sliding_window:
            sliding_attn_mask = create_4d_mask(
                seq_len=seq_len, dtype=dtype, device=device,
                attention_mask=attention_mask, sliding_window=self.sliding_window,
                is_sliding_window=True, is_causal=False
            )

        self_attn_mask_mapping = {
            "full_attention": full_attn_mask,
            "sliding_attention": sliding_attn_mask,
        }

        for layer_module in self.layers:
            layer_outputs = layer_module(
                hidden_states, position_embeddings,
                attention_mask=self_attn_mask_mapping[layer_module.attention_type],
                **flash_attn_kwargs,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        cls_output = hidden_states[:, 0, :]
        return rearrange(cls_output, "(b t) c -> b t c", b=B)


class AceStepAudioTokenizer(nn.Module):
    """Converts continuous acoustic features (VAE latents) into discrete quantized tokens.

    Input: [B, T, 64] (VAE latent dim)
    Output: quantized [B, T/5, 2048], indices [B, T/5, 1]
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        layer_types: Optional[list] = None,
        head_dim: Optional[int] = None,
        sliding_window: Optional[int] = 128,
        use_sliding_window: bool = True,
        rope_theta: float = 1000000,
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        audio_acoustic_hidden_dim: int = 64,
        pool_window_size: int = 5,
        fsq_dim: int = 2048,
        fsq_input_levels: list = None,
        fsq_input_num_quantizers: int = 1,
        num_attention_pooler_hidden_layers: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Default matches target library config (24 alternating entries).
        self.layer_types = layer_types or (["sliding_attention", "full_attention"] * 12)
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.sliding_window = sliding_window
        self.use_sliding_window = use_sliding_window
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.audio_acoustic_hidden_dim = audio_acoustic_hidden_dim
        self.pool_window_size = pool_window_size
        self.fsq_dim = fsq_dim
        self.fsq_input_levels = fsq_input_levels or [8, 8, 8, 5, 5, 5]
        self.fsq_input_num_quantizers = fsq_input_num_quantizers
        self.num_attention_pooler_hidden_layers = num_attention_pooler_hidden_layers
        self._attn_implementation = kwargs.get("_attn_implementation", "sdpa")

        self.audio_acoustic_proj = nn.Linear(audio_acoustic_hidden_dim, hidden_size)
        # Slice layer_types for the attention pooler
        pooler_layer_types = self.layer_types[:num_attention_pooler_hidden_layers]
        self.attention_pooler = AttentionPooler(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            layer_types=pooler_layer_types,
            head_dim=head_dim,
            sliding_window=sliding_window,
            use_sliding_window=use_sliding_window,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            num_attention_pooler_hidden_layers=num_attention_pooler_hidden_layers,
        )
        self.quantizer = ResidualFSQ(
            dim=self.fsq_dim,
            levels=self.fsq_input_levels,
            num_quantizers=self.fsq_input_num_quantizers,
            force_quantization_f32=False,  # avoid autocast bug in vector_quantize_pytorch
        )

    @can_return_tuple
    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.audio_acoustic_proj(hidden_states)
        hidden_states = self.attention_pooler(hidden_states)
        quantized, indices = self.quantizer(hidden_states)
        return quantized, indices

    def tokenize(self, x):
        """Convenience: takes [B, T, 64], rearranges to patches, runs forward."""
        x = rearrange(x, 'n (t_patch p) d -> n t_patch p d', p=self.pool_window_size)
        return self.forward(x)


class AudioTokenDetokenizer(nn.Module):
    """Converts quantized audio tokens back to continuous acoustic representations.

    Input: [B, T/5, hidden_size] (quantized vectors)
    Output: [B, T, 64] (VAE-latent-shaped continuous features)
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        layer_types: Optional[list] = None,
        head_dim: Optional[int] = None,
        sliding_window: Optional[int] = 128,
        use_sliding_window: bool = True,
        rope_theta: float = 1000000,
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        pool_window_size: int = 5,
        audio_acoustic_hidden_dim: int = 64,
        num_attention_pooler_hidden_layers: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Default matches target library config (24 alternating entries).
        self.layer_types = layer_types or (["sliding_attention", "full_attention"] * 12)
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.sliding_window = sliding_window
        self.use_sliding_window = use_sliding_window
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.pool_window_size = pool_window_size
        self.audio_acoustic_hidden_dim = audio_acoustic_hidden_dim
        self.num_attention_pooler_hidden_layers = num_attention_pooler_hidden_layers
        self._attn_implementation = kwargs.get("_attn_implementation", "sdpa")

        self.embed_tokens = nn.Linear(hidden_size, hidden_size)
        self.norm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        # Slice layer_types to our own layer count (use num_audio_decoder_hidden_layers)
        detok_layer_types = self.layer_types[:num_attention_pooler_hidden_layers]
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
            'layer_types': detok_layer_types,
            'sliding_window': sliding_window,
            '_attn_implementation': self._attn_implementation,
        })()
        self.rotary_emb = Qwen3RotaryEmbedding(rope_config)
        self.gradient_checkpointing = False
        self.special_tokens = nn.Parameter(torch.randn(1, pool_window_size, hidden_size) * 0.02)
        self.layers = nn.ModuleList([
            AceStepEncoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                rms_norm_eps=rms_norm_eps,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
                layer_types=detok_layer_types,
                head_dim=head_dim,
                sliding_window=sliding_window,
                layer_idx=layer_idx,
            )
            for layer_idx in range(num_attention_pooler_hidden_layers)
        ])
        self.proj_out = nn.Linear(hidden_size, audio_acoustic_hidden_dim)

    @can_return_tuple
    def forward(
        self,
        x,
        attention_mask: Optional[torch.Tensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        B, T, D = x.shape
        x = self.embed_tokens(x)
        x = x.unsqueeze(2).repeat(1, 1, self.pool_window_size, 1)
        special_tokens = self.special_tokens.expand(B, T, -1, -1)
        x = x + special_tokens.to(x.device)
        x = rearrange(x, "b t p c -> (b t) p c")

        cache_position = torch.arange(0, x.shape[1], device=x.device)
        position_ids = cache_position.unsqueeze(0)
        hidden_states = x
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        seq_len = x.shape[1]
        dtype = x.dtype
        device = x.device

        full_attn_mask = create_4d_mask(
            seq_len=seq_len, dtype=dtype, device=device,
            attention_mask=attention_mask, sliding_window=None,
            is_sliding_window=False, is_causal=False
        )
        sliding_attn_mask = None
        if self.use_sliding_window:
            sliding_attn_mask = create_4d_mask(
                seq_len=seq_len, dtype=dtype, device=device,
                attention_mask=attention_mask, sliding_window=self.sliding_window,
                is_sliding_window=True, is_causal=False
            )

        self_attn_mask_mapping = {
            "full_attention": full_attn_mask,
            "sliding_attention": sliding_attn_mask,
        }

        for layer_module in self.layers:
            layer_outputs = layer_module(
                hidden_states, position_embeddings,
                attention_mask=self_attn_mask_mapping[layer_module.attention_type],
                **flash_attn_kwargs,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        return rearrange(hidden_states, "(b t) p c -> b (t p) c", b=B, p=self.pool_window_size)


class AceStepTokenizer(nn.Module):
    """Container for AceStepAudioTokenizer + AudioTokenDetokenizer.

    Provides encode/decode convenience methods for VAE latent discretization.
    Used in cover song mode to convert source audio latents to discrete tokens
    and back to continuous conditioning hints.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        layer_types: Optional[list] = None,
        head_dim: Optional[int] = None,
        sliding_window: Optional[int] = 128,
        use_sliding_window: bool = True,
        rope_theta: float = 1000000,
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        audio_acoustic_hidden_dim: int = 64,
        pool_window_size: int = 5,
        fsq_dim: int = 2048,
        fsq_input_levels: list = None,
        fsq_input_num_quantizers: int = 1,
        num_attention_pooler_hidden_layers: int = 2,
        num_audio_decoder_hidden_layers: int = 24,
        **kwargs,
    ):
        super().__init__()
        # Default layer_types matches target library config (24 alternating entries).
        # Sub-modules (pooler/detokenizer) slice first N entries for their own layer count.
        if layer_types is None:
            layer_types = ["sliding_attention", "full_attention"] * 12
        self.tokenizer = AceStepAudioTokenizer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            layer_types=layer_types,
            head_dim=head_dim,
            sliding_window=sliding_window,
            use_sliding_window=use_sliding_window,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            audio_acoustic_hidden_dim=audio_acoustic_hidden_dim,
            pool_window_size=pool_window_size,
            fsq_dim=fsq_dim,
            fsq_input_levels=fsq_input_levels,
            fsq_input_num_quantizers=fsq_input_num_quantizers,
            num_attention_pooler_hidden_layers=num_attention_pooler_hidden_layers,
            **kwargs,
        )
        self.detokenizer = AudioTokenDetokenizer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            layer_types=layer_types,
            head_dim=head_dim,
            sliding_window=sliding_window,
            use_sliding_window=use_sliding_window,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            pool_window_size=pool_window_size,
            audio_acoustic_hidden_dim=audio_acoustic_hidden_dim,
            num_attention_pooler_hidden_layers=num_attention_pooler_hidden_layers,
            **kwargs,
        )

    def encode(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """VAE latent [B, T, 64] → discrete tokens."""
        return self.tokenizer(hidden_states)

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """Discrete tokens [B, T/5, hidden_size] → continuous [B, T, 64]."""
        return self.detokenizer(quantized)

    def tokenize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience: [B, T, 64] → quantized + indices via patch rearrangement."""
        return self.tokenizer.tokenize(x)
