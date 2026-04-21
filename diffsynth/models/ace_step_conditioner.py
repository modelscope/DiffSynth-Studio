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


def pack_sequences(hidden1: torch.Tensor, hidden2: torch.Tensor, mask1: torch.Tensor, mask2: torch.Tensor):
    hidden_cat = torch.cat([hidden1, hidden2], dim=1)
    mask_cat = torch.cat([mask1, mask2], dim=1)
    B, L, D = hidden_cat.shape
    sort_idx = mask_cat.argsort(dim=1, descending=True, stable=True)
    hidden_left = torch.gather(hidden_cat, 1, sort_idx.unsqueeze(-1).expand(B, L, D))
    lengths = mask_cat.sum(dim=1)
    new_mask = (torch.arange(L, dtype=torch.long, device=hidden_cat.device).unsqueeze(0) < lengths.unsqueeze(1))
    return hidden_left, new_mask


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
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
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


class AceStepLyricEncoder(nn.Module):
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
        text_hidden_dim: int = 1024,
        num_lyric_encoder_hidden_layers: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.num_lyric_encoder_hidden_layers = num_lyric_encoder_hidden_layers
        self.text_hidden_dim = text_hidden_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.layer_types = layer_types or (["sliding_attention", "full_attention"] * (num_hidden_layers // 2))
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.sliding_window = sliding_window
        self.use_sliding_window = use_sliding_window
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self._attn_implementation = kwargs.get("_attn_implementation", "sdpa")

        self.embed_tokens = nn.Linear(text_hidden_dim, hidden_size)
        self.norm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
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
            '_attn_implementation': self._attn_implementation,
        })()
        self.rotary_emb = Qwen3RotaryEmbedding(rope_config)
        self.gradient_checkpointing = False

        self.layers = nn.ModuleList([
            AceStepEncoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                rms_norm_eps=rms_norm_eps,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
                layer_types=self.layer_types,
                head_dim=head_dim,
                sliding_window=sliding_window,
                layer_idx=layer_idx,
            )
            for layer_idx in range(num_lyric_encoder_hidden_layers)
        ])


    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutput:
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        assert input_ids is None, "Only `inputs_embeds` is supported for the lyric encoder."
        assert attention_mask is not None, "Attention mask must be provided for the lyric encoder."
        assert inputs_embeds is not None, "Inputs embeddings must be provided for the lyric encoder."

        inputs_embeds = self.embed_tokens(inputs_embeds)
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        seq_len = inputs_embeds.shape[1]
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device

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

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer_module in self.layers[: self.num_lyric_encoder_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, position_embeddings,
                self_attn_mask_mapping[layer_module.attention_type],
                position_ids, output_attentions,
                **flash_attn_kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class AceStepTimbreEncoder(nn.Module):
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
        timbre_hidden_dim: int = 64,
        num_timbre_encoder_hidden_layers: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.layer_types = layer_types or (["sliding_attention", "full_attention"] * (num_hidden_layers // 2))
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.sliding_window = sliding_window
        self.use_sliding_window = use_sliding_window
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.timbre_hidden_dim = timbre_hidden_dim
        self.num_timbre_encoder_hidden_layers = num_timbre_encoder_hidden_layers
        self._attn_implementation = kwargs.get("_attn_implementation", "sdpa")

        self.embed_tokens = nn.Linear(timbre_hidden_dim, hidden_size)
        self.norm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
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
            '_attn_implementation': self._attn_implementation,
        })()
        self.rotary_emb = Qwen3RotaryEmbedding(rope_config)
        self.gradient_checkpointing = False
        self.special_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.layers = nn.ModuleList([
            AceStepEncoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                rms_norm_eps=rms_norm_eps,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
                layer_types=self.layer_types,
                head_dim=head_dim,
                sliding_window=sliding_window,
                layer_idx=layer_idx,
            )
            for layer_idx in range(num_timbre_encoder_hidden_layers)
        ])


    def unpack_timbre_embeddings(self, timbre_embs_packed, refer_audio_order_mask):
        N, d = timbre_embs_packed.shape
        device = timbre_embs_packed.device
        dtype = timbre_embs_packed.dtype
        B = int(refer_audio_order_mask.max().item() + 1)
        counts = torch.bincount(refer_audio_order_mask, minlength=B)
        max_count = counts.max().item()
        sorted_indices = torch.argsort(refer_audio_order_mask * N + torch.arange(N, device=device), stable=True)
        sorted_batch_ids = refer_audio_order_mask[sorted_indices]
        positions = torch.arange(N, device=device)
        batch_starts = torch.cat([torch.tensor([0], device=device), torch.cumsum(counts, dim=0)[:-1]])
        positions_in_sorted = positions - batch_starts[sorted_batch_ids]
        inverse_indices = torch.empty_like(sorted_indices)
        inverse_indices[sorted_indices] = torch.arange(N, device=device)
        positions_in_batch = positions_in_sorted[inverse_indices]
        indices_2d = refer_audio_order_mask * max_count + positions_in_batch
        one_hot = F.one_hot(indices_2d, num_classes=B * max_count).to(dtype)
        timbre_embs_flat = one_hot.t() @ timbre_embs_packed
        timbre_embs_unpack = timbre_embs_flat.reshape(B, max_count, d)
        mask_flat = (one_hot.sum(dim=0) > 0).long()
        new_mask = mask_flat.reshape(B, max_count)
        return timbre_embs_unpack, new_mask

    @can_return_tuple
    def forward(
        self,
        refer_audio_acoustic_hidden_states_packed: Optional[torch.FloatTensor] = None,
        refer_audio_order_mask: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutput:
        inputs_embeds = refer_audio_acoustic_hidden_states_packed
        inputs_embeds = self.embed_tokens(inputs_embeds)
        seq_len = inputs_embeds.shape[1]
        cache_position = torch.arange(0, seq_len, device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)

        dtype = inputs_embeds.dtype
        device = inputs_embeds.device

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

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_module in self.layers[: self.num_timbre_encoder_hidden_layers]:
            layer_outputs = layer_module(
                hidden_states, position_embeddings,
                self_attn_mask_mapping[layer_module.attention_type],
                position_ids,
                **flash_attn_kwargs,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states[:, 0, :]
        # For packed input: reshape [1, T, D] -> [T, D] for unpacking
        timbre_embs_unpack, timbre_embs_mask = self.unpack_timbre_embeddings(hidden_states, refer_audio_order_mask)
        return timbre_embs_unpack, timbre_embs_mask


class AceStepConditionEncoder(nn.Module):
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
        text_hidden_dim: int = 1024,
        timbre_hidden_dim: int = 64,
        num_lyric_encoder_hidden_layers: int = 8,
        num_timbre_encoder_hidden_layers: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.layer_types = layer_types or (["sliding_attention", "full_attention"] * (num_hidden_layers // 2))
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.sliding_window = sliding_window
        self.use_sliding_window = use_sliding_window
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.text_hidden_dim = text_hidden_dim
        self.timbre_hidden_dim = timbre_hidden_dim
        self.num_lyric_encoder_hidden_layers = num_lyric_encoder_hidden_layers
        self.num_timbre_encoder_hidden_layers = num_timbre_encoder_hidden_layers
        self._attn_implementation = kwargs.get("_attn_implementation", "sdpa")

        self.text_projector = nn.Linear(text_hidden_dim, hidden_size, bias=False)
        self.null_condition_emb = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.lyric_encoder = AceStepLyricEncoder(
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
            text_hidden_dim=text_hidden_dim,
            num_lyric_encoder_hidden_layers=num_lyric_encoder_hidden_layers,
        )
        self.timbre_encoder = AceStepTimbreEncoder(
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
            timbre_hidden_dim=timbre_hidden_dim,
            num_timbre_encoder_hidden_layers=num_timbre_encoder_hidden_layers,
        )

    def forward(
        self,
        text_hidden_states: Optional[torch.FloatTensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        lyric_hidden_states: Optional[torch.LongTensor] = None,
        lyric_attention_mask: Optional[torch.Tensor] = None,
        reference_latents: Optional[torch.Tensor] = None,
        refer_audio_order_mask: Optional[torch.LongTensor] = None,
    ):
        text_hidden_states = self.text_projector(text_hidden_states)
        lyric_encoder_outputs = self.lyric_encoder(
            inputs_embeds=lyric_hidden_states,
            attention_mask=lyric_attention_mask,
        )
        lyric_hidden_states = lyric_encoder_outputs.last_hidden_state
        timbre_embs_unpack, timbre_embs_mask = self.timbre_encoder(reference_latents, refer_audio_order_mask)
        encoder_hidden_states, encoder_attention_mask = pack_sequences(
            lyric_hidden_states, timbre_embs_unpack, lyric_attention_mask, timbre_embs_mask
        )
        encoder_hidden_states, encoder_attention_mask = pack_sequences(
            encoder_hidden_states, text_hidden_states, encoder_attention_mask, text_attention_mask
        )
        return encoder_hidden_states, encoder_attention_mask
