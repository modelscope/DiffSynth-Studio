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


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, time_embed_dim, scale=1):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act1 = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)
        self.in_channels = in_channels

        self.act2 = nn.SiLU()
        self.time_proj = nn.Linear(time_embed_dim, time_embed_dim * 6)
        self.scale = scale

    def timestep_embedding(self, t, dim, max_period=10000):
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
        encoder_hidden_states: Optional[torch.Tensor] = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        # Project and normalize query states
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

        # Process KV
        if self.is_cross_attention:
            encoder_hidden_shape = (*encoder_hidden_states.shape[:-1], -1, self.head_dim)
            key_states = self.k_norm(self.k_proj(encoder_hidden_states).view(encoder_hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(encoder_hidden_states).view(encoder_hidden_shape).transpose(1, 2)
        else:
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            if position_embeddings is not None:
                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Use DiffSynth unified attention
        attn_output = attention_forward(
            query_states, key_states, value_states,
            q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b s (n d)",
            window_size=None if attention_mask is None else attention_mask["window_size"],
        )
        attn_output = self.o_proj(attn_output)
        return attn_output


class AceStepDiTLayer(nn.Module):
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
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Extract scale-shift parameters for adaptive layer norm from timestep embeddings
        # 6 values: (shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table.to(temb.device) + temb
        ).chunk(6, dim=1)

        # Step 1: Self-attention with adaptive layer norm (AdaLN)
        # Apply adaptive normalization: norm(x) * (1 + scale) + shift
        norm_hidden_states = (self.self_attn_norm(hidden_states) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.self_attn(
            hidden_states=norm_hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        # Apply gated residual connection: x = x + attn_output * gate
        hidden_states = (hidden_states + attn_output * gate_msa).type_as(hidden_states)

        # Step 2: Cross-attention (if enabled) for conditioning on encoder outputs
        if self.use_cross_attention:
            norm_hidden_states = self.cross_attn_norm(hidden_states).type_as(hidden_states)
            attn_output = self.cross_attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
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

        return outputs


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class AceStepDiTModel(nn.Module):
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
        residual = None,
        output_residual = False,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):
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
        if hidden_states.shape[1] % self.patch_size != 0:
            pad_length = self.patch_size - (hidden_states.shape[1] % self.patch_size)
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_length), mode='constant', value=0)

        # Project input to patches and project encoder states
        hidden_states = self.proj_in(hidden_states)
        encoder_hidden_states = self.condition_embedder(encoder_hidden_states)

        # Cache positions and Position IDs
        cache_position = torch.arange(0, hidden_states.shape[1], device=hidden_states.device)
        position_ids = cache_position.unsqueeze(0)

        # Build mask mapping
        self_attn_mask_mapping = {
            "full_attention": None,
            "sliding_attention": {"window_size": self.sliding_window},
            "encoder_attention_mask": None,
        }

        # Create position embeddings to be shared across all decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Process through transformer layers
        generated_residual = []
        for index_block, layer_module in enumerate(self.layers):
            # Prepare layer arguments
            layer_args = (
                hidden_states,
                position_embeddings,
                timestep_proj,
                self_attn_mask_mapping[layer_module.attention_type],
                encoder_hidden_states,
                self_attn_mask_mapping["encoder_attention_mask"],
            )

            # Use gradient checkpointing if enabled
            layer_outputs = gradient_checkpoint_forward(
                layer_module,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                *layer_args,
            )
            hidden_states = layer_outputs[0]

            # Residual control
            if residual is not None:
                block_residual = residual[index_block]
                if block_residual.shape[1] > hidden_states.shape[1]:
                    block_residual = block_residual[:, :hidden_states.shape[1]]
                elif block_residual.shape[1] < hidden_states.shape[1]:
                    block_residual = torch.concat([block_residual, torch.zeros_like(hidden_states)[:, :hidden_states.shape[1] - block_residual.shape[1]]], dim=1)
                hidden_states = hidden_states + block_residual
            if output_residual:
                generated_residual.append(hidden_states)

        if return_hidden_states:
            return hidden_states
        if output_residual:
            return generated_residual

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
        return outputs
