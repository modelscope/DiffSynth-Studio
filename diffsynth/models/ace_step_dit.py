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
import time
from typing import Callable, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange

# Transformers imports (sorted by submodule, then alphabetically)
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, can_return_tuple, logging
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from vector_quantize_pytorch import ResidualFSQ

# Local config import with fallback



# Configuration class
from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation


class AceStepConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AceStepModel`]. It is used to instantiate an
    AceStep model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 64003):
            Vocabulary size of the AceStep model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling the model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
        head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from acestep.models import AceStepConfig

    >>> # Initializing an AceStep configuration
    >>> configuration = AceStepConfig()

    >>> # Initializing a model from the configuration
    >>> model = AceStepConditionGenerationModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "acestep"
    keys_to_ignore_at_inference = ["past_key_values"]
    
    # Default tensor parallel plan for the base model
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    def __init__(
        self,
        vocab_size=64003,
        fsq_dim=2048,
        fsq_input_levels=[8, 8, 8, 5, 5, 5],
        fsq_input_num_quantizers=1,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=True,
        rope_theta=1000000,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=True,
        sliding_window=128,
        layer_types=None,
        attention_dropout=0.0,
        num_lyric_encoder_hidden_layers=8,
        audio_acoustic_hidden_dim=64,
        pool_window_size=5,
        text_hidden_dim=1024,
        in_channels=192,
        data_proportion=0.5,
        timestep_mu=-0.4,
        timestep_sigma=1.0,
        timbre_hidden_dim=64,
        num_timbre_encoder_hidden_layers=4,
        timbre_fix_frame=750,
        patch_size=2,
        num_attention_pooler_hidden_layers=2,
        num_audio_decoder_hidden_layers=24,
        model_version="turbo",
        **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        
        # Text encoder configuration
        self.text_hidden_dim = text_hidden_dim

        # Lyric encoder configuration
        self.num_lyric_encoder_hidden_layers = num_lyric_encoder_hidden_layers
        self.patch_size = patch_size

        # Audio semantic token generation configuration
        self.audio_acoustic_hidden_dim = audio_acoustic_hidden_dim
        self.pool_window_size = pool_window_size
        self.in_channels = in_channels
        self.data_proportion = data_proportion
        self.timestep_mu = timestep_mu
        self.timestep_sigma = timestep_sigma
        
        # FSQ (Finite Scalar Quantization) configuration
        self.fsq_dim = fsq_dim
        self.fsq_input_levels = fsq_input_levels
        self.fsq_input_num_quantizers = fsq_input_num_quantizers
        
        # Timbre encoder configuration
        self.timbre_hidden_dim = timbre_hidden_dim
        self.num_timbre_encoder_hidden_layers = num_timbre_encoder_hidden_layers
        self.timbre_fix_frame = timbre_fix_frame
        self.num_attention_pooler_hidden_layers = num_attention_pooler_hidden_layers
        self.num_audio_decoder_hidden_layers = num_audio_decoder_hidden_layers
        self.vocab_size = vocab_size

        # Backward compatibility: ensure num_key_value_heads is set
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.model_version = model_version
        
        # Validate rotary position embeddings parameters
        # Backward compatibility: if there is a 'type' field, move it to 'rope_type'
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self.layer_types = layer_types

        # Set default layer types if not specified
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["AceStepConfig"]



logger = logging.get_logger(__name__)


def create_4d_mask(
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask: Optional[torch.Tensor] = None, # [Batch, Seq_Len]
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


def sample_t_r(batch_size, device, dtype, data_proportion=0.0, timestep_mu=-0.4, timestep_sigma=1.0, use_meanflow=True):
    """
    Sample timestep t and r for flow matching training.

    Args:
        batch_size: Batch size
        device: Device to create tensors on
        dtype: Data type for tensors
        data_proportion: Proportion of data samples (0.0 to 1.0)
        timestep_mu: Mean for timestep sampling
        timestep_sigma: Standard deviation for timestep sampling
        use_meanflow: Whether to use meanflow (if False, data_proportion is set to 1.0)

    Returns:
        Tuple of (t, r) tensors, each of shape [batch_size]
    """
    t = torch.sigmoid(torch.randn((batch_size,), device=device, dtype=dtype) * timestep_sigma + timestep_mu)
    r = torch.sigmoid(torch.randn((batch_size,), device=device, dtype=dtype) * timestep_sigma + timestep_mu)
    # Assign t = max, r = min, for each pair
    t, r = torch.maximum(t, r), torch.minimum(t, r)
    if not use_meanflow:
        data_proportion = 1.0
    data_size = int(batch_size * data_proportion)
    zero_mask = torch.arange(batch_size, device=device) < data_size
    r = torch.where(zero_mask, t, r)
    return t, r


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
        scale: float = 1000,
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

    def __init__(self, config: AceStepConfig, layer_idx: int, is_cross_attention: bool = False, is_causal: bool = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        if is_cross_attention:
            is_causal = False
        self.is_causal = is_causal
        self.is_cross_attention = is_cross_attention

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        # Apply RMS normalization only on the head dimension (unlike OLMo)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

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

        attention_interface: Callable = eager_attention_forward
        if is_cross_attention and output_attentions:
            attention_interface: Callable = eager_attention_forward
        elif self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window if not self.is_cross_attention else None,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class AceStepEncoderLayer(GradientCheckpointingLayer):
    """
    Encoder layer for AceStep model.

    Consists of self-attention and MLP (feed-forward) sub-layers with residual connections.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx

        # Self-attention sub-layer
        self.self_attn = AceStepAttention(
            config=config,
            layer_idx=layer_idx,
            is_cross_attention=False,
            is_causal=False,
        )
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MLP (feed-forward) sub-layer
        self.mlp = Qwen3MLP(config)
        self.attention_type = config.layer_types[layer_idx]

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


class AceStepDiTLayer(GradientCheckpointingLayer):
    """
    DiT (Diffusion Transformer) layer for AceStep model.
    
    Implements a transformer layer with three main components:
    1. Self-attention with adaptive layer norm (AdaLN)
    2. Cross-attention (optional) for conditioning on encoder outputs
    3. Feed-forward MLP with adaptive layer norm
    
    Uses scale-shift modulation from timestep embeddings for adaptive normalization.
    """
    def __init__(self, config: AceStepConfig, layer_idx: int, use_cross_attention: bool = True):
        super().__init__()

        # 1. Self-attention sub-layer with adaptive normalization
        self.self_attn_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = AceStepAttention(config=config, layer_idx=layer_idx)

        # 2. Cross-attention sub-layer (optional, for encoder conditioning)
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.cross_attn_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.cross_attn = AceStepAttention(config=config, layer_idx=layer_idx, is_cross_attention=True)

        # 3. Feed-forward MLP sub-layer with adaptive normalization
        self.mlp_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)

        # Scale-shift table for adaptive layer norm modulation (6 values: 3 for self-attn, 3 for MLP)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, config.hidden_size) / config.hidden_size**0.5)
        self.attention_type = config.layer_types[layer_idx]
    
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
            self.scale_shift_table + temb
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


@auto_docstring
class AceStepPreTrainedModel(PreTrainedModel):
    config_class = AceStepConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AceStepEncoderLayer", "AceStepDiTLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_3 = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        """
        Initialize weights for different module types.

        TODO: Support separate initialization for encoders and decoders.
        """
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen3RMSNorm):
            module.weight.data.fill_(1.0)


class AceStepLyricEncoder(AceStepPreTrainedModel):
    """
    Encoder for processing lyric text embeddings.
    
    Encodes lyric text hidden states using a transformer encoder architecture
    with bidirectional attention. Projects text embeddings to model hidden size
    and processes them through multiple encoder layers.
    """
    def __init__(self, config):
        super().__init__(config)
        
        # Project text embeddings to model hidden size
        self.embed_tokens = nn.Linear(config.text_hidden_dim, config.hidden_size)
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Stack of encoder layers
        self.layers = nn.ModuleList(
            [AceStepEncoderLayer(config, layer_idx) for layer_idx in range(config.num_lyric_encoder_hidden_layers)]
        )

        # Initialize weights and apply final processing
        self.post_init()

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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        assert input_ids is None, "Only `input_ids` is supported for the lyric encoder."
        assert attention_mask is not None, "Attention mask must be provided for the lyric encoder."
        assert inputs_embeds is not None, "Inputs embeddings must be provided for the lyric encoder."
        
        # Project input embeddings: N x T x text_hidden_dim -> N x T x hidden_size
        inputs_embeds = self.embed_tokens(inputs_embeds)
        # Cache position: only used for mask construction (not for actual caching)
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)

        # Positional IDs
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Attention masks
        seq_len = inputs_embeds.shape[1]
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device
        
        # 判断是否使用 Flash Attention 2
        is_flash_attn = (self.config._attn_implementation == "flash_attention_2")

        # 初始化 Mask 变量
        full_attn_mask = None
        sliding_attn_mask = None

        if is_flash_attn:
            # -------------------------------------------------------
            # 场景 A: Flash Attention 模式
            # -------------------------------------------------------
            # FA 不需要 4D Mask。
            # 如果有 padding mask (attention_mask [B, L])，直接传给它即可。
            # 如果没有 padding mask，传 None。
            # 滑动窗口逻辑由 Layer 内部传给 FA kernel 的 sliding_window 参数控制。
            full_attn_mask = attention_mask
            
            # 这里的逻辑是：如果配置启用了滑动窗口，FA 模式下我们也只需要传基础的 padding mask
            # Layer 会自己决定是否调用带 sliding window 的 kernel
            sliding_attn_mask = attention_mask if self.config.use_sliding_window else None

        else:
            # -------------------------------------------------------
            # 场景 B: CPU / Mac / SDPA (Eager 模式)
            # -------------------------------------------------------
            # 必须手动生成 4D Mask [B, 1, L, L]
            
            # 1. Full Attention (Bidirectional, Global)
            # 对应原来的 create_causal_mask + bidirectional
            full_attn_mask = create_4d_mask(
                seq_len=seq_len,
                dtype=dtype,
                device=device,
                attention_mask=attention_mask,     # [B, L]
                sliding_window=None,
                is_sliding_window=False,
                is_causal=False                    # <--- 关键：双向注意力
            )

            # 2. Sliding Attention (Bidirectional, Local)
            # 对应原来的 create_sliding_window... + bidirectional
            if self.config.use_sliding_window:
                sliding_attn_mask = create_4d_mask(
                    seq_len=seq_len,
                    dtype=dtype,
                    device=device,
                    attention_mask=attention_mask, # [B, L]
                    sliding_window=self.config.sliding_window,
                    is_sliding_window=True,        # <--- 开启滑动窗口
                    is_causal=False                # <--- 关键：双向注意力
                )

        # 构建 Mapping
        self_attn_mask_mapping = {
            "full_attention": full_attn_mask,
            "sliding_attention": sliding_attn_mask,
        }

        # Initialize hidden states with input embeddings
        hidden_states = inputs_embeds

        # Create position embeddings to be shared across all layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Pass through transformer layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer_module in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                position_embeddings,
                self_attn_mask_mapping[layer_module.attention_type],
                position_ids,
                output_attentions,
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


class AttentionPooler(AceStepPreTrainedModel):
    """
    Attention-based pooling module.
    
    Pools sequences of patches using a special token and attention mechanism.
    The special token attends to all patches and its output is used as the
    pooled representation. Used for aggregating patch-level features into
    sequence-level representations.
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Special token used for pooling (CLS-like token)
        self.special_token = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)
        self.layers = nn.ModuleList(
            [AceStepEncoderLayer(config, layer_idx) for layer_idx in range(config.num_attention_pooler_hidden_layers)]
        )

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(self,
        x,
        attention_mask: Optional[torch.Tensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutput:
        B, T, P, D = x.shape
        x = self.embed_tokens(x)
        special_tokens = self.special_token.expand(B, T, 1, -1)
        x = torch.cat([special_tokens, x], dim=2)
        x = rearrange(x, "b t p c -> (b t) p c")

        # Cache position: only used for mask construction.
        cache_position = torch.arange(0, x.shape[1], device=x.device)
        # Postional ids.
        position_ids = cache_position.unsqueeze(0)

        # embed positions
        hidden_states = x

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        seq_len = x.shape[1]
        dtype = x.dtype
        device = x.device
        
        # 判断是否使用 Flash Attention 2
        is_flash_attn = (self.config._attn_implementation == "flash_attention_2")

        # 初始化 Mask 变量
        full_attn_mask = None
        sliding_attn_mask = None

        if is_flash_attn:
            # -------------------------------------------------------
            # 场景 A: Flash Attention 模式
            # -------------------------------------------------------
            # FA 不需要 4D Mask。
            # 如果有 padding mask (attention_mask [B, L])，直接传给它即可。
            # 如果没有 padding mask，传 None。
            # 滑动窗口逻辑由 Layer 内部传给 FA kernel 的 sliding_window 参数控制。
            full_attn_mask = attention_mask
            
            # 这里的逻辑是：如果配置启用了滑动窗口，FA 模式下我们也只需要传基础的 padding mask
            # Layer 会自己决定是否调用带 sliding window 的 kernel
            sliding_attn_mask = attention_mask if self.config.use_sliding_window else None

        else:
            # -------------------------------------------------------
            # 场景 B: CPU / Mac / SDPA (Eager 模式)
            # -------------------------------------------------------
            # 必须手动生成 4D Mask [B, 1, L, L]
            
            # 1. Full Attention (Bidirectional, Global)
            # 对应原来的 create_causal_mask + bidirectional
            full_attn_mask = create_4d_mask(
                seq_len=seq_len,
                dtype=dtype,
                device=device,
                attention_mask=attention_mask,     # [B, L]
                sliding_window=None,
                is_sliding_window=False,
                is_causal=False                    # <--- 关键：双向注意力
            )

            # 2. Sliding Attention (Bidirectional, Local)
            # 对应原来的 create_sliding_window... + bidirectional
            if self.config.use_sliding_window:
                sliding_attn_mask = create_4d_mask(
                    seq_len=seq_len,
                    dtype=dtype,
                    device=device,
                    attention_mask=attention_mask, # [B, L]
                    sliding_window=self.config.sliding_window,
                    is_sliding_window=True,        # <--- 开启滑动窗口
                    is_causal=False                # <--- 关键：双向注意力
                )

        # 构建 Mapping
        self_attn_mask_mapping = {
            "full_attention": full_attn_mask,
            "sliding_attention": sliding_attn_mask,
        }
        
        for layer_module in self.layers:
            layer_outputs = layer_module(
                hidden_states,
                position_embeddings,
                attention_mask=self_attn_mask_mapping[layer_module.attention_type],
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        
        # Extract the special token output (first position) as pooled representation
        cls_output = hidden_states[:, 0, :]
        cls_output = rearrange(cls_output, "(b t) c -> b t c", b=B)
        return cls_output


class AudioTokenDetokenizer(AceStepPreTrainedModel):
    """
    Audio token detokenizer module.
    
    Converts quantized audio tokens back to continuous acoustic representations.
    Expands each token into multiple patches using special tokens, processes them
    through encoder layers, and projects to acoustic hidden dimension.
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Special tokens for expanding each quantized token into patches
        self.special_tokens = nn.Parameter(torch.randn(1, config.pool_window_size, config.hidden_size) * 0.02)
        self.layers = nn.ModuleList(
            [AceStepEncoderLayer(config, layer_idx) for layer_idx in range(config.num_attention_pooler_hidden_layers)]
        )
        # Project back to acoustic hidden dimension
        self.proj_out = nn.Linear(config.hidden_size, config.audio_acoustic_hidden_dim)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(self,
        x,
        attention_mask: Optional[torch.Tensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutput:
        B, T, D = x.shape
        x = self.embed_tokens(x)
        # Expand and add special tokens: N x T x D -> N x T x P x D
        # Each token is expanded into pool_window_size patches
        x = x.unsqueeze(2)  # N x T x 1 x D
        x = x.repeat(1, 1, self.config.pool_window_size, 1)  # N x T x P x D
        # Add learnable special tokens to each patch
        special_tokens = self.special_tokens.expand(B, T, -1, -1)
        x = x + special_tokens
        # Reshape for processing: (batch * time) x patches x hidden
        x = rearrange(x, "b t p c -> (b t) p c")

        # Cache position: only used for mask construction
        cache_position = torch.arange(0, x.shape[1], device=x.device)
        # Positional IDs
        position_ids = cache_position.unsqueeze(0)

        # Initialize hidden states
        hidden_states = x

        # Create position embeddings to be shared across all layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        seq_len = x.shape[1]
        dtype = x.dtype
        device = x.device
        
        # 判断是否使用 Flash Attention 2
        is_flash_attn = (self.config._attn_implementation == "flash_attention_2")

        # 初始化 Mask 变量
        full_attn_mask = None
        sliding_attn_mask = None

        if is_flash_attn:
            # -------------------------------------------------------
            # 场景 A: Flash Attention 模式
            # -------------------------------------------------------
            # FA 不需要 4D Mask。
            # 如果有 padding mask (attention_mask [B, L])，直接传给它即可。
            # 如果没有 padding mask，传 None。
            # 滑动窗口逻辑由 Layer 内部传给 FA kernel 的 sliding_window 参数控制。
            full_attn_mask = attention_mask
            
            # 这里的逻辑是：如果配置启用了滑动窗口，FA 模式下我们也只需要传基础的 padding mask
            # Layer 会自己决定是否调用带 sliding window 的 kernel
            sliding_attn_mask = attention_mask if self.config.use_sliding_window else None

        else:
            # -------------------------------------------------------
            # 场景 B: CPU / Mac / SDPA (Eager 模式)
            # -------------------------------------------------------
            # 必须手动生成 4D Mask [B, 1, L, L]
            
            # 1. Full Attention (Bidirectional, Global)
            # 对应原来的 create_causal_mask + bidirectional
            full_attn_mask = create_4d_mask(
                seq_len=seq_len,
                dtype=dtype,
                device=device,
                attention_mask=attention_mask,     # [B, L]
                sliding_window=None,
                is_sliding_window=False,
                is_causal=False                    # <--- 关键：双向注意力
            )

            # 2. Sliding Attention (Bidirectional, Local)
            # 对应原来的 create_sliding_window... + bidirectional
            if self.config.use_sliding_window:
                sliding_attn_mask = create_4d_mask(
                    seq_len=seq_len,
                    dtype=dtype,
                    device=device,
                    attention_mask=attention_mask, # [B, L]
                    sliding_window=self.config.sliding_window,
                    is_sliding_window=True,        # <--- 开启滑动窗口
                    is_causal=False                # <--- 关键：双向注意力
                )

        # 构建 Mapping
        self_attn_mask_mapping = {
            "full_attention": full_attn_mask,
            "sliding_attention": sliding_attn_mask,
        }
        
        for layer_module in self.layers:
            layer_outputs = layer_module(
                hidden_states,
                position_embeddings,
                attention_mask=self_attn_mask_mapping[layer_module.attention_type],
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        
        hidden_states = self.proj_out(hidden_states)

        hidden_states = rearrange(hidden_states, "(b t) p c -> b (t p) c", b=B, p=self.config.pool_window_size)
        return hidden_states


class AceStepTimbreEncoder(AceStepPreTrainedModel):
    """
    Encoder for extracting timbre embeddings from reference audio.
    
    Processes packed reference audio acoustic features to extract timbre
    representations. Uses a special token (CLS-like) to aggregate information
    from the entire reference audio sequence. Outputs are unpacked back to
    batch format for use in conditioning.
    """
    def __init__(self, config):
        super().__init__(config)
        
        # Project acoustic features to model hidden size
        self.embed_tokens = nn.Linear(config.timbre_hidden_dim, config.hidden_size)
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Special token for aggregating timbre information (prepended to sequence)
        self.special_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.layers = nn.ModuleList(
            [AceStepEncoderLayer(config, layer_idx) for layer_idx in range(config.num_timbre_encoder_hidden_layers)]
        )

        # Initialize weights and apply final processing
        self.post_init()

    def unpack_timbre_embeddings(self, timbre_embs_packed, refer_audio_order_mask):
        """
        Unpack packed timbre embeddings into batch format.

        Args:
            timbre_embs_packed: Packed timbre embeddings of shape [N, d]
            refer_audio_order_mask: Order mask indicating batch assignment for each packed embedding

        Returns:
            Tuple of (unpacked_embeddings, mask):
            - unpacked_embeddings: Unpacked embeddings of shape [B, max_count, d]
            - new_mask: Mask indicating valid positions, shape [B, max_count]
        """
        N, d = timbre_embs_packed.shape
        device = timbre_embs_packed.device
        dtype = timbre_embs_packed.dtype
        
        # Get batch size
        B = int(refer_audio_order_mask.max().item() + 1)
        
        # Calculate element count and positions for each batch
        counts = torch.bincount(refer_audio_order_mask, minlength=B)
        max_count = counts.max().item()
        
        # Calculate positions within batch
        sorted_indices = torch.argsort(refer_audio_order_mask * N + torch.arange(N, device=device), stable=True)
        sorted_batch_ids = refer_audio_order_mask[sorted_indices]
        
        positions = torch.arange(N, device=device)
        batch_starts = torch.cat([torch.tensor([0], device=device), 
                                torch.cumsum(counts, dim=0)[:-1]])
        positions_in_sorted = positions - batch_starts[sorted_batch_ids]
        
        inverse_indices = torch.empty_like(sorted_indices)
        inverse_indices[sorted_indices] = torch.arange(N, device=device)
        positions_in_batch = positions_in_sorted[inverse_indices]
        
        # Use one-hot encoding and matrix multiplication (gradient-friendly approach)
        # Create one-hot encoding
        indices_2d = refer_audio_order_mask * max_count + positions_in_batch  # (N,)
        one_hot = F.one_hot(indices_2d, num_classes=B * max_count).to(dtype)  # (N, B*max_count)
        
        # Rearrange using matrix multiplication
        timbre_embs_flat = one_hot.t() @ timbre_embs_packed  # (B*max_count, d)
        timbre_embs_unpack = timbre_embs_flat.reshape(B, max_count, d)
        
        # Create mask indicating valid positions
        mask_flat = (one_hot.sum(dim=0) > 0).long()  # (B*max_count,)
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
        # Project embeddings: N x T x timbre_hidden_dim -> N x T x hidden_size
        inputs_embeds = self.embed_tokens(inputs_embeds)
        # Prepend special token for timbre aggregation (CLS-like token)
        # inputs_embeds = torch.cat([self.special_token.expand(inputs_embeds.shape[0], 1, -1), inputs_embeds], dim=1)
        # Cache position: only used for mask construction (not for actual caching)
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        # Positional IDs
        position_ids = cache_position.unsqueeze(0)

        seq_len = inputs_embeds.shape[1]
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device
        
        # 判断是否使用 Flash Attention 2
        is_flash_attn = (self.config._attn_implementation == "flash_attention_2")

        # 初始化 Mask 变量
        full_attn_mask = None
        sliding_attn_mask = None

        if is_flash_attn:
            # -------------------------------------------------------
            # 场景 A: Flash Attention 模式
            # -------------------------------------------------------
            # FA 不需要 4D Mask。
            # 如果有 padding mask (attention_mask [B, L])，直接传给它即可。
            # 如果没有 padding mask，传 None。
            # 滑动窗口逻辑由 Layer 内部传给 FA kernel 的 sliding_window 参数控制。
            full_attn_mask = attention_mask
            
            # 这里的逻辑是：如果配置启用了滑动窗口，FA 模式下我们也只需要传基础的 padding mask
            # Layer 会自己决定是否调用带 sliding window 的 kernel
            sliding_attn_mask = attention_mask if self.config.use_sliding_window else None

        else:
            # -------------------------------------------------------
            # 场景 B: CPU / Mac / SDPA (Eager 模式)
            # -------------------------------------------------------
            # 必须手动生成 4D Mask [B, 1, L, L]
            
            # 1. Full Attention (Bidirectional, Global)
            # 对应原来的 create_causal_mask + bidirectional
            full_attn_mask = create_4d_mask(
                seq_len=seq_len,
                dtype=dtype,
                device=device,
                attention_mask=attention_mask,     # [B, L]
                sliding_window=None,
                is_sliding_window=False,
                is_causal=False                    # <--- 关键：双向注意力
            )

            # 2. Sliding Attention (Bidirectional, Local)
            # 对应原来的 create_sliding_window... + bidirectional
            if self.config.use_sliding_window:
                sliding_attn_mask = create_4d_mask(
                    seq_len=seq_len,
                    dtype=dtype,
                    device=device,
                    attention_mask=attention_mask, # [B, L]
                    sliding_window=self.config.sliding_window,
                    is_sliding_window=True,        # <--- 开启滑动窗口
                    is_causal=False                # <--- 关键：双向注意力
                )

        # 构建 Mapping
        self_attn_mask_mapping = {
            "full_attention": full_attn_mask,
            "sliding_attention": sliding_attn_mask,
        }
        
        # Initialize hidden states
        hidden_states = inputs_embeds

        # Create position embeddings to be shared across all layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Pass through transformer layers
        for layer_module in self.layers[: self.config.num_hidden_layers]:
            layer_outputs = layer_module(
                hidden_states,
                position_embeddings,
                self_attn_mask_mapping[layer_module.attention_type],
                position_ids,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        # Extract special token output (first position) as timbre embedding: N x T x D -> N x D
        hidden_states = hidden_states[:, 0, :]
        # Unpack packed embeddings back to batch format
        timbre_embs_unpack, timbre_embs_mask = self.unpack_timbre_embeddings(hidden_states, refer_audio_order_mask)
        return timbre_embs_unpack, timbre_embs_mask


class AceStepAudioTokenizer(AceStepPreTrainedModel):
    """
    Audio tokenizer module.
    
    Converts continuous acoustic features into discrete quantized tokens.
    Process: project -> pool patches -> quantize. Used for converting audio
    representations into discrete tokens for processing by the diffusion model.
    """
    def __init__(self, config):
        super().__init__(config)
        # Project acoustic features to hidden size
        self.audio_acoustic_proj = nn.Linear(config.audio_acoustic_hidden_dim, config.hidden_size)
        # Pool patches into sequence-level representations
        self.attention_pooler = AttentionPooler(config)
        # Quantize continuous representations into discrete tokens
        self.quantizer = ResidualFSQ(
            dim=config.fsq_dim,
            levels=config.fsq_input_levels,
            num_quantizers=config.fsq_input_num_quantizers
        )
        self.pool_window_size = config.pool_window_size
        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutput:
        
        # Project acoustic features to hidden size
        hidden_states = self.audio_acoustic_proj(hidden_states)
        # Pool sequences: N x T//pool_window_size x pool_window_size x d -> N x T//pool_window_size x d
        hidden_states = self.attention_pooler(hidden_states)
        # Quantize continuous representations into discrete tokens: N x T//pool_window_size x d
        quantized, indices = self.quantizer(hidden_states)
        return quantized, indices

    def tokenize(self, x):
        x = rearrange(x, 'n (t_patch p) d -> n t_patch p d', p=self.pool_window_size)
        quantized, indices = self.forward(x)
        return quantized, indices

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


class AceStepDiTModel(AceStepPreTrainedModel):
    """
    DiT (Diffusion Transformer) model for AceStep.
    
    Main diffusion model that generates audio latents conditioned on text, lyrics,
    and timbre. Uses patch-based processing with transformer layers, timestep
    conditioning, and cross-attention to encoder outputs.
    """
    def __init__(self, config: AceStepConfig):
        super().__init__(config)
        # Rotary position embeddings for transformer layers
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        # Stack of DiT transformer layers
        self.layers = nn.ModuleList(
            [AceStepDiTLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        in_channels = config.in_channels
        inner_dim = config.hidden_size
        patch_size = config.patch_size
        self.patch_size = patch_size
        
        # Input projection: patch embedding using 1D convolution
        # Converts sequence into patches for efficient processing
        self.proj_in = nn.Sequential(
            Lambda(lambda x: x.transpose(1, 2)),  # [B, T, C] -> [B, C, T]
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=inner_dim,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
            ),
            Lambda(lambda x: x.transpose(1, 2)),  # [B, C, T//patch_size] -> [B, T//patch_size, C]
        )

        # Timestep embeddings for diffusion conditioning
        # Two embeddings: one for timestep t, one for timestep difference (t - r)
        self.time_embed = TimestepEmbedding(in_channels=256, time_embed_dim=inner_dim)
        self.time_embed_r = TimestepEmbedding(in_channels=256, time_embed_dim=inner_dim)
        
        # Project encoder hidden states to model dimension
        self.condition_embedder = nn.Linear(inner_dim, inner_dim, bias=True)

        # Output normalization and projection
        # Adaptive layer norm with scale-shift modulation, then de-patchify
        self.norm_out = Qwen3RMSNorm(inner_dim, eps=config.rms_norm_eps)
        self.proj_out = nn.Sequential(
            Lambda(lambda x: x.transpose(1, 2)),  # [B, T//patch_size, inner_dim] -> [B, inner_dim, T//patch_size]
            nn.ConvTranspose1d(
                in_channels=inner_dim,
                out_channels=config.audio_acoustic_hidden_dim,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
            ),
            Lambda(lambda x: x.transpose(1, 2)),  # [B, out_channels, T] -> [B, T, out_channels]
        )
        # Scale-shift table for adaptive output normalization (2 values: shift, scale)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

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
        use_cache: Optional[bool] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        return_hidden_states: int = None,
        custom_layers_config: Optional[dict] = None,
        enable_early_exit: bool = False,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache

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
        
        # 判断是否使用 Flash Attention 2
        is_flash_attn = (self.config._attn_implementation == "flash_attention_2")

        # 初始化 Mask 变量
        full_attn_mask = None
        sliding_attn_mask = None
        encoder_attention_mask = None
        attention_mask = None
        if is_flash_attn:
            # -------------------------------------------------------
            # 场景 A: Flash Attention 模式
            # -------------------------------------------------------
            # FA 不需要 4D Mask。
            # 如果有 padding mask (attention_mask [B, L])，直接传给它即可。
            # 如果没有 padding mask，传 None。
            # 滑动窗口逻辑由 Layer 内部传给 FA kernel 的 sliding_window 参数控制。
            full_attn_mask = attention_mask
            
            # 这里的逻辑是：如果配置启用了滑动窗口，FA 模式下我们也只需要传基础的 padding mask
            # Layer 会自己决定是否调用带 sliding window 的 kernel
            sliding_attn_mask = attention_mask if self.config.use_sliding_window else None

        else:
            # -------------------------------------------------------
            # 场景 B: CPU / Mac / SDPA (Eager 模式)
            # -------------------------------------------------------
            # 必须手动生成 4D Mask [B, 1, L, L]
            
            # 1. Full Attention (Bidirectional, Global)
            # 对应原来的 create_causal_mask + bidirectional
            full_attn_mask = create_4d_mask(
                seq_len=seq_len,
                dtype=dtype,
                device=device,
                attention_mask=attention_mask,     # [B, L]
                sliding_window=None,
                is_sliding_window=False,
                is_causal=False                    # <--- 关键：双向注意力
            )
            max_len = max(seq_len, encoder_seq_len)
            
            encoder_attention_mask = create_4d_mask(
                seq_len=max_len,
                dtype=dtype,
                device=device,
                attention_mask=attention_mask,     # [B, L]
                sliding_window=None,
                is_sliding_window=False,
                is_causal=False                    # <--- 关键：双向注意力
            )
            encoder_attention_mask = encoder_attention_mask[:, :, :seq_len, :encoder_seq_len]
            # 2. Sliding Attention (Bidirectional, Local)
            # 对应原来的 create_sliding_window... + bidirectional
            if self.config.use_sliding_window:
                sliding_attn_mask = create_4d_mask(
                    seq_len=seq_len,
                    dtype=dtype,
                    device=device,
                    attention_mask=attention_mask, # [B, L]
                    sliding_window=self.config.sliding_window,
                    is_sliding_window=True,        # <--- 开启滑动窗口
                    is_causal=False                # <--- 关键：双向注意力
                )

        # 构建 Mapping
        self_attn_mask_mapping = {
            "full_attention": full_attn_mask,
            "sliding_attention": sliding_attn_mask,
            "encoder_attention_mask": encoder_attention_mask,
        }

        # Create position embeddings to be shared across all decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_cross_attentions = () if output_attentions else None

        # Handle early exit for custom layer configurations
        max_needed_layer = float('inf')
        if custom_layers_config is not None and enable_early_exit:
            max_needed_layer = max(custom_layers_config.keys())
            # Force output_attentions to True when early exit is enabled for attention extraction
            output_attentions = True
            if all_cross_attentions is None:
                all_cross_attentions = ()

        # Process through transformer layers
        for index_block, layer_module in enumerate(self.layers):

            layer_outputs = layer_module(
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
                **flash_attn_kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions and self.layers[index_block].use_cross_attention:
                # layer_outputs structure: (hidden_states, self_attn_weights, cross_attn_weights)
                # Extract the last element which is cross_attn_weights
                if len(layer_outputs) >= 3:
                    all_cross_attentions += (layer_outputs[2],)
        
        if return_hidden_states:
            return hidden_states

        # Extract scale-shift parameters for adaptive output normalization
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
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

class AceStepConditionEncoder(AceStepPreTrainedModel):
    """
    Condition encoder for AceStep model.
    
    Encodes multiple conditioning inputs (text, lyrics, timbre) and packs them
    into a single sequence for cross-attention in the diffusion model. Handles
    projection, encoding, and sequence packing.
    """
    def __init__(self, config: AceStepConfig):
        super().__init__(config)
        self.config = config
        # Project text embeddings to model hidden size
        self.text_projector = nn.Linear(config.text_hidden_dim, config.hidden_size, bias=False)
        # Encoder for lyric text
        self.lyric_encoder = AceStepLyricEncoder(config)
        # Encoder for timbre from reference audio
        self.timbre_encoder = AceStepTimbreEncoder(config)

    def forward(
        self,
        # Text inputs
        text_hidden_states: Optional[torch.FloatTensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        # Lyric inputs
        lyric_hidden_states: Optional[torch.LongTensor] = None,
        lyric_attention_mask: Optional[torch.Tensor] = None,
        # Reference audio for timbre
        refer_audio_acoustic_hidden_states_packed: Optional[torch.Tensor] = None,
        refer_audio_order_mask: Optional[torch.LongTensor] = None,
    ):
        # Project and encode text
        text_hidden_states = self.text_projector(text_hidden_states)
        # Encode lyrics
        lyric_encoder_outputs = self.lyric_encoder(
            inputs_embeds=lyric_hidden_states,
            attention_mask=lyric_attention_mask,
        )
        lyric_hidden_states = lyric_encoder_outputs.last_hidden_state
        # Encode timbre from reference audio
        timbre_embs_unpack, timbre_embs_mask = self.timbre_encoder(refer_audio_acoustic_hidden_states_packed, refer_audio_order_mask)

        # Pack sequences: combine lyrics and timbre, then add text
        # This creates a single sequence with all conditioning information
        encoder_hidden_states, encoder_attention_mask = pack_sequences(lyric_hidden_states, timbre_embs_unpack, lyric_attention_mask, timbre_embs_mask)
        encoder_hidden_states, encoder_attention_mask = pack_sequences(encoder_hidden_states, text_hidden_states, encoder_attention_mask, text_attention_mask)
        return encoder_hidden_states, encoder_attention_mask


class AceStepConditionGenerationModel(AceStepPreTrainedModel):
    """
    Main conditional generation model for AceStep.
    
    End-to-end model for generating audio conditioned on text, lyrics, and timbre.
    Combines encoder (for conditioning), decoder (diffusion model), tokenizer
    (for discrete tokenization), and detokenizer (for reconstruction).
    Supports flow matching training and inference with various sampling methods.
    """
    def __init__(self, config: AceStepConfig):
        super().__init__(config)
        self.config = config
        # Diffusion model components
        self.decoder = AceStepDiTModel(config)  # Main diffusion transformer
        self.encoder = AceStepConditionEncoder(config)  # Condition encoder
        self.tokenizer = AceStepAudioTokenizer(config)  # Audio tokenizer
        self.detokenizer = AudioTokenDetokenizer(config)  # Audio detokenizer
        # Null condition embedding for classifier-free guidance
        self.null_condition_emb = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        # Initialize weights and apply final processing
        self.post_init()

    def tokenize(self, x, silence_latent, attention_mask):
        if x.shape[1] % self.config.pool_window_size != 0:
            pad_len = self.config.pool_window_size - (x.shape[1] % self.config.pool_window_size)
            x = torch.cat([x,  silence_latent[:1,:pad_len].repeat(x.shape[0],1,1)], dim=1)
            attention_mask = F.pad(attention_mask, (0, pad_len), mode='constant', value=0)
        x = rearrange(x, 'n (t_patch p) d -> n t_patch p d', p=self.config.pool_window_size)
        seq_len = x.shape[1]
        chunk = math.ceil(attention_mask.shape[1] / seq_len)
        attention_mask = attention_mask.to(x.dtype)
        attention_mask = F.max_pool1d(attention_mask.unsqueeze(1), kernel_size=chunk, stride=chunk, ceil_mode=True).squeeze(1)
        quantized, indices = self.tokenizer(x)
        return quantized, indices, attention_mask

    def detokenize(self, quantized):
        """
        Detokenize quantized audio tokens back to continuous representations.

        Args:
            quantized: Quantized tokens of shape [N, T//pool_window_size, d]

        Returns:
            Detokenized hidden states of shape [N, T, d]
        """
        hidden_states = self.detokenizer(quantized)
        return hidden_states

    @torch.no_grad()
    def prepare_condition(
        self,
        text_hidden_states: torch.FloatTensor,
        text_attention_mask: torch.Tensor,
        lyric_hidden_states: torch.FloatTensor,
        lyric_attention_mask: torch.Tensor,
        refer_audio_acoustic_hidden_states_packed: torch.FloatTensor,
        refer_audio_order_mask: torch.Tensor,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.Tensor,
        silence_latent: torch.FloatTensor,
        src_latents: torch.FloatTensor,
        chunk_masks: torch.Tensor,
        is_covers: torch.Tensor,
        precomputed_lm_hints_25Hz: Optional[torch.FloatTensor] = None,
        audio_codes: torch.FloatTensor = None,
    ):
        
        dtype = hidden_states.dtype
        encoder_hidden_states, encoder_attention_mask = self.encoder(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask=refer_audio_order_mask,
        )

        # N x T x d -> N x T//pool_window_size x pool_window_size x d
        # tokenize and detokenize to get LM hints for cover songs (when is_covers=True)
        # Use precomputed hints if provided (e.g., from audio codes), otherwise tokenize hidden_states
        if precomputed_lm_hints_25Hz is not None:
            print("Using precomputed LM hints")
            lm_hints_25Hz = precomputed_lm_hints_25Hz[:, :src_latents.shape[1], :]
        else:
            if audio_codes is not None:
                lm_hints_5Hz = self.tokenize.quantizer.get_output_from_indices(audio_codes)
            else:
                lm_hints_5Hz, indices, llm_mask = self.tokenize(hidden_states, silence_latent, attention_mask)
            lm_hints_25Hz = self.detokenize(lm_hints_5Hz)
            # Crop lm_hints_25Hz to match src_latents length (tokenize may have added padding)
            lm_hints_25Hz = lm_hints_25Hz[:, :src_latents.shape[1], :]
        src_latents = torch.where(is_covers.unsqueeze(-1).unsqueeze(-1) > 0, lm_hints_25Hz, src_latents)
        # Concatenate source latents with chunk masks as context
        context_latents = torch.cat([src_latents, chunk_masks.to(dtype)], dim=-1)
        return encoder_hidden_states, encoder_attention_mask, context_latents

    def forward(
        self,
        # Diffusion inputs
        hidden_states: torch.FloatTensor,
        attention_mask: torch.Tensor,
        # Encoder inputs
        # Text
        text_hidden_states: Optional[torch.FloatTensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        # Lyric
        lyric_hidden_states: Optional[torch.LongTensor] = None,
        lyric_attention_mask: Optional[torch.Tensor] = None,
        # Reference audio for timbre
        refer_audio_acoustic_hidden_states_packed: Optional[torch.Tensor] = None,
        refer_audio_order_mask: Optional[torch.LongTensor] = None,
        src_latents: torch.FloatTensor = None,
        chunk_masks: torch.FloatTensor = None,
        is_covers: torch.Tensor = None,
        silence_latent: torch.FloatTensor = None,
        cfg_ratio: float = 0.15,
    ):
        """
        Forward pass for training (computes training losses).
        """
        # Prepare conditioning inputs (encoder states, context latents)
        encoder_hidden_states, encoder_attention_mask, context_latents = self.prepare_condition(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask=refer_audio_order_mask,
            hidden_states=src_latents,
            attention_mask=attention_mask,
            silence_latent=silence_latent,
            src_latents=src_latents,
            chunk_masks=chunk_masks,
            is_covers=is_covers,
        )
        bsz, device, dtype = hidden_states.shape[0], hidden_states.device, hidden_states.dtype
        # Classifier-free guidance: randomly drop conditions with probability cfg_ratio
        # This helps the model learn to work with and without conditions
        full_cfg_condition_mask = torch.where(
            (torch.rand(size=(bsz,), device=device, dtype=dtype) < cfg_ratio),
            torch.zeros(size=(bsz,), device=device, dtype=dtype),
            torch.ones(size=(bsz,), device=device, dtype=dtype)
        ).view(-1, 1, 1)
        # Replace dropped conditions with null condition embedding
        encoder_hidden_states = torch.where(full_cfg_condition_mask > 0, encoder_hidden_states, self.null_condition_emb.expand_as(encoder_hidden_states))

        # Flow matching setup: sample noise x1 and interpolate with data x0
        x1 = torch.randn_like(hidden_states)  # Noise
        x0 = hidden_states  # Data
        # Sample timesteps t and r for flow matching
        t, r = sample_t_r(bsz, device, dtype, self.config.data_proportion, self.config.timestep_mu, self.config.timestep_sigma, use_meanflow=False)
        t_ = t.unsqueeze(-1).unsqueeze(-1)
        # Interpolate: x_t = t * x1 + (1 - t) * x0
        xt = t_ * x1 + (1.0 - t_) * x0
        
        # Predict flow (velocity) from diffusion model
        decoder_outputs = self.decoder(
            hidden_states=xt,
            timestep=t,
            timestep_r=t,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
        )
        # Flow matching loss: predict the flow field v = x1 - x0
        flow = x1 - x0
        diffusion_loss = F.mse_loss(decoder_outputs[0], flow)
        return {
            "diffusion_loss": diffusion_loss,
        }
        
    def training_losses(self, **kwargs):
        return self.forward(**kwargs)
    
    def prepare_noise(self, context_latents: torch.FloatTensor, seed: Union[int, List[int], None] = None):
        """
        Prepare noise tensor for generation with optional seeding.

        Args:
            context_latents: Context latents to determine noise shape
            seed: Can be int, List[int], or None. If None, uses random noise.

        Returns:
            Noise tensor of appropriate shape
        """
        bsz = context_latents.shape[0]
        device = context_latents.device
        dtype = context_latents.dtype
        # Handle seed: can be int, List[int], or None
        src_latents_shape = (context_latents.shape[0], context_latents.shape[1], context_latents.shape[-1] // 2)
        if seed is None:
            # No seed provided - use random
            noise = torch.randn(src_latents_shape, device=device, dtype=dtype)
        elif isinstance(seed, list):
            # List of seeds - generate noise for each sample separately
            noise_list = []
            for i, s in enumerate(seed):
                if s is None or s < 0:
                    # Random seed for this sample
                    noise_i = torch.randn(1, src_latents_shape[1], src_latents_shape[2], device=device, dtype=dtype)
                else:
                    # Use specific seed for this sample
                    generator = torch.Generator(device=device).manual_seed(int(s))
                    noise_i = torch.randn(1, src_latents_shape[1], src_latents_shape[2], generator=generator, device=device, dtype=dtype)
                noise_list.append(noise_i)
            noise = torch.cat(noise_list, dim=0)
        else:
            # Single seed for all samples
            generator = torch.Generator(device=device).manual_seed(int(seed))
            noise = torch.randn(src_latents_shape, generator=generator, device=device, dtype=dtype)

        return noise

    def get_x0_from_noise(self, zt, vt, t):
        return zt - vt * t.unsqueeze(-1).unsqueeze(-1)

    def renoise(self, x, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        if isinstance(t, torch.Tensor) and t.ndim != x.ndim:
            t = t.unsqueeze(-1).unsqueeze(-1)
        xt = t * noise + (1 - t) * x
        return xt
    
    def generate_audio(
        self,
        text_hidden_states: torch.FloatTensor,
        text_attention_mask: torch.FloatTensor,
        lyric_hidden_states: torch.FloatTensor,
        lyric_attention_mask: torch.FloatTensor,
        refer_audio_acoustic_hidden_states_packed: torch.FloatTensor,
        refer_audio_order_mask: torch.LongTensor,
        src_latents: torch.FloatTensor,
        chunk_masks: torch.FloatTensor,
        is_covers: torch.Tensor,
        silence_latent: Optional[torch.FloatTensor] = None,
        attention_mask: torch.Tensor = None,
        seed: int = None,
        fix_nfe: int = 8,
        infer_method: str = "ode",
        use_cache: bool = True,
        audio_cover_strength: float = 1.0,
        non_cover_text_hidden_states: Optional[torch.FloatTensor] = None,
        non_cover_text_attention_mask: Optional[torch.FloatTensor] = None,
        precomputed_lm_hints_25Hz: Optional[torch.FloatTensor] = None,
        audio_codes: Optional[torch.FloatTensor] = None,
        shift: float = 3.0,
        timesteps: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Valid shifts: only discrete values 1, 2, 3 are supported
        VALID_SHIFTS = [1.0, 2.0, 3.0]
        
        # Valid timesteps: all unique timesteps from shift=1,2,3 with fix_nfe=8 (total 20 values)
        VALID_TIMESTEPS = [
            1.0, 0.9545454545454546, 0.9333333333333333, 0.9, 0.875, 
            0.8571428571428571, 0.8333333333333334, 0.7692307692307693, 0.75, 
            0.6666666666666666, 0.6428571428571429, 0.625, 0.5454545454545454, 
            0.5, 0.4, 0.375, 0.3, 0.25, 0.2222222222222222, 0.125
        ]
        
        # Pre-defined timestep schedules for each valid shift (fix_nfe=8, excluding final 0)
        SHIFT_TIMESTEPS = {
            1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
            2.0: [1.0, 0.9333333333333333, 0.8571428571428571, 0.7692307692307693, 0.6666666666666666, 0.5454545454545454, 0.4, 0.2222222222222222],
            3.0: [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3],
        }
        
        # Determine the timestep schedule to use
        t_schedule_list = None
        
        if timesteps is not None:
            # Process custom timesteps: map each value to nearest valid timestep
            timesteps_list = timesteps.tolist() if isinstance(timesteps, torch.Tensor) else list(timesteps)
            
            # Remove trailing zeros
            while len(timesteps_list) > 0 and timesteps_list[-1] == 0:
                timesteps_list.pop()
            
            # Validate length: 1-20
            if len(timesteps_list) < 1:
                logger.warning(f"timesteps length is too short after removing trailing zeros, using default shift={shift}")
            elif len(timesteps_list) > 20:
                logger.warning(f"timesteps length={len(timesteps_list)} exceeds maximum 20, truncating to 20")
                timesteps_list = timesteps_list[:20]
                t_schedule_list = timesteps_list
            else:
                t_schedule_list = timesteps_list
            
            if t_schedule_list is not None:
                # Map each timestep to nearest valid timestep
                original_timesteps = t_schedule_list.copy()
                mapped_timesteps = []
                for t in t_schedule_list:
                    nearest = min(VALID_TIMESTEPS, key=lambda x: abs(x - t))
                    mapped_timesteps.append(nearest)
                
                if original_timesteps != mapped_timesteps:
                    logger.warning(f"timesteps mapped to nearest valid values: {original_timesteps} -> {mapped_timesteps}")
                
                t_schedule_list = mapped_timesteps
        
        if t_schedule_list is None:
            # Use shift-based schedule: round to nearest valid shift
            original_shift = shift
            shift = min(VALID_SHIFTS, key=lambda x: abs(x - shift))
            if original_shift != shift:
                logger.warning(f"shift={original_shift} not supported, rounded to nearest valid shift={shift}")
            t_schedule_list = SHIFT_TIMESTEPS[shift]
        
        if attention_mask is None:
            latent_length = src_latents.shape[1]
            attention_mask = torch.ones(src_latents.shape[0], latent_length, device=src_latents.device, dtype=src_latents.dtype)
        
        time_costs = {}
        start_time = time.time()
        total_start_time = start_time
        encoder_hidden_states, encoder_attention_mask, context_latents = self.prepare_condition(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask=refer_audio_order_mask,
            hidden_states=src_latents,
            attention_mask=attention_mask,
            silence_latent=silence_latent,
            src_latents=src_latents,
            chunk_masks=chunk_masks,
            is_covers=is_covers,
            precomputed_lm_hints_25Hz=precomputed_lm_hints_25Hz,
            audio_codes=audio_codes,
        )
        
        encoder_hidden_states_non_cover, encoder_attention_mask_non_cover, context_latents_non_cover = None, None, None
        if audio_cover_strength < 1.0:
            non_is_covers = torch.zeros_like(is_covers, device=is_covers.device, dtype=is_covers.dtype)
            # Use silence_latent for non-cover condition to simulate text2music mode (no reference audio)
            silence_latent_expanded = silence_latent[:, :src_latents.shape[1], :].expand(src_latents.shape[0], -1, -1)
            encoder_hidden_states_non_cover, encoder_attention_mask_non_cover, context_latents_non_cover = self.prepare_condition(
            text_hidden_states=non_cover_text_hidden_states,
            text_attention_mask=non_cover_text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask=refer_audio_order_mask,
            hidden_states=silence_latent_expanded,
            attention_mask=attention_mask,
            silence_latent=silence_latent,
            src_latents=silence_latent_expanded,
            chunk_masks=chunk_masks,
            is_covers=non_is_covers,
            precomputed_lm_hints_25Hz=None,
            audio_codes=None,
        )
        
        end_time = time.time()
        time_costs["encoder_time_cost"] = end_time - start_time
        start_time = end_time
        
        noise = self.prepare_noise(context_latents, seed)
        bsz, device, dtype = context_latents.shape[0], context_latents.device, context_latents.dtype
        past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
        
        # Use pre-computed t_schedule_list (already validated and mapped to valid timesteps)
        t_schedule = torch.tensor(t_schedule_list, device=device, dtype=dtype)
        num_steps = len(t_schedule)
        
        # Recalculate cover_steps based on actual num_steps
        cover_steps = int(num_steps * audio_cover_strength)
        
        xt = noise
        for step_idx in range(num_steps):
            current_timestep = t_schedule[step_idx].item()
            t_curr_tensor = current_timestep * torch.ones((bsz,), device=device, dtype=dtype)
            
            if step_idx >= cover_steps:
                encoder_hidden_states = encoder_hidden_states_non_cover
                encoder_attention_mask = encoder_attention_mask_non_cover
                context_latents = context_latents_non_cover
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
            
            with torch.no_grad():        
                decoder_outputs = self.decoder(
                    hidden_states=xt,
                    timestep=t_curr_tensor,
                    timestep_r=t_curr_tensor,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    context_latents=context_latents,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                
            vt = decoder_outputs[0]
            past_key_values = decoder_outputs[1]
            
            # On final step, directly compute x0 from noise
            if step_idx == num_steps - 1:
                xt = self.get_x0_from_noise(xt, vt, t_curr_tensor)
                break
            
            # Update x_t based on inference method
            if infer_method == "sde":
                # Stochastic Differential Equation: predict clean, then re-add noise
                pred_clean = self.get_x0_from_noise(xt, vt, t_curr_tensor)
                next_timestep = t_schedule[step_idx + 1].item()
                xt = self.renoise(pred_clean, next_timestep)
            elif infer_method == "ode":
                # Ordinary Differential Equation: Euler method
                # dx/dt = -v, so x_{t+1} = x_t - v_t * dt
                next_timestep = t_schedule[step_idx + 1].item()
                dt = current_timestep - next_timestep
                dt_tensor = dt * torch.ones((bsz,), device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1)
                xt = xt - vt * dt_tensor
        
        x_gen = xt
        end_time = time.time()
        time_costs["diffusion_time_cost"] = end_time - start_time
        time_costs["diffusion_per_step_time_cost"] = time_costs["diffusion_time_cost"] / num_steps
        time_costs["total_time_cost"] = end_time - total_start_time
        return {
            "target_latents": x_gen,
            "time_costs": time_costs,
        }


def test_forward(model, seed=42):
    # Fix random seed for reproducibility
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Get model dtype and device
    model_dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    
    print(f"Testing with dtype: {model_dtype}, device: {device}, seed: {seed}")
    
    # Test data preparation with matching dtype
    text_hidden_states = torch.randn(2, 77, 1024, dtype=model_dtype, device=device)
    text_attention_mask = torch.ones(2, 77, dtype=model_dtype, device=device)
    lyric_hidden_states = torch.randn(2, 123, 1024, dtype=model_dtype, device=device)
    lyric_attention_mask = torch.ones(2, 123, dtype=model_dtype, device=device)
    refer_audio_acoustic_hidden_states_packed = torch.randn(3, 750, 64, dtype=model_dtype, device=device)
    refer_audio_order_mask = torch.LongTensor([0, 0, 1]).to(device)

    # Base config: 25 Hz hidden states → 10 s = 250 frames (round to int)
    base_seconds = 10
    frames_per_second = 25
    base_seq_len = base_seconds * frames_per_second

    hidden_states = torch.randn(2, base_seq_len, 64, dtype=model_dtype, device=device)
    attention_mask = torch.ones(2, base_seq_len, dtype=model_dtype, device=device)
    # Add some padding to test mask behavior
    pad_start = max(base_seq_len // 2, 1)
    attention_mask[0, pad_start:] = 0
    chunk_mask = torch.ones(2, base_seq_len, 64, dtype=model_dtype, device=device)
    chunk_mask[0, pad_start:] = 0

    silence_latent = torch.randn(2, base_seq_len, 64, dtype=model_dtype, device=device)
    # New required parameters for updated training logic
    src_latents = torch.randn(2, base_seq_len, 64, dtype=model_dtype, device=device)  # Source latents for context
    is_covers = torch.tensor([0, 1], dtype=torch.long, device=device)  # Cover song indicators (0=original, 1=cover)

    # Test 1: Flow matching training (using 10s sequence for sanity check by default)
    print(f"Testing flow matching training with {base_seconds}s sequence ({base_seq_len} frames @ {frames_per_second}Hz)...")
    outputs = model.training_losses(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        chunk_masks=chunk_mask,
        text_hidden_states=text_hidden_states,
        text_attention_mask=text_attention_mask,
        lyric_hidden_states=lyric_hidden_states,
        lyric_attention_mask=lyric_attention_mask,
        refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
        refer_audio_order_mask=refer_audio_order_mask,
        silence_latent=silence_latent,
        src_latents=src_latents,
        is_covers=is_covers,
        cfg_ratio=0.15,
    )
    loss = outputs['diffusion_loss']
    print(f"Flow matching loss: {loss.item():.6f}")
    print(f"  Loss stats - min: {loss.min().item():.6f}, max: {loss.max().item():.6f}, mean: {loss.mean().item():.6f}, std: {loss.std().item() if loss.numel() > 1 else 0:.6f}")

    # Test 2: Generation with flow matching, testing throughput for different sequence lengths
    lengths_seconds = [10, 30, 60, 120, 180, 240]
    infer_steps = 2  # Can be increased as needed (e.g., 50/100) to better approximate real inference

    print("\n===== Throughput benchmark (25Hz hidden states) =====")
    for seconds in lengths_seconds:
        seq_len = seconds * frames_per_second

        # Reconstruct inputs for current sequence length
        cur_hidden_states = torch.randn(2, seq_len, 64, dtype=model_dtype, device=device)
        cur_attention_mask = torch.ones(2, seq_len, dtype=model_dtype, device=device)
        cur_chunk_mask = torch.ones(2, seq_len, 64, dtype=model_dtype, device=device)
        cur_silence_latent = torch.randn(2, seq_len, 64, dtype=model_dtype, device=device)
        cur_src_latents = torch.randn(2, seq_len, 64, dtype=model_dtype, device=device)

        print(f"\n--- {seconds}s input ({seq_len} frames @ {frames_per_second}Hz) ---")
        outputs = model.generate_audio(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask=refer_audio_order_mask,
            src_latents=cur_src_latents,
            chunk_masks=cur_chunk_mask,
            silence_latent=cur_silence_latent,
            infer_steps=infer_steps,
            is_covers=is_covers,
            seed=1234,
        )

        target_latents = outputs["target_latents"]
        time_costs = outputs.get("time_costs", {})

        total_time = time_costs.get("total_time_cost", None)
        diffusion_time = time_costs.get("diffusion_time_cost", None)

        # Output shape and statistics
        print(f"Generated latents shape: {target_latents.shape}")
        print(
            f"Stats - min: {target_latents.min().item():.4f}, "
            f"max: {target_latents.max().item():.4f}, "
            f"mean: {target_latents.mean().item():.4f}, "
            f"std: {target_latents.std().item():.4f}"
        )

        # Calculate throughput: statistics by frame count and audio seconds
        bsz, t_len = target_latents.shape[0], target_latents.shape[1]
        audio_seconds = t_len / frames_per_second

        if total_time is not None:
            frames_throughput = (bsz * t_len) / total_time
            seconds_throughput = (bsz * audio_seconds) / total_time
            print(
                f"Time costs: total={total_time:.4f}s, diffusion={diffusion_time:.4f}s "
                f"({infer_steps} steps)"
                if diffusion_time is not None
                else f"Time costs: total={total_time:.4f}s"
            )
            print(
                f"Throughput (based on total_time): "
                f"{frames_throughput:.2f} frames/s, "
                f"{seconds_throughput:.2f} audio-seconds/s (batch={bsz})"
            )
        else:
            print("Time costs not available in outputs['time_costs']; only basic stats printed.")


if __name__ == "__main__":
    from torch.profiler import profile, record_function, ProfilerActivity
    import os, torch
    import time
    from transformers import AutoModel
    config = AceStepConfig()
    start = time.time()
    import os
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model = AceStepConditionGenerationModel.from_pretrained(model_dir)
    end = time.time()
    # model.config._attn_implementation = "sdpa"
    model.config._attn_implementation = "flash_attention_2"
    model.eval()
    # model = model.to("cpu")
    # model = model.float()
    model = model.to("cuda")
    model = model.bfloat16()
    test_forward(model)

# Wrapper class for ModelPool compatibility
class AceStepConditionGenerationModelWrapper(torch.nn.Module):
    """
    Wrapper for AceStepConditionGenerationModel to make it compatible with ModelPool.
    
    ModelPool expects models to accept **kwargs in __init__, but PreTrainedModel
    subclasses require a config object as the first positional argument.
    This wrapper handles the conversion.
    """
    def __init__(self, config_path=None, **config_kwargs):
        super().__init__()
        if config_path is not None:
            # Load from pretrained config
            config = AceStepConfig.from_pretrained(config_path)
        elif config_kwargs:
            # Create config from kwargs
            config = AceStepConfig(**config_kwargs)
        else:
            # Use default config
            config = AceStepConfig()
        
        self.model = AceStepConditionGenerationModel(config)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def __getattr__(self, name):
        # Delegate attribute access to the wrapped model
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
