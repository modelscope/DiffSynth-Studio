# Copyright (c) Ant Group. All rights reserved.
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

import math, warnings, os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import (
    MoeModelOutputWithPast,
    MoeCausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    logging,
    replace_return_docstrings,
)

"""Bailing MoE model configuration"""

class BailingMoeV2Config(PretrainedConfig):
    model_type = "bailing_moe_v2"
    def __init__(
        self,
        vocab_size=30592,
        hidden_size=1024,
        intermediate_size=None,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=0,
        hidden_act="silu",
        use_qkv_bias=False,  # bailing only
        use_qk_norm=False,
        use_bias=True,  # bailing only
        rms_norm_eps=1e-05,
        norm_head=False,  # bailing only
        tie_word_embeddings=False,  # PretrainedConfig key, here change default value.
        embedding_dropout=0.1,
        attention_dropout=0.1,
        output_dropout=0.1,
        initializer_range=0.02,
        max_position_embeddings=16384,
        rope_theta=10000.0,
        use_cache=True,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        rope_scaling=None,
        pad_token_id=126081,
        num_experts=16,
        num_shared_experts=0,
        num_experts_per_tok=2,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        moe_intermediate_size=None,
        first_k_dense_replace=0,
        head_dim=None,
        output_router_logits=False,
        partial_rotary_factor=0.5,
        router_type="topN",
        _attn_implementation="flash_attention_2",
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.use_qkv_bias = use_qkv_bias
        self.use_bias = use_bias
        self.norm_head = norm_head
        self.rms_norm_eps = rms_norm_eps
        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.head_dim = head_dim or self.hidden_size // self.num_attention_heads
        self.rope_scaling = rope_scaling
        # MoE configs
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.output_router_logits = output_router_logits
        self.routed_scaling_factor = routed_scaling_factor
        self.partial_rotary_factor = partial_rotary_factor
        self.router_type = router_type
        super().__init__(pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)
        self._attn_implementation = _attn_implementation





logger = logging.get_logger(__name__)

class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

def patch_continuous_features(
    input_embeddings: torch.Tensor,
    placeholder_loc_lens: torch.Tensor,
    encoded_feats: torch.Tensor,
    encoded_feat_lens: torch.Tensor,
):
    """
    Patch continuous features into input embeddings, while keeping a valid gradient flow.

    input_embeddings: torch.Tensor, size = [B, C?, T, D]
    placeholder_loc_lens: torch.LongTensor, size = [B, N, 2]
        Each 2-tuple represents (start, length) of a placeholder.
    encoded_feats: torch.Tensor, size = [B, L1 + L2 + ... + LN, ...]
    encoded_feat_lens: torch.LongTensor, size = [B, N]

    Example ('X' for patch placeholder tokens):
    Inputs:
        input_embeddings = [[1, 2, 3, X, X, X, 4, 5, 6, X, X, X, 7, 8]]
        placeholder_loc_lens = [[[3, 3]], [[9, 3]]]
        encoded_feats = [[A, A, A, B, B]]
        encoded_feat_lens = [[3], [2]]
    Outputs:
        embeddings = [[1, 2, 3, A, A, A, 4, 5, 6, B, B, X, 7, 8]]
    """
    batch_size = input_embeddings.size(0)
    for i in range(batch_size):
        audio_feat_start = 0
        for j in range(placeholder_loc_lens.shape[1]):
            placeholder_start: int = int(placeholder_loc_lens[i, j, 0].item())
            placeholder_len: int = int(placeholder_loc_lens[i, j, 1].item())
            if placeholder_len <= 0:
                break
            feat_len = int(encoded_feat_lens[i, j].item())
            real_feat_len = feat_len
            if feat_len > placeholder_len:
                logger.warning(
                    f"Feature length ({feat_len}) > placeholder length ({placeholder_len}). "
                    "This is not expected; please check estimate_audio_feature_length(). "
                    "We truncate the feature to avoid errors."
                )
                feat_len = placeholder_len
            target_len = min(feat_len, placeholder_len)
            input_embeddings[i, placeholder_start:placeholder_start + target_len] = encoded_feats[i, audio_feat_start:audio_feat_start + target_len]
            audio_feat_start += real_feat_len
    return input_embeddings

def build_modality_mask(placeholder_loc_lens: torch.Tensor, shape: torch.Size):
    mask = torch.zeros(shape, dtype=torch.bool)
    for i in range(placeholder_loc_lens.shape[0]):
        for j in range(placeholder_loc_lens.shape[1]):
            start: int = int(placeholder_loc_lens[i, j, 0].item())
            length: int = int(placeholder_loc_lens[i, j, 1].item())
            if length <= 0:
                break
            mask[i, start:start + length] = True
    return mask

def encode_audio_segments(
    encoder,
    proj_layer,
    wav_feats=None,
    wav_feats_lengths=None,
    waveforms=None,
    waveforms_lengths=None,
    use_waveform=False,
    audio_config=None,
):
    """
    Apply audio encoder to input audio features in wrapped format.
    See the documentation of unwrap_feats() for details about 'wrapped format'.
    """

    # Forward audio encoder.
    if use_waveform:
        assert waveforms is not None and waveforms_lengths is not None
        # Unwrap the waveforms so each waveform is placed at an independent row.
        waveform_segs_batch, waveform_seg_lengths = unwrap_feats(waveforms, waveforms_lengths)
        audio_feats_seg, audio_feat_seg_lengths = encoder(waveform_segs_batch, waveform_seg_lengths)[:2]
    else:
        assert wav_feats is not None and wav_feats_lengths is not None
        # Unwrap the features so the feature of each waveform is placed at an independent row.
        feat_segs_batch, feat_seg_lengths = unwrap_feats(wav_feats, wav_feats_lengths)
        # for whisper encoder
        # feat_segs_batch: [B, T, n_mels]
        # feat_seg_lengths: [B]
        audio_feats_seg = encoder(feat_segs_batch)
        audio_feats_seg_proj = proj_layer(audio_feats_seg.transpose(-1, -2)).transpose(-1, -2)
        feat_seg_lengths = feat_seg_lengths.to(feat_segs_batch.device)
        # whisper encoder conv
        audio_feat_seg_lengths = (feat_seg_lengths - 3 + 2 * 1) // 2 + 1
        # project layer conv
        audio_feat_seg_lengths = (audio_feat_seg_lengths - audio_config.ds_kernel_size + 2 *
                                  (audio_config.ds_kernel_size//2)) // audio_config.ds_stride + 1

    # Wrap the features so the 1st dim represents batch_size.
    input_lengths = waveforms_lengths if use_waveform else wav_feats_lengths
    assert input_lengths is not None
    audio_feats, _, audio_feats_lengths = wrap_feats(audio_feats_seg, input_lengths, audio_feat_seg_lengths)
    audio_feats_proj, _, audio_feats_lengths2 = wrap_feats(audio_feats_seg_proj, input_lengths, audio_feat_seg_lengths)
    assert torch.all(audio_feats_lengths == audio_feats_lengths2), f"{audio_feats_lengths}, {audio_feats_lengths2}"

    return audio_feats_proj, audio_feats, audio_feats_lengths

def unwrap_feats(feats: torch.Tensor, feats_lengths: torch.Tensor):
    """
    The input feats are in the "wrapped" format, which means that features from (at most) N audios are concatenated
    as a single sample feats[i]. In this case, each row of feats_lengths contains the lengths of the concatenated
    feature. This function unwraps the features.
    For samples with less than N segments, one should pad feats_lengths with 0. The result will contain valid
    segments only.

    feats: torch.Tensor, size = [B, L1 + L2 + ... + LN, ...]
    feats_lengths: torch.LongTensor, size = [B, N]

    Example ('X' for padding):
    Inputs:
        feats = [[A, A, A, A, X],
                 [B, B, C, C, C]]
        feats_lengths = [[4, 0],
                         [2, 3]]
    Outputs:
        feat_segs = [[A, A, A, A],
                     [B, B, X, X],
                     [C, C, C, X]]
        feat_seg_lengths = [4, 2, 3]
    """
    feat_segs = []
    feat_seg_lengths = []
    for i in range(feats_lengths.shape[0]):
        feat_index = 0
        for j in range(feats_lengths.shape[1]):
            feat_len = feats_lengths[i, j].item()
            if feat_len == 0: break
            feat_segs.append(feats[i, feat_index:feat_index + feat_len])
            feat_seg_lengths.append(feat_len)
            feat_index += feat_len
    feat_segs_batch = torch.nn.utils.rnn.pad_sequence(feat_segs, True).to(feats.device)
    feat_seg_lengths = torch.tensor(feat_seg_lengths, dtype=torch.long, device=feats.device)
    return feat_segs_batch, feat_seg_lengths

def wrap_feats(feat_segs: torch.Tensor, feats_lengths: torch.Tensor, feats_seg_lengths: Optional[torch.Tensor] = None):
    """
    Wrap segmented features back to the wrapped format.
    This function is the inverse operation of unwrap_feats(). See its documentation for details.
    Note that the feats_lengths value does not matter a lot. We only check the location of the first 0 to determine the
    number of feature segments.
    """
    feat_idx = 0
    feats_buffer = []
    feats_locs_buffer = []
    feats_lengths_buffer = []
    for i in range(feats_lengths.shape[0]):
        feat_buffer = []
        feat_locs_buffer = []
        feat_lengths_buffer = []
        feat_total_len = 0
        for j in range(feats_lengths.shape[1]):
            feat_len = feats_lengths[i, j].item()
            if feat_len == 0:
                break
            if feats_seg_lengths is not None:
                feat_len = feats_seg_lengths[feat_idx].item()
            feat_buffer.append(feat_segs[feat_idx, :feat_len])
            feat_locs_buffer.append(feat_total_len)
            feat_lengths_buffer.append(feat_len)
            feat_idx += 1
            feat_total_len += feat_len
        feats_buffer.append(torch.cat(feat_buffer))
        feats_locs_buffer.append(torch.tensor(feat_locs_buffer, dtype=torch.long))
        feats_lengths_buffer.append(torch.tensor(feat_lengths_buffer, dtype=torch.long))
    feats = torch.nn.utils.rnn.pad_sequence(feats_buffer, True).to(feat_segs.device)
    feats_locs = torch.nn.utils.rnn.pad_sequence(feats_locs_buffer, True).to(feats_lengths.device)
    feats_new_lengths = torch.nn.utils.rnn.pad_sequence(feats_lengths_buffer, True).to(feats_lengths.device)
    return feats, feats_locs, feats_new_lengths


# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch Qwen2_5_ViT model."""






try:
    import transformer_engine.pytorch as te
except (ImportError, OSError):
    # Transformer Engine 1.11 reads flash-attn package metadata at import
    # time, so flash-attn-free environments cannot import it at all. The
    # vision tower only needs te.RMSNorm; a native fallback is defined below.
    te = None

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb

else:
    flash_attn_varlen_func = None
    apply_rotary_emb = None

logger = logging.get_logger(__name__)

class Qwen2_5_VLVisionConfig(PretrainedConfig):
    model_type = "qwen2_5_vit"

    def __init__(
        self,
        depth=32,
        hidden_size=3584,
        hidden_act="silu",
        intermediate_size=3420,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokens_per_second=4,
        window_size=112,
        out_hidden_size=3584,
        fullatt_block_indexes=[7, 15, 23, 31],
        _attn_implementation="flash_attention_2",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.out_hidden_size = out_hidden_size
        self._attn_implementation = _attn_implementation

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if 'vision_config' in config_dict:
            config_dict = config_dict['vision_config']

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)

class Qwen2_5_VLMLP(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

class Qwen2_5_VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states).view(-1, self.embed_dim)
        return hidden_states

class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.theta = theta
        self.dim = dim

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

    def reset_parameters(self):
        # recompute inv_freq (for dynamic adjustment)
        new_inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))
        self.inv_freq.copy_(new_inv_freq)

if te is not None:
    _RMSNormImpl = te.RMSNorm
else:

    class _RMSNormImpl(nn.Module):
        """Native RMSNorm used when Transformer Engine is unavailable."""

        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.float()
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return (self.weight * hidden_states).to(input_dtype)


class Qwen2RMSNorm(_RMSNormImpl):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__(hidden_size, eps=eps)

class Qwen2_5_VLPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x

def apply_rotary_pos_emb_flashatt(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    tensor_ = tensor.float()
    cos = freqs.cos().float()
    sin = freqs.sin().float()
    output = apply_rotary_emb(tensor_, cos, sin).type_as(tensor)
    return output

def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output

class Qwen2_5_VLVisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.full(
            [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]: cu_seqlens[i], cu_seqlens[i - 1]: cu_seqlens[i]] = 0

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output

class Qwen2_5_VLVisionSdpaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]: cu_seqlens[i], cu_seqlens[i - 1]: cu_seqlens[i]] = True
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output

QWEN2_5_VL_VISION_ATTENTION_CLASSES = {
    "eager": Qwen2_5_VLVisionAttention,
    "sdpa": Qwen2_5_VLVisionSdpaAttention,
}

class Qwen2_5_VLVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = QWEN2_5_VL_VISION_ATTENTION_CLASSES[attn_implementation](
            config.hidden_size, num_heads=config.num_heads
        )
        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

class Qwen2_5_VisionTransformer(PreTrainedModel):
    config_class = Qwen2_5_VLVisionConfig
    _no_split_modules = ["Qwen2_5_VLVisionBlock"]
    _supports_sdpa = True

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Qwen2_5_VLVisionBlock(config, config._attn_implementation) for _ in range(config.depth)]
        )
        self.merger = Qwen2_5_VLPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.gradient_checkpointing = False
        self.image_emb_dim = 8192

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, is_list=False, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        if is_list:
            image_grid_thw_array = []
            for image_grid_thw_ in grid_thw:
                image_grid_thw_array.extend(image_grid_thw_)
            grid_thw = torch.tensor(image_grid_thw_array, device=hidden_states.device)

        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens_now, rotary_pos_emb
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens_now,
                    rotary_pos_emb=rotary_pos_emb,
                )

        hidden_states = self.merger(hidden_states)

        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states


# coding=utf-8
# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch BailingMoE model."""


try:
    # Imported for parity with the training environment only; nothing in this
    # file references `te`. Transformer Engine 1.11 reads flash-attn package
    # metadata at import time, so tolerate it being unavailable in
    # flash-attn-free installs.
    import transformer_engine.pytorch as te  # noqa: F401
except (ImportError, OSError):
    te = None



logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BailingMoeV2Config"


def _rope_init_default(config, device=None):
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    return inv_freq, 1.0


def _causal_4d_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length=0):
    bsz, q_len = input_shape
    dtype, device = inputs_embeds.dtype, inputs_embeds.device
    min_dtype = torch.finfo(dtype).min
    kv_len = q_len + past_key_values_length
    mask = torch.full((q_len, kv_len), min_dtype, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1 + past_key_values_length)
    mask = mask[None, None, :, :].expand(bsz, 1, q_len, kv_len)
    if attention_mask is not None:
        mask = mask + (1.0 - attention_mask[:, None, None, :].to(dtype)) * min_dtype
    return mask


class BailingMoeV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        BailingMoeV2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def reset_parameters(self):
        nn.init.ones_(self.weight)  # explicit reset for enabling fsdp


class BailingMoeV2RotaryEmbedding(nn.Module):
    def __init__(self, config: BailingMoeV2Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = _rope_init_default

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        position_ids = position_ids[:, 0, :] 
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()


        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def reset_parameters(self) -> None:
        new_inv_freq, self.attention_scaling = self.rope_init_fn(self.config, None)
        self.inv_freq.copy_(new_inv_freq)
    
    
class BailingMoeV2RotaryEmbedding3D(nn.Module):
    def __init__(self, config: BailingMoeV2Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        self.rope_init_type = "default"
        self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = _rope_init_default

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        self.config = config

    @torch.no_grad()
    def forward(self, x, position_ids):
        if self.rope_type == "3D" or self.rope_type == "video_rope":
            inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
            position_ids_expanded = position_ids[:, :, None, :].float()
        else:
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
            position_ids_expanded = position_ids[:, None, :].float()

        if self.rope_type == "3D" or self.rope_type == "video_rope":
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        else:
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def reset_parameters(self) -> None:
        new_inv_freq, self.attention_scaling = self.rope_init_fn(self.config, None)
        self.inv_freq.copy_(new_inv_freq)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_3d_rotary_pos_emb(
    q, 
    k, 
    cos, 
    sin, 
    mrope_section=[8, 12, 12], 
    unsqueeze_dim=1, 
    rope_type="m_rope", 
    rotary_half=True
):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).
    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        rope_type (`str`, *optional*, defaults to "m_rope"):
        rotary_half (`bool`, *optional*, defaults to `False`): Keep half or full tensor for later concatenation
    Returns:
        `tuple(torch.Tensor)` comprising the query and key tensors rotated using the Rotary Position Embedding.
    """
    if rope_type == "3D":  # rename rope_type
        rope_type = "m_rope"

    if rope_type == "m_rope":
        mrope_section = mrope_section * 2
        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim).to(q.device)
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim).to(q.device)
    elif rope_type == "video_rope":
        mrope_section = list(mrope_section)
        mrope_section = [mrope_section[0], mrope_section[1] + mrope_section[2]]
        mrope_section = mrope_section * 2
        # adjust t last -> (48, 16, 48, 16)
        mrope_section = mrope_section[::-1]
        index = 0
        result_cos = []
        result_sin = []
        # get x1, y1, x2, y2, ..., t1, t2, ...
        for i, section in enumerate(mrope_section):
            if i % 2 == 0:
                for j in range(section):
                    row = 1 if j % 2 == 0 else 2
                    result_cos.append(cos[row, ..., index: index + 1])
                    result_sin.append(sin[row, ..., index: index + 1])
                    index += 1
            else:
                result_cos.append(cos[0, ..., index:index + section])
                result_sin.append(sin[0, ..., index:index + section])
                index += section
        cos, sin = torch.cat(result_cos, dim=-1).unsqueeze(dim=unsqueeze_dim).to(q.device), torch.cat(result_sin, dim=-1).unsqueeze(dim=unsqueeze_dim).to(q.device)
    else:  # vanilla rope for llm
        cos = cos.unsqueeze(unsqueeze_dim).to(q.device)
        sin = sin.unsqueeze(unsqueeze_dim).to(q.device)

    if rotary_half:
        rotary_dim = cos.shape[-1]
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

        # Apply rotary embeddings on the first half
        q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

        # Concatenate back to full shape
        q_embed = torch.cat([q_embed, q_pass], dim=-1)
        k_embed = torch.cat([k_embed, k_pass], dim=-1)
    else:
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def get_t_scale_rope_index(
    config,
    input_ids: torch.LongTensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    scale_factor: float = 1.0,
    second_per_grid_ts: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    spatial_merge_size = config.spatial_merge_size
    image_token_id = config.image_patch_token
    video_token_id = config.video_patch_token
    image_start_token_id = config.image_start_token
    video_start_token_id = config.video_start_token
    use_abs_time_pos = second_per_grid_ts is not None

    mrope_position_deltas = []
    if image_grid_thw is not None or video_grid_thw is not None:
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)

        for i, input_ids in enumerate(total_input_ids):
            if attention_mask is not None:
                input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            if image_grid_thw is not None:
                vision_start_indices = torch.argwhere(input_ids == image_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
            if video_grid_thw is not None:
                vision_start_indices = torch.argwhere(input_ids == video_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                video_nums = (vision_tokens == video_token_id).sum()

            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1][0].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # body-diagonal symmetry
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(
                    -1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(
                    llm_grid_t, -1, llm_grid_w).flatten() - (llm_grid_h - 1) // 2
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(
                    llm_grid_t, llm_grid_h, -1).flatten() - (llm_grid_w - 1) // 2

                # time dim adjust step size
                if use_abs_time_pos:
                    t_index = t_index * second_per_grid_t * scale_factor
                else:
                    t_index = t_index * scale_factor
                t_index = t_index + text_len + st_idx

                h_index = h_index + t_index
                w_index = w_index + t_index
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]))
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                # next text token near last video token position = last t + 1
                st_idx = llm_pos_ids_list[-1][0].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            llm_positions = llm_positions.to(dtype=position_ids.dtype, device=position_ids.device)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions
            # generate first token = last t + 1
            mrope_position_deltas.append(llm_positions[0].max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

    return position_ids, mrope_position_deltas

def get_rope_index(
    config,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = config.spatial_merge_size
        image_token_id = config.image_patch_token
        video_token_id = config.video_patch_token
        image_start_token_id = config.image_start_token
        video_start_token_id = config.video_start_token

        use_abs_time_pos = second_per_grid_ts is not None

        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                if image_grid_thw is not None:
                    vision_start_indices = torch.argwhere(input_ids == image_start_token_id).squeeze(1)
                    vision_tokens = input_ids[vision_start_indices + 1]
                    image_nums = (vision_tokens == image_token_id).sum()
                if video_grid_thw is not None:
                    vision_start_indices = torch.argwhere(input_ids == video_start_token_id).squeeze(1)
                    vision_tokens = input_ids[vision_start_indices + 1]
                    video_nums = (vision_tokens == video_token_id).sum()

                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                    if use_abs_time_pos:
                        time_tensor = expanded_range * second_per_grid_t * config.tokens_per_second
                        time_tensor_long = time_tensor.long()
                    else:
                        time_tensor_long = expanded_range.long()
                    t_index = time_tensor_long.flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

        return position_ids, mrope_position_deltas


class BailingMoeV2MLP(nn.Module):
    def __init__(self, config: BailingMoeV2Config, intermediate_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, image_mask=None, audio_mask=None):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class BailingMoeV2Gate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts

        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.num_experts, self.gating_dim)))
        self.routed_scaling_factor = config.routed_scaling_factor
        self.bias_update_coeff = 0.001

        self.expert_bias = torch.nn.Parameter(torch.zeros(self.num_experts), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def update_bias(self, distribution: torch.Tensor):
        with torch.no_grad():
            delta_bias = (distribution.mean() - distribution).sign()
        self.expert_bias.data = self.expert_bias.data + self.bias_update_coeff * delta_bias

    def group_limited_topk(
        self,
        scores: torch.Tensor,
    ):
        num_tokens, _ = scores.size()
        # Organize the experts into groups
        group_scores = scores.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        # Mask the experts based on selection groups
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
            .reshape(num_tokens, -1)
        )

        masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
        probs, top_indices = torch.topk(masked_scores, k=self.top_k, dim=-1, sorted=False)

        return probs, top_indices

    def forward(self, hidden_states):
        # compute gating score
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))

        scores = torch.sigmoid(logits)

        scores_for_routing = scores + self.expert_bias
        _, topk_idx = self.group_limited_topk(scores_for_routing)

        scores = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)

        topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else scores
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight, logits


class BailingMoeV2SparseMoeBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config: BailingMoeV2Config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self._setup_experts()
        self.router_type = config.router_type
        if self.router_type == "topN":
            logger.info(f"use topN Router")
            self.gate = BailingMoeV2Gate(config)
        elif self.router_type == "MultiRouter":
            logger.info(f"use MultiRouter")
            self.gate = BailingMoeV2Gate(config)
            self.image_gate = BailingMoeV2Gate(config)
            self.audio_gate = BailingMoeV2Gate(config)
        if config.num_shared_experts is not None:
            self.shared_experts = BailingMoeV2MLP(
                config=config, intermediate_size=config.moe_intermediate_size * config.num_shared_experts
            )

    def _setup_experts(self):
        self.experts = nn.ModuleList(
            [
                BailingMoeV2MLP(config=self.config, intermediate_size=self.config.moe_intermediate_size)
                for _ in range(self.config.num_experts)
            ]
        )

    def forward(self, hidden_states, image_mask, audio_mask):
        identity = hidden_states
        bsz, seq_len, h = hidden_states.shape
        if self.router_type == "MultiRouter":
            if image_mask is not None:
                if len(image_mask.shape) == 3:
                    assert image_mask.shape[-1] == 1
                elif len(image_mask.shape) == 2:
                    assert image_mask.shape == hidden_states.shape[:2]
                    image_mask = image_mask.unsqueeze(-1)
                else:
                    raise ValueError(
                        "unexpected image mask shape: "
                        f"{tuple(image_mask.shape)}"
                    )
            if audio_mask is not None:
                if len(audio_mask.shape) == 3:
                    assert audio_mask.shape[-1] == 1
                elif len(audio_mask.shape) == 2:
                    assert audio_mask.shape == hidden_states.shape[:2]
                    audio_mask = audio_mask.unsqueeze(-1)
                else:
                    raise ValueError(
                        "unexpected audio mask shape: "
                        f"{tuple(audio_mask.shape)}"
                    )
            if image_mask is not None and audio_mask is not None:
                assert torch.logical_and(image_mask, audio_mask).sum() == 0
            
            image_topk_idx, image_topk_weight, image_router_logits = self.image_gate(hidden_states)
            audio_topk_idx, audio_topk_weight, audio_router_logits = self.audio_gate(hidden_states)
            topk_idx, topk_weight, router_logits = self.gate(hidden_states)

            if image_mask is not None:
                image_mask = image_mask.view(-1, 1)
                topk_idx = image_topk_idx * image_mask + topk_idx * torch.logical_not(image_mask)
                topk_weight = image_topk_weight * image_mask + topk_weight * torch.logical_not(image_mask)
                router_logits = image_router_logits * image_mask + router_logits * torch.logical_not(image_mask)
            if audio_mask is not None:
                audio_mask = audio_mask.view(-1, 1)
                audio_mask = audio_mask.to(router_logits.device)
                topk_idx = audio_topk_idx * audio_mask + topk_idx * torch.logical_not(audio_mask)
                topk_weight = audio_topk_weight * audio_mask + topk_weight * torch.logical_not(audio_mask)
                router_logits = audio_router_logits * audio_mask + router_logits * torch.logical_not(audio_mask)
        else:
            topk_idx, topk_weight, router_logits = self.gate(hidden_states)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(bsz, seq_len, h)
        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y, (router_logits.view(bsz, seq_len, -1), topk_idx.view(bsz, seq_len, -1))

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape
        tokens_per_expert = tokens_per_expert.cpu().numpy()
        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out.to(x.device))
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(x.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->BailingMoeV2
class BailingMoeV2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: BailingMoeV2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim or self.hidden_size // self.num_heads
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        self.rope_dim = int(self.head_dim * partial_rotary_factor)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.query_key_value = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.use_qkv_bias,
        )

        self.q_norm = BailingMoeV2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = BailingMoeV2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.dense = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.use_bias)

        if self.config.rope_scaling is not None:
            self.rope_scaling = {"type": "mrope", "mrope_section": [8, 12, 12]}
            self.mrope_section = self.rope_scaling["mrope_section"]

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if not self.training:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
            del qkv

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        kv_seq_len = key_states.shape[-2]
        cos, sin = position_embeddings
        query_states, key_states = apply_3d_rotary_pos_emb(
            query_states, 
            key_states, 
            cos, 
            sin, 
            mrope_section=self.rope_scaling["mrope_section"], 
            rope_type=self.config.rope_scaling["type"]
        )
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states / math.sqrt(self.head_dim), key_states.transpose(2, 3))

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.dense(attn_output)

        if not output_attentions:
            attn_weights = None
        
        if not self.training:
            del query_states, key_states, value_states

        return attn_output, attn_weights, past_key_value


ATTENTION_CLASSES = {
    "eager": BailingMoeV2Attention,
}


class BailingMoeV2DecoderLayer(nn.Module):
    def __init__(self, config: BailingMoeV2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.attention = ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = (
            BailingMoeV2SparseMoeBlock(config)
            if (config.num_experts is not None and layer_idx >= config.first_k_dense_replace)
            else BailingMoeV2MLP(config=config, intermediate_size=config.intermediate_size)
        )
        self.input_layernorm = BailingMoeV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BailingMoeV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*):
                cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, image_mask, audio_mask)
        #hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states.to(residual.device)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


BAILINGMOEV2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BailingMoeV2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare BailingMoeV2 Model outputting raw hidden-states without any specific head on top.",
    BAILINGMOEV2_START_DOCSTRING,
)
class BailingMoeV2PreTrainedModel(PreTrainedModel):
    config_class = BailingMoeV2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BailingMoeV2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        from transformers.integrations import is_deepspeed_zero3_enabled
        if is_deepspeed_zero3_enabled():
            return
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


BAILINGMOEV2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare BailingMoeV2 Model outputting raw hidden-states without any specific head on top.",
    BAILINGMOEV2_START_DOCSTRING,
)
class BailingMoeV2Model(BailingMoeV2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`BailingMoeV2DecoderLayer`]

    Args:
        config: BailingMoeV2Config
    """

    def __init__(self, config: BailingMoeV2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [BailingMoeV2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = BailingMoeV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.config.rope_scaling is not None:
            self.rotary_emb = BailingMoeV2RotaryEmbedding3D(config=config)
        else:
            self.rotary_emb = BailingMoeV2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        config.spatial_merge_size = 2 
        config.tokens_per_second = 2
        self.rope_deltas = None
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value
 
    def prompt_wrap_vision(self, input_ids, inputs_embeds, vision_embeds, vision_token_id):
        if vision_embeds is None or input_ids is None:
            return inputs_embeds

        if len(vision_embeds.shape) == 3:
            vision_embeds = vision_embeds.reshape(-1, vision_embeds.shape[-1])

        n_image_tokens = (input_ids == vision_token_id).sum().item()
        n_image_features = vision_embeds.shape[0]

        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        vision_mask = (input_ids == vision_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        
        image_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(vision_mask, image_embeds)

        return inputs_embeds

    def prompt_wrap_audio(self, input_ids, inputs_embeds, audio_embeds, audio_embeds_lengths, placeholder_audio_loc_lens):
        inputs_embeds = patch_continuous_features(
           input_embeddings=inputs_embeds, placeholder_loc_lens=placeholder_audio_loc_lens,
           encoded_feats=audio_embeds, encoded_feat_lens=audio_embeds_lengths,
        )
        router_mask_audio = build_modality_mask(placeholder_audio_loc_lens, inputs_embeds.shape[:-1])
        router_mask_audio = router_mask_audio.to(inputs_embeds.device)
        return inputs_embeds, router_mask_audio
     
    def prompt_wrap_navit(self, input_ids, config, query_embeds_image=None, query_embeds_video=None, query_embeds_audio=None,
        query_embeds_audio_lengths=None, placeholder_audio_loc_lens=None, target_embeds=None):
        inputs_embeds = self.word_embeddings(input_ids)
        vision_mask = None
        audio_mask = None
        if query_embeds_image is None and query_embeds_video is None and query_embeds_audio is None and target_embeds is None:
            return inputs_embeds, vision_mask, audio_mask

        if query_embeds_image is not None:
            inputs_embeds = self.prompt_wrap_vision(input_ids, inputs_embeds, query_embeds_image, config.image_patch_token)
        if query_embeds_video is not None:
            inputs_embeds = self.prompt_wrap_vision(input_ids, inputs_embeds, query_embeds_video, config.video_patch_token)

        image_mask = input_ids == config.image_patch_token
        video_mask = input_ids == config.video_patch_token
        vision_mask = (image_mask + video_mask) > 0
        vision_mask = vision_mask.unsqueeze(-1).to(input_ids.device)

        if query_embeds_audio is not None:
            inputs_embeds, audio_mask = self.prompt_wrap_audio(
                input_ids, inputs_embeds, query_embeds_audio, query_embeds_audio_lengths, placeholder_audio_loc_lens,
            )

        return inputs_embeds, vision_mask, audio_mask


    @add_start_docstrings_to_model_forward(BAILINGMOEV2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        query_embeds_image: Optional[torch.Tensor] = None,
        query_embeds_video: Optional[torch.Tensor] = None,
        query_embeds_audio: Optional[torch.Tensor] = None,
        query_embeds_audio_lengths: Optional[torch.Tensor] = None,
        placeholder_audio_loc_lens: Optional[torch.Tensor] = None,
        target_embeds: Optional[torch.Tensor] = None, 
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        image_grid_thw_video: Optional[torch.Tensor] = None, 
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        image_mask=None,
        audio_mask=None,
        **kwargs,
    ) -> Union[Tuple, MoeModelOutputWithPast]:

        if inputs_embeds is not None:
            words_embeddings = inputs_embeds
            input_shape = inputs_embeds.size()[:2]
        else:
            if (query_embeds_image is None and query_embeds_video is None and query_embeds_audio is None and target_embeds is None) or input_ids.size(1) == 1:
                words_embeddings = self.word_embeddings(input_ids.clip(0, self.word_embeddings.weight.shape[0] - 1))
                input_shape = input_ids.size()
                image_mask = None
                audio_mask = None
            else:
                words_embeddings, image_mask, audio_mask = self.prompt_wrap_navit(
                    input_ids.clip(0, self.word_embeddings.weight.shape[0] - 1), self.config, query_embeds_image, query_embeds_video, query_embeds_audio,
                    query_embeds_audio_lengths, placeholder_audio_loc_lens, target_embeds,  # noqa
                )

                input_shape = words_embeddings.size()[:2]
        embeddings = words_embeddings

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            assert input_ids.size(1) == inputs_embeds.size(1), "{} vs {}".format(
                input_ids.size,
                inputs_embeds.size,
            )
            batch_size, seq_length = inputs_embeds.shape[:2]
            #raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        past_key_values_length = 0

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if self.rotary_emb.rope_type == "video_rope" and input_ids.size(1) != 1:
            position_ids, rope_deltas = get_t_scale_rope_index(
                self.config,
                input_ids,
                image_grid_thw,
                image_grid_thw_video,
                attention_mask,
                scale_factor=2.0,
                second_per_grid_ts=second_per_grid_ts,
            )
            self.rope_deltas = rope_deltas
        elif self.rotary_emb.rope_type == "3D" and input_ids.size(1) != 1:
            position_ids, rope_deltas = get_rope_index(
                self.config,
                input_ids,
                image_grid_thw,
                image_grid_thw_video,
                attention_mask,
                second_per_grid_ts,
            )
            self.rope_deltas = rope_deltas
        elif self.rotary_emb.rope_type == "3D" or self.rotary_emb.rope_type == "video_rope": # decode stage
            batch_size, seq_length = input_ids.shape
            if past_key_values and self.rope_deltas:
                delta = past_key_values[0][1].shape[2] + self.rope_deltas
            elif past_key_values:
                delta = torch.tensor(past_key_values[0][1].shape[2])
            else:
                delta = torch.tensor(0)
            delta = delta.to(input_ids.device)
            position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        inputs_embeds = embeddings
        attention_mask = _causal_4d_mask(attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    image_mask,
                    audio_mask,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    image_mask=image_mask,
                    audio_mask=audio_mask,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class BailingMoeV2ForCausalLM(BailingMoeV2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: BailingMoeV2Config):
        super().__init__(config)
        self.model = BailingMoeV2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.word_embeddings

    def set_input_embeddings(self, value):
        self.model.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(BAILINGMOEV2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoeCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        query_embeds_image: Optional[torch.Tensor] = None,
        query_embeds_video: Optional[torch.Tensor] = None,
        query_embeds_audio: Optional[torch.Tensor] = None,
        query_embeds_audio_lengths: Optional[torch.Tensor] = None,
        placeholder_audio_loc_lens: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        image_grid_thw_video: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        num_logits_to_keep: Optional[int] = 0,
        image_mask=None,
        audio_mask=None,
        **kwargs,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer

        >>> model = BailingMoeV2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            query_embeds_image=query_embeds_image,
            query_embeds_video=query_embeds_video,
            query_embeds_audio=query_embeds_audio,
            query_embeds_audio_lengths=query_embeds_audio_lengths,
            placeholder_audio_loc_lens=placeholder_audio_loc_lens,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            image_grid_thw=image_grid_thw,
            image_grid_thw_video=image_grid_thw_video,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            second_per_grid_ts=second_per_grid_ts,
            image_mask=image_mask,
            audio_mask=audio_mask,
            **kwargs,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        # logits = logits.float()

        loss = None
        aux_loss = None

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(inplace_backward=True)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )



class MingImageTextEncoder(nn.Module):
    def __init__(
        self,
        llm_config={'_name_or_path': '', 'add_cross_attention': False, 'architectures': ['BailingMoeV2ForCausalLM'], 'attention_dropout': 0.0, 'bad_words_ids': None, 'begin_suppress_tokens': None, 'bos_token_id': None, 'chunk_size_feed_forward': 0, 'cross_attention_hidden_size': None, 'decoder_start_token_id': None, 'diversity_penalty': 0.0, 'do_sample': False, 'early_stopping': False, 'embedding_dropout': 0.0, 'encoder_no_repeat_ngram_size': 0, 'eos_token_id': 156895, 'exponential_decay_length_penalty': None, 'finetuning_task': None, 'first_k_dense_replace': 1, 'forced_bos_token_id': None, 'forced_eos_token_id': None, 'head_dim': 128, 'hidden_act': 'silu', 'hidden_size': 2048, 'id2label': {'0': 'LABEL_0', '1': 'LABEL_1'}, 'image_end_token': 157159, 'image_patch_token': 157157, 'image_start_token': 157158, 'video_end_token': 157161, 'video_patch_token': 157175, 'video_start_token': 157160, 'initializer_range': 0.006, 'intermediate_size': 5120, 'is_decoder': False, 'is_encoder_decoder': False, 'label2id': {'LABEL_0': 0, 'LABEL_1': 1}, 'length_penalty': 1.0, 'max_length': 20, 'max_position_embeddings': 32768, 'max_window_layers': 28, 'min_length': 0, 'model_type': 'bailing_moe_v2', 'moe_intermediate_size': 512, 'moe_router_topk_scaling_factor': 2.5, 'n_group': 8, 'no_repeat_ngram_size': 0, 'norm_head': False, 'norm_softmax': False, 'norm_topk_prob': True, 'num_attention_heads': 16, 'num_beam_groups': 1, 'num_beams': 1, 'num_experts': 256, 'num_experts_per_tok': 8, 'num_hidden_layers': 20, 'num_key_value_heads': 4, 'num_return_sequences': 1, 'num_shared_experts': 1, 'output_attentions': False, 'output_dropout': 0.0, 'output_hidden_states': False, 'output_router_logits': False, 'output_scores': False, 'pad_token_id': 156892, 'partial_rotary_factor': 0.5, 'prefix': None, 'pretraining_tp': 1, 'problem_type': None, 'pruned_heads': {}, 'remove_invalid_values': False, 'repetition_penalty': 1.0, 'return_dict': True, 'return_dict_in_generate': False, 'rms_norm_eps': 1e-06, 'rope_scaling': {'factor': None, 'type': 'video_rope'}, 'rope_theta': 600000, 'routed_scaling_factor': 2.5, 'router_type': 'MultiRouter', 'sep_token_id': None, 'sliding_window': 4096, 'spatial_merge_size': 2, 'suppress_tokens': None, 'task_specific_params': None, 'temperature': 1.0, 'tf_legacy_loss': False, 'tie_encoder_decoder': False, 'tie_word_embeddings': False, 'tokenizer_class': None, 'tokens_per_second': 2, 'top_k': 50, 'top_p': 1.0, 'topk_group': 4, 'torch_dtype': 'bfloat16', 'torchscript': False, 'typical_p': 1.0, 'use_bfloat16': False, 'use_bias': False, 'use_cache': True, 'use_expert_bias': True, 'use_qkv_bias': False, 'use_sliding_window': False, 'vocab_size': 157184, '_attn_implementation': 'eager'},
        vision_config={'_name_or_path': '', 'add_cross_attention': False, 'architectures': ['Qwen2_5_VisionTransformer'], 'bad_words_ids': None, 'begin_suppress_tokens': None, 'bos_token_id': None, 'chunk_size_feed_forward': 0, 'cross_attention_hidden_size': None, 'decoder_start_token_id': None, 'depth': 32, 'diversity_penalty': 0.0, 'do_sample': False, 'early_stopping': False, 'encoder_no_repeat_ngram_size': 0, 'eos_token_id': None, 'exponential_decay_length_penalty': None, 'finetuning_task': None, 'forced_bos_token_id': None, 'forced_eos_token_id': None, 'fullatt_block_indexes': [7, 15, 23, 31], 'hidden_act': 'silu', 'hidden_size': 1280, 'id2label': {'0': 'LABEL_0', '1': 'LABEL_1'}, 'in_channels': 3, 'in_chans': 3, 'intermediate_size': 3456, 'is_decoder': False, 'is_encoder_decoder': False, 'label2id': {'LABEL_0': 0, 'LABEL_1': 1}, 'length_penalty': 1.0, 'max_length': 20, 'min_length': 0, 'model_type': 'qwen2_5_vit', 'no_repeat_ngram_size': 0, 'num_beam_groups': 1, 'num_beams': 1, 'num_heads': 16, 'num_return_sequences': 1, 'out_hidden_size': 8192, 'output_attentions': False, 'output_hidden_states': False, 'output_scores': False, 'pad_token_id': None, 'patch_size': 14, 'prefix': None, 'problem_type': None, 'pruned_heads': {}, 'remove_invalid_values': False, 'repetition_penalty': 1.0, 'return_dict': True, 'return_dict_in_generate': False, 'sep_token_id': None, 'spatial_merge_size': 2, 'spatial_patch_size': 14, 'suppress_tokens': None, 'task_specific_params': None, 'temperature': 1.0, 'temporal_patch_size': 2, 'tf_legacy_loss': False, 'tie_encoder_decoder': False, 'tie_word_embeddings': True, 'tokenizer_class': None, 'tokens_per_second': 2, 'top_k': 50, 'top_p': 1.0, 'torch_dtype': 'bfloat16', 'torchscript': False, 'typical_p': 1.0, 'use_bfloat16': False, 'window_size': 112, '_attn_implementation': 'eager'},
        mlp_depth=2,
    ):
        super().__init__()
        self.vision = Qwen2_5_VisionTransformer(Qwen2_5_VLVisionConfig(**vision_config))
        self.model = BailingMoeV2ForCausalLM(BailingMoeV2Config(**llm_config))
        mlp_modules_img = [nn.Linear(self.vision.image_emb_dim, self.model.config.hidden_size)]
        for _ in range(1, mlp_depth):
            mlp_modules_img.append(nn.GELU())
            mlp_modules_img.append(nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size))
        self.linear_proj = nn.Sequential(*mlp_modules_img)

    def extract_image_feature(self, pixel_values, grid_thw):
        image_embeds = self.vision(pixel_values, grid_thw=grid_thw)
        image_embeds = self.linear_proj(image_embeds)
        image_embeds = F.normalize(image_embeds, dim=-1)
        return image_embeds

    def forward(
        self,
        input_ids,
        attention_mask,
        inputs_embeds=None,
        image_grid_thw=None,
        image_mask=None,
        output_hidden_states=True,
    ):
        return self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            image_grid_thw=image_grid_thw,
            use_cache=False,
            image_mask=image_mask,
            output_hidden_states=output_hidden_states,
        )
