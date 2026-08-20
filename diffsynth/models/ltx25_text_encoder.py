import copy
import math
from typing import NamedTuple

import torch

from .ltx2_common import rms_norm
from .ltx2_dit import (
    Attention,
    FeedForward,
    LTXRopeType,
    generate_freq_grid_np,
    generate_freq_grid_pytorch,
    precompute_freqs_cis,
)


LTX25_GEMMA_CONFIG = {'architectures': ['Gemma4UnifiedForConditionalGeneration'],
 'audio_config': {'_name_or_path': '',
                  'architectures': None,
                  'audio_embed_dim': 640,
                  'chunk_size_feed_forward': 0,
                  'dtype': 'bfloat16',
                  'id2label': {'0': 'LABEL_0', '1': 'LABEL_1'},
                  'initializer_range': 0.02,
                  'is_encoder_decoder': False,
                  'label2id': {'LABEL_0': 0, 'LABEL_1': 1},
                  'model_type': 'gemma4_unified_audio',
                  'output_attentions': False,
                  'output_hidden_states': False,
                  'problem_type': None,
                  'return_dict': True,
                  'rms_norm_eps': 1e-06},
 'audio_token_id': 258881,
 'boa_token_id': 256000,
 'boi_token_id': 255999,
 'dtype': 'bfloat16',
 'eoa_token_index': 258883,
 'eoi_token_id': 258882,
 'eos_token_id': [1, 106],
 'gemma_version': 'gemma4-12b-ltx-v1',
 'image_token_id': 258880,
 'initializer_range': 0.02,
 'model_type': 'gemma4_unified',
 'text_config': {'attention_bias': False,
                 'attention_dropout': 0.0,
                 'attention_k_eq_v': True,
                 'bos_token_id': 2,
                 'dtype': 'bfloat16',
                 'enable_moe_block': False,
                 'eos_token_id': 1,
                 'final_logit_softcapping': 30.0,
                 'global_head_dim': 512,
                 'head_dim': 256,
                 'hidden_activation': 'gelu_pytorch_tanh',
                 'hidden_size': 3840,
                 'hidden_size_per_layer_input': 0,
                 'initializer_range': 0.02,
                 'intermediate_size': 15360,
                 'layer_types': ['sliding_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'full_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'full_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'full_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'full_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'full_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'full_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'full_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'sliding_attention',
                                 'full_attention'],
                 'max_position_embeddings': 262144,
                 'model_type': 'gemma4_unified_text',
                 'moe_intermediate_size': None,
                 'num_attention_heads': 16,
                 'num_experts': None,
                 'num_global_key_value_heads': 1,
                 'num_hidden_layers': 48,
                 'num_key_value_heads': 8,
                 'num_kv_shared_layers': 0,
                 'pad_token_id': 0,
                 'rms_norm_eps': 1e-06,
                 'rope_parameters': {'full_attention': {'partial_rotary_factor': 0.25,
                                                        'rope_theta': 1000000.0,
                                                        'rope_type': 'proportional'},
                                     'sliding_attention': {'rope_theta': 10000.0, 'rope_type': 'default'}},
                 'sliding_window': 1024,
                 'tie_word_embeddings': True,
                 'top_k_experts': None,
                 'use_bidirectional_attention': 'vision',
                 'use_cache': True,
                 'use_double_wide_mlp': False,
                 'vocab_size': 262144,
                 'vocab_size_per_layer_input': 262144},
 'tie_word_embeddings': True,
 'transformers_version': '5.10.1',
 'video_token_id': 258884,
 'vision_config': {'_name_or_path': '',
                   'architectures': None,
                   'chunk_size_feed_forward': 0,
                   'dtype': 'bfloat16',
                   'id2label': {'0': 'LABEL_0', '1': 'LABEL_1'},
                   'initializer_range': 0.02,
                   'is_encoder_decoder': False,
                   'label2id': {'LABEL_0': 0, 'LABEL_1': 1},
                   'mm_embed_dim': 3840,
                   'mm_posemb_size': 1120,
                   'model_type': 'gemma4_unified_vision',
                   'num_soft_tokens': 280,
                   'output_attentions': False,
                   'output_hidden_states': False,
                   'output_proj_dims': 3840,
                   'patch_size': 16,
                   'pooling_kernel_size': 3,
                   'problem_type': None,
                   'return_dict': True,
                   'rms_norm_eps': 1e-06}}


class LTX25TextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration

        self.config = Gemma4UnifiedConfig(**copy.deepcopy(LTX25_GEMMA_CONFIG))
        self.model = Gemma4UnifiedForConditionalGeneration(self.config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def norm_and_concat_per_token_rms(
    encoded_text: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, sequence_length, embedding_dim, num_layers = encoded_text.shape
    variance = torch.mean(encoded_text**2, dim=2, keepdim=True)
    normed = encoded_text * torch.rsqrt(variance + 1e-6)
    normed = normed.reshape(batch_size, sequence_length, embedding_dim * num_layers)
    return torch.where(attention_mask.bool().unsqueeze(-1), normed, torch.zeros_like(normed))


def _rescale_norm(x: torch.Tensor, target_dim: int, source_dim: int) -> torch.Tensor:
    return x * math.sqrt(target_dim / source_dim)


class LTX25FeatureExtractorV2(torch.nn.Module):
    def __init__(
        self,
        video_aggregate_embed: torch.nn.Linear,
        embedding_dim: int,
        audio_aggregate_embed: torch.nn.Linear | None = None,
    ):
        super().__init__()
        self.video_aggregate_embed = video_aggregate_embed
        self.audio_aggregate_embed = audio_aggregate_embed
        self.embedding_dim = embedding_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_side: str = "left",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del padding_side
        encoded = torch.stack(hidden_states, dim=-1) if isinstance(hidden_states, (list, tuple)) else hidden_states
        normed = norm_and_concat_per_token_rms(encoded, attention_mask).to(encoded.dtype)
        video = self.video_aggregate_embed(
            _rescale_norm(normed, self.video_aggregate_embed.out_features, self.embedding_dim)
        )
        audio = None
        if self.audio_aggregate_embed is not None:
            audio = self.audio_aggregate_embed(
                _rescale_norm(normed, self.audio_aggregate_embed.out_features, self.embedding_dim)
            )
        return video, audio


class LTX25Embeddings1DConnector(torch.nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        num_attention_heads: int,
        num_layers: int,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        num_learnable_registers: int | None = 128,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        double_precision_rope: bool = True,
        apply_gated_attention: bool = True,
        ff_bias: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos if positional_embedding_max_pos is not None else [1]
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.transformer_1d_blocks = torch.nn.ModuleList(
            [
                LTX25BasicTransformerBlock1D(
                    dim=self.inner_dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    rope_type=rope_type,
                    apply_gated_attention=apply_gated_attention,
                    ff_bias=ff_bias,
                )
                for _ in range(num_layers)
            ]
        )
        self.num_learnable_registers = num_learnable_registers
        if self.num_learnable_registers:
            self.learnable_registers = torch.nn.Parameter(
                torch.rand(self.num_learnable_registers, self.inner_dim, dtype=torch.bfloat16) * 2.0 - 1.0
            )

    def _replace_padded_with_learnable_registers(
        self,
        hidden_states: torch.Tensor,
        additive_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = hidden_states.shape
        assert sequence_length % self.num_learnable_registers == 0
        registers = self.learnable_registers.to(hidden_states).repeat(sequence_length // self.num_learnable_registers, 1)
        registers = registers.unsqueeze(0).expand(batch_size, -1, -1)
        binary_mask = (additive_attention_mask[:, 0, 0, :].unsqueeze(-1) >= 0).to(hidden_states.dtype)
        hidden_states = binary_mask * hidden_states + (1 - binary_mask) * registers
        return hidden_states, torch.zeros_like(additive_attention_mask)

    def forward(
        self,
        hidden_states: torch.Tensor,
        additive_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_learnable_registers:
            hidden_states, additive_attention_mask = self._replace_padded_with_learnable_registers(
                hidden_states,
                additive_attention_mask,
            )
        indices_grid = torch.arange(hidden_states.shape[1], dtype=torch.float32, device=hidden_states.device)
        indices_grid = indices_grid[None, None, :].expand(hidden_states.shape[0], -1, -1)
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        freqs_cis = precompute_freqs_cis(
            indices_grid=indices_grid,
            dim=self.inner_dim,
            out_dtype=hidden_states.dtype,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            num_attention_heads=self.num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
        )
        for block in self.transformer_1d_blocks:
            hidden_states = block(hidden_states, additive_attention_mask=additive_attention_mask, pe=freqs_cis)
        return rms_norm(hidden_states), additive_attention_mask


class LTX25BasicTransformerBlock1D(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rope_type: LTXRopeType,
        apply_gated_attention: bool,
        ff_bias: bool,
    ):
        super().__init__()
        self.attn1 = Attention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            rope_type=rope_type,
            apply_gated_attention=apply_gated_attention,
        )
        self.ff = FeedForward(dim, dim_out=dim, bias=ff_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        additive_attention_mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm_hidden_states = rms_norm(hidden_states).squeeze(1)
        hidden_states = self.attn1(norm_hidden_states, mask=additive_attention_mask, pe=pe) + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        hidden_states = self.ff(rms_norm(hidden_states)) + hidden_states
        return hidden_states.squeeze(1) if hidden_states.ndim == 4 else hidden_states


class LTX25EmbeddingsProcessorOutput(NamedTuple):
    video_encoding: torch.Tensor
    audio_encoding: torch.Tensor | None
    attention_mask: torch.Tensor


def _convert_to_additive_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (attention_mask.to(torch.int64) - 1).to(dtype).reshape(
        attention_mask.shape[0], 1, 1, attention_mask.shape[-1]
    ) * torch.finfo(dtype).max


def _right_pad_order(additive_attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    binary = (additive_attention_mask[:, 0, 0, :] >= 0).to(torch.int32)
    sort_indices = torch.argsort(binary, dim=-1, descending=True, stable=True)
    reordered = torch.gather(binary, 1, sort_indices)
    additive = (reordered.to(additive_attention_mask.dtype) - 1) * torch.finfo(additive_attention_mask.dtype).max
    return sort_indices, additive[:, None, None, :]


class LTX25TextEncoderPostModules(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int = 3840,
        num_layers: int = 49,
        video_attention_heads: int = 32,
        video_attention_head_dim: int = 128,
        audio_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        num_connector_layers: int = 8,
        connector_max_positions: list[int] | None = None,
        connector_ff_bias: bool = True,
    ):
        super().__init__()
        self.feature_extractor = LTX25FeatureExtractorV2(
            video_aggregate_embed=torch.nn.Linear(
                embedding_dim * num_layers,
                video_attention_heads * video_attention_head_dim,
                bias=True,
            ),
            embedding_dim=embedding_dim,
            audio_aggregate_embed=torch.nn.Linear(
                embedding_dim * num_layers,
                audio_attention_heads * audio_attention_head_dim,
                bias=True,
            ),
        )
        connector_max_positions = [4096] if connector_max_positions is None else connector_max_positions
        self.video_connector = LTX25Embeddings1DConnector(
            attention_head_dim=video_attention_head_dim,
            num_attention_heads=video_attention_heads,
            num_layers=num_connector_layers,
            positional_embedding_max_pos=connector_max_positions,
            ff_bias=connector_ff_bias,
        )
        self.audio_connector = LTX25Embeddings1DConnector(
            attention_head_dim=audio_attention_head_dim,
            num_attention_heads=audio_attention_heads,
            num_layers=num_connector_layers,
            positional_embedding_max_pos=connector_max_positions,
            ff_bias=connector_ff_bias,
        )

    def create_embeddings(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor | None,
        additive_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if audio_features is None:
            raise ValueError("LTX-2.5 requires audio features for the audio connector.")
        sort_indices, connector_mask = _right_pad_order(additive_attention_mask)
        video_features = torch.gather(video_features, 1, sort_indices.unsqueeze(-1).expand_as(video_features))
        video_encoded, video_mask = self.video_connector(video_features, connector_mask)
        binary_mask = (video_mask < 0.000001).to(torch.int64).reshape(video_encoded.shape[0], video_encoded.shape[1], 1)
        video_encoded = video_encoded * binary_mask
        audio_features = torch.gather(audio_features, 1, sort_indices.unsqueeze(-1).expand_as(audio_features))
        audio_encoded, _ = self.audio_connector(audio_features, connector_mask)
        return video_encoded, audio_encoded, binary_mask.squeeze(-1)

    def process_hidden_states(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
        padding_side: str = "left",
    ) -> LTX25EmbeddingsProcessorOutput:
        video_features, audio_features = self.feature_extractor(hidden_states, attention_mask, padding_side)
        additive_attention_mask = _convert_to_additive_mask(attention_mask, video_features.dtype)
        video_encoding, audio_encoding, binary_mask = self.create_embeddings(
            video_features,
            audio_features,
            additive_attention_mask,
        )
        return LTX25EmbeddingsProcessorOutput(video_encoding, audio_encoding, binary_mask)
