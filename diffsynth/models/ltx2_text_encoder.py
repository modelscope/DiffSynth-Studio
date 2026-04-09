import math
import torch
import torch.nn as nn
from einops import rearrange
from transformers import Gemma3ForConditionalGeneration, Gemma3Config, AutoTokenizer
from .ltx2_dit import (LTXRopeType, generate_freq_grid_np, generate_freq_grid_pytorch, precompute_freqs_cis, Attention,
                       FeedForward)
from .ltx2_common import rms_norm


class LTX2TextEncoder(Gemma3ForConditionalGeneration):
    def __init__(self):
        config = Gemma3Config(
            **{
                "architectures": ["Gemma3ForConditionalGeneration"],
                "boi_token_index": 255999,
                "dtype": "bfloat16",
                "eoi_token_index": 256000,
                "eos_token_id": [1, 106],
                "image_token_index": 262144,
                "initializer_range": 0.02,
                "mm_tokens_per_image": 256,
                "model_type": "gemma3",
                "text_config": {
                    "_sliding_window_pattern": 6,
                    "attention_bias": False,
                    "attention_dropout": 0.0,
                    "attn_logit_softcapping": None,
                    "cache_implementation": "hybrid",
                    "dtype": "bfloat16",
                    "final_logit_softcapping": None,
                    "head_dim": 256,
                    "hidden_activation": "gelu_pytorch_tanh",
                    "hidden_size": 3840,
                    "initializer_range": 0.02,
                    "intermediate_size": 15360,
                    "layer_types": [
                        "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
                        "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
                        "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
                        "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
                        "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"
                    ],
                    "max_position_embeddings": 131072,
                    "model_type": "gemma3_text",
                    "num_attention_heads": 16,
                    "num_hidden_layers": 48,
                    "num_key_value_heads": 8,
                    "query_pre_attn_scalar": 256,
                    "rms_norm_eps": 1e-06,
                    "rope_local_base_freq": 10000,
                    "rope_scaling": {
                        "factor": 8.0,
                        "rope_type": "linear"
                    },
                    "rope_theta": 1000000,
                    "sliding_window": 1024,
                    "sliding_window_pattern": 6,
                    "use_bidirectional_attention": False,
                    "use_cache": True,
                    "vocab_size": 262208
                },
                "transformers_version": "4.57.3",
                "vision_config": {
                    "attention_dropout": 0.0,
                    "dtype": "bfloat16",
                    "hidden_act": "gelu_pytorch_tanh",
                    "hidden_size": 1152,
                    "image_size": 896,
                    "intermediate_size": 4304,
                    "layer_norm_eps": 1e-06,
                    "model_type": "siglip_vision_model",
                    "num_attention_heads": 16,
                    "num_channels": 3,
                    "num_hidden_layers": 27,
                    "patch_size": 14,
                    "vision_use_head": False
                }
            })
        super().__init__(config)


class LTXVGemmaTokenizer:
    """
    Tokenizer wrapper for Gemma models compatible with LTXV processes.
    This class wraps HuggingFace's `AutoTokenizer` for use with Gemma text encoders,
    ensuring correct settings and output formatting for downstream consumption.
    """

    def __init__(self, tokenizer_path: str, max_length: int = 1024):
        """
        Initialize the tokenizer.
        Args:
            tokenizer_path (str): Path to the pretrained tokenizer files or model directory.
            max_length (int, optional): Max sequence length for encoding. Defaults to 256.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True, model_max_length=max_length
        )
        # Gemma expects left padding for chat-style prompts; for plain text it doesn't matter much.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length

    def tokenize_with_weights(self, text: str, return_word_ids: bool = False) -> dict[str, list[tuple[int, int]]]:
        """
        Tokenize the given text and return token IDs and attention weights.
        Args:
            text (str): The input string to tokenize.
            return_word_ids (bool, optional): If True, includes the token's position (index) in the output tuples.
                                              If False (default), omits the indices.
        Returns:
            dict[str, list[tuple[int, int]]] OR dict[str, list[tuple[int, int, int]]]:
                A dictionary with a "gemma" key mapping to:
                    - a list of (token_id, attention_mask) tuples if return_word_ids is False;
                    - a list of (token_id, attention_mask, index) tuples if return_word_ids is True.
        Example:
            >>> tokenizer = LTXVGemmaTokenizer("path/to/tokenizer", max_length=8)
            >>> tokenizer.tokenize_with_weights("hello world")
            {'gemma': [(1234, 1), (5678, 1), (2, 0), ...]}
        """
        text = text.strip()
        encoded = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask
        tuples = [
            (token_id, attn, i) for i, (token_id, attn) in enumerate(zip(input_ids[0], attention_mask[0], strict=True))
        ]
        out = {"gemma": tuples}

        if not return_word_ids:
            # Return only (token_id, attention_mask) pairs, omitting token position
            out = {k: [(t, w) for t, w, _ in v] for k, v in out.items()}

        return out


class GemmaFeaturesExtractorProjLinear(nn.Module):
    """
    Feature extractor module for Gemma models.
    This module applies a single linear projection to the input tensor.
    It expects a flattened feature tensor of shape (batch_size, 3840*49).
    The linear layer maps this to a (batch_size, 3840) embedding.
    Attributes:
        aggregate_embed (nn.Linear): Linear projection layer.
    """

    def __init__(self) -> None:
        """
        Initialize the GemmaFeaturesExtractorProjLinear module.
        The input dimension is expected to be 3840 * 49, and the output is 3840.
        """
        super().__init__()
        self.aggregate_embed = nn.Linear(3840 * 49, 3840, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_side: str = "left",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        encoded = torch.stack(hidden_states, dim=-1) if isinstance(hidden_states, (list, tuple)) else hidden_states
        dtype = encoded.dtype
        sequence_lengths = attention_mask.sum(dim=-1)
        normed = _norm_and_concat_padded_batch(encoded, sequence_lengths, padding_side)
        features = self.aggregate_embed(normed.to(dtype))
        return features, features


class GemmaSeperatedFeaturesExtractorProjLinear(nn.Module):
    """22B: per-token RMS norm → rescale → dual aggregate embeds"""

    def __init__(
        self,
        num_layers: int,
        embedding_dim: int,
        video_inner_dim: int,
        audio_inner_dim: int,
    ):
        super().__init__()
        in_dim = embedding_dim * num_layers
        self.video_aggregate_embed = torch.nn.Linear(in_dim, video_inner_dim, bias=True)
        self.audio_aggregate_embed = torch.nn.Linear(in_dim, audio_inner_dim, bias=True)
        self.embedding_dim = embedding_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_side: str = "left",  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        encoded = torch.stack(hidden_states, dim=-1) if isinstance(hidden_states, (list, tuple)) else hidden_states
        normed = norm_and_concat_per_token_rms(encoded, attention_mask)
        normed = normed.to(encoded.dtype)
        v_dim = self.video_aggregate_embed.out_features
        video = self.video_aggregate_embed(_rescale_norm(normed, v_dim, self.embedding_dim))
        audio = None
        if self.audio_aggregate_embed is not None:
            a_dim = self.audio_aggregate_embed.out_features
            audio = self.audio_aggregate_embed(_rescale_norm(normed, a_dim, self.embedding_dim))
        return video, audio



class _BasicTransformerBlock1D(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        apply_gated_attention: bool = False,
    ):
        super().__init__()

        self.attn1 = Attention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            rope_type=rope_type,
            apply_gated_attention=apply_gated_attention,
        )

        self.ff = FeedForward(
            dim,
            dim_out=dim,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.

        # 1. Normalization Before Self-Attention
        norm_hidden_states = rms_norm(hidden_states)

        norm_hidden_states = norm_hidden_states.squeeze(1)

        # 2. Self-Attention
        attn_output = self.attn1(norm_hidden_states, mask=attention_mask, pe=pe)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 3. Normalization before Feed-Forward
        norm_hidden_states = rms_norm(hidden_states)

        # 4. Feed-forward
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class Embeddings1DConnector(nn.Module):
    """
    Embeddings1DConnector applies a 1D transformer-based processing to sequential embeddings (e.g., for video, audio, or
    other modalities). It supports rotary positional encoding (rope), optional causal temporal positioning, and can
    substitute padded positions with learnable registers. The module is highly configurable for head size, number of
    layers, and register usage.
    Args:
        attention_head_dim (int): Dimension of each attention head (default=128).
        num_attention_heads (int): Number of attention heads (default=30).
        num_layers (int): Number of transformer layers (default=2).
        positional_embedding_theta (float): Scaling factor for position embedding (default=10000.0).
        positional_embedding_max_pos (list[int] | None): Max positions for positional embeddings (default=[1]).
        causal_temporal_positioning (bool): If True, uses causal attention (default=False).
        num_learnable_registers (int | None): Number of learnable registers to replace padded tokens. If None, disables
            register replacement. (default=128)
        rope_type (LTXRopeType): The RoPE variant to use (default=DEFAULT_ROPE_TYPE).
        double_precision_rope (bool): Use double precision rope calculation (default=False).
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        attention_head_dim: int = 128,
        num_attention_heads: int = 30,
        num_layers: int = 2,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = [4096],
        causal_temporal_positioning: bool = False,
        num_learnable_registers: int | None = 128,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        double_precision_rope: bool = True,
        apply_gated_attention: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.causal_temporal_positioning = causal_temporal_positioning
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = (
            positional_embedding_max_pos if positional_embedding_max_pos is not None else [1]
        )
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.transformer_1d_blocks = nn.ModuleList(
            [
                _BasicTransformerBlock1D(
                    dim=self.inner_dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    rope_type=rope_type,
                    apply_gated_attention=apply_gated_attention,
                )
                for _ in range(num_layers)
            ]
        )

        self.num_learnable_registers = num_learnable_registers
        if self.num_learnable_registers:
            self.learnable_registers = nn.Parameter(
                torch.rand(self.num_learnable_registers, self.inner_dim, dtype=torch.bfloat16) * 2.0 - 1.0
            )

    def _replace_padded_with_learnable_registers(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden_states.shape[1] % self.num_learnable_registers == 0, (
            f"Hidden states sequence length {hidden_states.shape[1]} must be divisible by num_learnable_registers "
            f"{self.num_learnable_registers}."
        )

        num_registers_duplications = hidden_states.shape[1] // self.num_learnable_registers
        learnable_registers = torch.tile(self.learnable_registers, (num_registers_duplications, 1))
        attention_mask_binary = (attention_mask.squeeze(1).squeeze(1).unsqueeze(-1) >= -9000.0).int()

        non_zero_hidden_states = hidden_states[:, attention_mask_binary.squeeze().bool(), :]
        non_zero_nums = non_zero_hidden_states.shape[1]
        pad_length = hidden_states.shape[1] - non_zero_nums
        adjusted_hidden_states = nn.functional.pad(non_zero_hidden_states, pad=(0, 0, 0, pad_length), value=0)
        flipped_mask = torch.flip(attention_mask_binary, dims=[1])
        hidden_states = flipped_mask * adjusted_hidden_states + (1 - flipped_mask) * learnable_registers

        attention_mask = torch.full_like(
            attention_mask,
            0.0,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        return hidden_states, attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Embeddings1DConnector.
        Args:
            hidden_states (torch.Tensor): Input tensor of embeddings (shape [batch, seq_len, feature_dim]).
            attention_mask (torch.Tensor|None): Optional mask for valid tokens (shape compatible with hidden_states).
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Processed features and the corresponding (possibly modified) mask.
        """
        if self.num_learnable_registers:
            hidden_states, attention_mask = self._replace_padded_with_learnable_registers(hidden_states, attention_mask)

        indices_grid = torch.arange(hidden_states.shape[1], dtype=torch.float32, device=hidden_states.device)
        indices_grid = indices_grid[None, None, :]
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
            hidden_states = block(hidden_states, attention_mask=attention_mask, pe=freqs_cis)

        hidden_states = rms_norm(hidden_states)

        return hidden_states, attention_mask


class LTX2TextEncoderPostModules(nn.Module):
    def __init__(
        self,
        separated_audio_video: bool = False,
        embedding_dim_gemma: int = 3840,
        num_layers_gemma: int = 49,
        video_attention_heads: int = 32,
        video_attention_head_dim: int = 128,
        audio_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        num_connector_layers: int = 2,
        apply_gated_attention: bool = False,
    ):
        super().__init__()
        if not separated_audio_video:
            self.feature_extractor_linear = GemmaFeaturesExtractorProjLinear()
            self.embeddings_connector = Embeddings1DConnector()
            self.audio_embeddings_connector = Embeddings1DConnector()
        else:
            # LTX-2.3
            self.feature_extractor_linear = GemmaSeperatedFeaturesExtractorProjLinear(
                num_layers_gemma, embedding_dim_gemma, video_attention_heads * video_attention_head_dim,
                audio_attention_heads * audio_attention_head_dim)
            self.embeddings_connector = Embeddings1DConnector(
                attention_head_dim=video_attention_head_dim,
                num_attention_heads=video_attention_heads,
                num_layers=num_connector_layers,
                apply_gated_attention=apply_gated_attention,
            )
            self.audio_embeddings_connector = Embeddings1DConnector(
                attention_head_dim=audio_attention_head_dim,
                num_attention_heads=audio_attention_heads,
                num_layers=num_connector_layers,
                apply_gated_attention=apply_gated_attention,
            )

    def create_embeddings(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor | None,
        additive_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        video_encoded, video_mask = self.embeddings_connector(video_features, additive_attention_mask)
        video_encoded, binary_mask = _to_binary_mask(video_encoded, video_mask)
        audio_encoded, _ = self.audio_embeddings_connector(audio_features, additive_attention_mask)

        return video_encoded, audio_encoded, binary_mask

    def process_hidden_states(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
        padding_side: str = "left",
    ):
        video_feats, audio_feats = self.feature_extractor_linear(hidden_states, attention_mask, padding_side)
        additive_mask = _convert_to_additive_mask(attention_mask, video_feats.dtype)
        video_enc, audio_enc, binary_mask = self.create_embeddings(video_feats, audio_feats, additive_mask)
        return video_enc, audio_enc, binary_mask


def _norm_and_concat_padded_batch(
    encoded_text: torch.Tensor,
    sequence_lengths: torch.Tensor,
    padding_side: str = "right",
) -> torch.Tensor:
    """Normalize and flatten multi-layer hidden states, respecting padding.
    Performs per-batch, per-layer normalization using masked mean and range,
    then concatenates across the layer dimension.
    Args:
        encoded_text: Hidden states of shape [batch, seq_len, hidden_dim, num_layers].
        sequence_lengths: Number of valid (non-padded) tokens per batch item.
        padding_side: Whether padding is on "left" or "right".
    Returns:
        Normalized tensor of shape [batch, seq_len, hidden_dim * num_layers],
        with padded positions zeroed out.
    """
    b, t, d, l = encoded_text.shape  # noqa: E741
    device = encoded_text.device
    # Build mask: [B, T, 1, 1]
    token_indices = torch.arange(t, device=device)[None, :]  # [1, T]
    if padding_side == "right":
        # For right padding, valid tokens are from 0 to sequence_length-1
        mask = token_indices < sequence_lengths[:, None]  # [B, T]
    elif padding_side == "left":
        # For left padding, valid tokens are from (T - sequence_length) to T-1
        start_indices = t - sequence_lengths[:, None]  # [B, 1]
        mask = token_indices >= start_indices  # [B, T]
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")
    mask = rearrange(mask, "b t -> b t 1 1")
    eps = 1e-6
    # Compute masked mean: [B, 1, 1, L]
    masked = encoded_text.masked_fill(~mask, 0.0)
    denom = (sequence_lengths * d).view(b, 1, 1, 1)
    mean = masked.sum(dim=(1, 2), keepdim=True) / (denom + eps)
    # Compute masked min/max: [B, 1, 1, L]
    x_min = encoded_text.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = encoded_text.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)
    range_ = x_max - x_min
    # Normalize only the valid tokens
    normed = 8 * (encoded_text - mean) / (range_ + eps)
    # concat to be [Batch, T,  D * L] - this preserves the original structure
    normed = normed.reshape(b, t, -1)  # [B, T, D * L]
    # Apply mask to preserve original padding (set padded positions to 0)
    mask_flattened = rearrange(mask, "b t 1 1 -> b t 1").expand(-1, -1, d * l)
    normed = normed.masked_fill(~mask_flattened, 0.0)

    return normed


def _convert_to_additive_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (attention_mask - 1).to(dtype).reshape(
        (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])) * torch.finfo(dtype).max

def _to_binary_mask(encoded: torch.Tensor, encoded_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert connector output mask to binary mask and apply to encoded tensor."""
    binary_mask = (encoded_mask < 0.000001).to(torch.int64)
    binary_mask = binary_mask.reshape([encoded.shape[0], encoded.shape[1], 1])
    encoded = encoded * binary_mask
    return encoded, binary_mask


def norm_and_concat_per_token_rms(
    encoded_text: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-token RMSNorm normalization for V2 models.
    Args:
        encoded_text: [B, T, D, L]
        attention_mask: [B, T] binary mask
    Returns:
        [B, T, D*L] normalized tensor with padding zeroed out.
    """
    B, T, D, L = encoded_text.shape  # noqa: N806
    variance = torch.mean(encoded_text**2, dim=2, keepdim=True)  # [B,T,1,L]
    normed = encoded_text * torch.rsqrt(variance + 1e-6)
    normed = normed.reshape(B, T, D * L)
    mask_3d = attention_mask.bool().unsqueeze(-1)  # [B, T, 1]
    return torch.where(mask_3d, normed, torch.zeros_like(normed))


def _rescale_norm(x: torch.Tensor, target_dim: int, source_dim: int) -> torch.Tensor:
    """Rescale normalization: x * sqrt(target_dim / source_dim)."""
    return x * math.sqrt(target_dim / source_dim)
