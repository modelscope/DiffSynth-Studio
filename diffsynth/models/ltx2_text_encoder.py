import torch
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


class GemmaFeaturesExtractorProjLinear(torch.nn.Module):
    """
    Feature extractor module for Gemma models.
    This module applies a single linear projection to the input tensor.
    It expects a flattened feature tensor of shape (batch_size, 3840*49).
    The linear layer maps this to a (batch_size, 3840) embedding.
    Attributes:
        aggregate_embed (torch.nn.Linear): Linear projection layer.
    """

    def __init__(self) -> None:
        """
        Initialize the GemmaFeaturesExtractorProjLinear module.
        The input dimension is expected to be 3840 * 49, and the output is 3840.
        """
        super().__init__()
        self.aggregate_embed = torch.nn.Linear(3840 * 49, 3840, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the feature extractor.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3840 * 49).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 3840).
        """
        return self.aggregate_embed(x)


class _BasicTransformerBlock1D(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
    ):
        super().__init__()

        self.attn1 = Attention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            rope_type=rope_type,
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


class Embeddings1DConnector(torch.nn.Module):
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
        self.transformer_1d_blocks = torch.nn.ModuleList(
            [
                _BasicTransformerBlock1D(
                    dim=self.inner_dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    rope_type=rope_type,
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
        adjusted_hidden_states = torch.nn.functional.pad(non_zero_hidden_states, pad=(0, 0, 0, pad_length), value=0)
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


class LTX2TextEncoderPostModules(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.feature_extractor_linear = GemmaFeaturesExtractorProjLinear()
        self.embeddings_connector = Embeddings1DConnector()
        self.audio_embeddings_connector = Embeddings1DConnector()
