import torch
import torch.nn as nn
from .ideogram4_dit import Ideogram4Fp8QuantBackend  # noqa: F401

LLM_TOKEN_INDICATOR = 3
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)

_DEFAULT_TEXT_ENCODER_CONFIG = {
    "architectures": ["Qwen3VLModel"],
    "dtype": "bfloat16",
    "image_token_id": 151655,
    "model_type": "qwen3_vl",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 12288,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "pad_token_id": None,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_theta": 5000000,
            "rope_type": "default",
        },
        "use_cache": True,
        "vocab_size": 151936,
    },
    "tie_word_embeddings": False,
    "transformers_version": "5.8.0",
    "video_token_id": 151656,
    "vision_config": {
        "deepstack_visual_indexes": [8, 16, 24],
        "depth": 27,
        "dtype": "bfloat16",
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "model_type": "qwen3_vl_vision",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 4096,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
    "ideogram_fp8_weight_only": True,
}


class Ideogram4TextEncoder(nn.Module):
    """Qwen3-VL-8B-Instruct wrapper that extracts hidden states from specific layers."""

    def __init__(self, config_path: str = None, **kwargs) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel
        if config_path is None:
            config_kwargs = {k: v for k, v in _DEFAULT_TEXT_ENCODER_CONFIG.items() if k != "model_type"}
            config = AutoConfig.for_model("qwen3_vl", **config_kwargs)
        else:
            config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
        self.model = AutoModel.from_config(config, trust_remote_code=True)
        self.config = config

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Extract hidden states from specific layers of Qwen3-VL.

        Args:
            token_ids: (B, L) token ids
            attention_mask: (B, L) attention mask
            text_position_ids: (B, L) position ids for text tokens

        Returns:
            (B, L, hidden_size * num_activation_layers) concatenated hidden states
        """
        from transformers.masking_utils import create_causal_mask

        language_model = self.model.language_model

        inputs_embeds = language_model.embed_tokens(token_ids)

        position_ids_4d = text_position_ids[None, ...].expand(4, text_position_ids.shape[0], -1)
        text_position_ids_4d = position_ids_4d[0]
        mrope_position_ids = position_ids_4d[1:]

        causal_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids_4d,
        )
        position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)

        tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
        captured: dict[int, torch.Tensor] = {}
        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(language_model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids_4d,
                past_key_values=None,
                position_embeddings=position_embeddings,
            )
            if layer_idx in tap_set:
                captured[layer_idx] = hidden_states

        selected = [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]
        stacked = torch.stack(selected, dim=0)
        stacked = torch.permute(stacked, (1, 2, 3, 0))
        batch_size, seq_len, hidden_size = stacked.shape[:3]
        stacked = stacked.reshape(batch_size, seq_len, -1)

        text_mask = attention_mask.to(stacked.dtype).unsqueeze(-1)
        stacked = stacked * text_mask
        return stacked.to(torch.float32)
