# SPDX-License-Identifier: Apache-2.0
# MiniMax H3 text encoder: frozen Qwen3-VL (32B), layer-50 hidden-state extractor.
# Mirrors the checkpoint recipe: retain only the first N decoder layers, replace
# the final norm with Identity, and drop the LM head — the encoder contract is
# the unnormalized hidden state after decoder layer N (hidden_states[N], dim 5120).
import torch


class MiniMaxH3TextEncoder(torch.nn.Module):
    def __init__(self, num_retained_layers: int = 50):
        super().__init__()
        from transformers import Qwen3VLConfig, Qwen3VLModel

        self.num_retained_layers = num_retained_layers
        config = Qwen3VLConfig(**{
            "architectures": ["Qwen3VLForConditionalGeneration"],
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
                "hidden_size": 5120,
                "initializer_range": 0.02,
                "intermediate_size": 25600,
                "max_position_embeddings": 262144,
                "model_type": "qwen3_vl_text",
                "num_attention_heads": 64,
                "num_hidden_layers": num_retained_layers,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-06,
                "rope_scaling": {
                    "mrope_interleaved": True,
                    "mrope_section": [24, 20, 20],
                    "rope_type": "default",
                },
                "rope_theta": 5000000,
                "use_cache": True,
                "vocab_size": 151936,
            },
            "tie_word_embeddings": False,
            "transformers_version": "4.57.0.dev0",
            "video_token_id": 151656,
            "vision_config": {
                "deepstack_visual_indexes": [8, 16, 24],
                "depth": 27,
                "hidden_act": "gelu_pytorch_tanh",
                "hidden_size": 1152,
                "in_channels": 3,
                "initializer_range": 0.02,
                "intermediate_size": 4304,
                "model_type": "qwen3_vl",
                "num_heads": 16,
                "num_position_embeddings": 2304,
                "out_hidden_size": 5120,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
            "vision_end_token_id": 151653,
            "vision_start_token_id": 151652,
        })
        self.model = Qwen3VLModel(config)
        # Layer-N hidden-state contract: unnormalized output after N decoder
        # layers. Drop the final norm so last_hidden_state == hidden_states[N].
        self.model.language_model.norm = torch.nn.Identity()
        self.config = config
        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
            **kwargs,
        )
        return outputs.last_hidden_state
