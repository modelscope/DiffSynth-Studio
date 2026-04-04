from transformers import Mistral3ForConditionalGeneration, Mistral3Config


class Flux2TextEncoder(Mistral3ForConditionalGeneration):
    def __init__(self):
        config = Mistral3Config(**{
            "architectures": [
                "Mistral3ForConditionalGeneration"
            ],
            "dtype": "bfloat16",
            "image_token_index": 10,
            "model_type": "mistral3",
            "multimodal_projector_bias": False,
            "projector_hidden_act": "gelu",
            "spatial_merge_size": 2,
            "text_config": {
                "attention_dropout": 0.0,
                "dtype": "bfloat16",
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 5120,
                "initializer_range": 0.02,
                "intermediate_size": 32768,
                "max_position_embeddings": 131072,
                "model_type": "mistral",
                "num_attention_heads": 32,
                "num_hidden_layers": 40,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-05,
                "rope_theta": 1000000000.0,
                "sliding_window": None,
                "use_cache": True,
                "vocab_size": 131072
            },
            "transformers_version": "4.57.1",
            "vision_config": {
                "attention_dropout": 0.0,
                "dtype": "bfloat16",
                "head_dim": 64,
                "hidden_act": "silu",
                "hidden_size": 1024,
                "image_size": 1540,
                "initializer_range": 0.02,
                "intermediate_size": 4096,
                "model_type": "pixtral",
                "num_attention_heads": 16,
                "num_channels": 3,
                "num_hidden_layers": 24,
                "patch_size": 14,
                "rope_theta": 10000.0
            },
            "vision_feature_layer": -1
        })
        super().__init__(config)
    
    def forward(self, input_ids = None, pixel_values = None, attention_mask = None, position_ids = None, past_key_values = None, inputs_embeds = None, labels = None, use_cache = None, output_attentions = None, output_hidden_states = None, return_dict = None, cache_position = None, logits_to_keep = 0, image_sizes = None, **kwargs):
        return super().forward(input_ids, pixel_values, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict, cache_position, logits_to_keep, image_sizes, **kwargs)

