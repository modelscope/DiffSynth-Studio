import torch
from torch import Tensor


class Krea2TextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
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
                "hidden_size": 2560,
                "initializer_range": 0.02,
                "intermediate_size": 9728,
                "max_position_embeddings": 262144,
                "model_type": "qwen3_vl_text",
                "num_attention_heads": 32,
                "num_hidden_layers": 36,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-06,
                "rope_scaling": {
                    "mrope_interleaved": True,
                    "mrope_section": [24, 20, 20],
                    "rope_type": "default",
                },
                "rope_theta": 5000000,
                "tie_word_embeddings": True,
                "use_cache": True,
                "vocab_size": 151936,
            },
            "tie_word_embeddings": True,
            "transformers_version": "4.57.0.dev0",
            "video_token_id": 151656,
            "vision_config": {
                "deepstack_visual_indexes": [5, 11, 17],
                "depth": 24,
                "hidden_act": "gelu_pytorch_tanh",
                "hidden_size": 1024,
                "in_channels": 3,
                "initializer_range": 0.02,
                "intermediate_size": 4096,
                "model_type": "qwen3_vl",
                "num_heads": 16,
                "num_position_embeddings": 2304,
                "out_hidden_size": 2560,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
            "vision_end_token_id": 151653,
            "vision_start_token_id": 151652,
        })
        self.model = Qwen3VLForConditionalGeneration(config)
        self.config = config

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        output_hidden_states=True,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        return outputs.hidden_states
