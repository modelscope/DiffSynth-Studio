from transformers import Qwen3Model, Qwen3Config
import torch


class ZImageTextEncoder(torch.nn.Module):
    def __init__(self, model_size="4B"):
        super().__init__()
        config_dict = {
            "0.6B": Qwen3Config(**{
                "architectures": [
                    "Qwen3ForCausalLM"
                ],
                "attention_bias": False,
                "attention_dropout": 0.0,
                "bos_token_id": 151643,
                "eos_token_id": 151645,
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 1024,
                "initializer_range": 0.02,
                "intermediate_size": 3072,
                "max_position_embeddings": 40960,
                "max_window_layers": 28,
                "model_type": "qwen3",
                "num_attention_heads": 16,
                "num_hidden_layers": 28,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-06,
                "rope_scaling": None,
                "rope_theta": 1000000,
                "sliding_window": None,
                "tie_word_embeddings": True,
                "torch_dtype": "bfloat16",
                "transformers_version": "4.51.0",
                "use_cache": True,
                "use_sliding_window": False,
                "vocab_size": 151936
            }),
            "4B": Qwen3Config(**{
                "architectures": [
                    "Qwen3ForCausalLM"
                ],
                "attention_bias": False,
                "attention_dropout": 0.0,
                "bos_token_id": 151643,
                "eos_token_id": 151645,
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 2560,
                "initializer_range": 0.02,
                "intermediate_size": 9728,
                "max_position_embeddings": 40960,
                "max_window_layers": 36,
                "model_type": "qwen3",
                "num_attention_heads": 32,
                "num_hidden_layers": 36,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-06,
                "rope_scaling": None,
                "rope_theta": 1000000,
                "sliding_window": None,
                "tie_word_embeddings": True,
                "torch_dtype": "bfloat16",
                "transformers_version": "4.51.0",
                "use_cache": True,
                "use_sliding_window": False,
                "vocab_size": 151936
            }),
            "8B": Qwen3Config(**{
                "architectures": [
                    "Qwen3ForCausalLM"
                ],
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
                "max_position_embeddings": 40960,
                "max_window_layers": 36,
                "model_type": "qwen3",
                "num_attention_heads": 32,
                "num_hidden_layers": 36,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-06,
                "rope_scaling": None,
                "rope_theta": 1000000,
                "sliding_window": None,
                "tie_word_embeddings": False,
                "transformers_version": "4.56.1",
                "use_cache": True,
                "use_sliding_window": False,
                "vocab_size": 151936
            })
        }
        config = config_dict[model_size]
        self.model = Qwen3Model(config)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
