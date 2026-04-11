"""
Ernie-Image TextEncoder for DiffSynth-Studio.

Wraps transformers Ministral3Model to output text embeddings.
Pattern: lazy import + manual config dict + torch.nn.Module wrapper.
Only loads the text (language) model, ignoring vision components.
"""

import torch


class ErnieImageTextEncoder(torch.nn.Module):
    """
    Text encoder using Ministral3Model (transformers).
    Only the text_config portion of the full Mistral3Model checkpoint.
    Uses the base model (no lm_head) since the checkpoint only has embeddings.
    """

    def __init__(self):
        super().__init__()
        from transformers import Ministral3Config, Ministral3Model

        text_config = {
            "attention_dropout": 0.0,
            "bos_token_id": 1,
            "dtype": "bfloat16",
            "eos_token_id": 2,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 3072,
            "initializer_range": 0.02,
            "intermediate_size": 9216,
            "max_position_embeddings": 262144,
            "model_type": "ministral3",
            "num_attention_heads": 32,
            "num_hidden_layers": 26,
            "num_key_value_heads": 8,
            "pad_token_id": 11,
            "rms_norm_eps": 1e-05,
            "rope_parameters": {
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": 16.0,
                "llama_4_scaling_beta": 0.1,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 16384,
                "rope_theta": 1000000.0,
                "rope_type": "yarn",
                "type": "yarn",
            },
            "sliding_window": None,
            "tie_word_embeddings": True,
            "use_cache": True,
            "vocab_size": 131072,
        }
        config = Ministral3Config(**text_config)
        self.model = Ministral3Model(config)
        self.config = config

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        return (outputs.hidden_states,)
