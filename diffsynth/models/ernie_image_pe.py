"""
Ernie-Image PE (Prompt Enhancement) for DiffSynth-Studio.

Wraps transformers Ministral3ForCausalLM for prompt rewriting/enhancement.
Pattern: lazy import + manual config dict + torch.nn.Module wrapper.
"""

import torch


class ErnieImagePE(torch.nn.Module):
    """
    Prompt Enhancement model using Ministral3ForCausalLM (transformers).
    Used to rewrite/enhance short prompts into detailed descriptions.
    """

    def __init__(self):
        super().__init__()
        from transformers import Ministral3Config, Ministral3ForCausalLM

        config = Ministral3Config(**{
            "architectures": ["Ministral3ForCausalLM"],
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
            "transformers_version": "5.3.0",
            "use_cache": True,
            "vocab_size": 131072,
        })
        self.model = Ministral3ForCausalLM(config)
        self.config = config

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        max_new_tokens=None,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        pad_token_id=None,
        eos_token_id=None,
        **kwargs,
    ):
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        return outputs
