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
        # transformers 5.8 trimmed Mistral3's positional args; pass everything
        # as kwargs so the call survives across transformers versions. Forward
        # the four removed-from-positional args through **kwargs (transformers
        # 5.8 honors output_hidden_states / output_attentions via
        # TransformersKwargs; return_dict / cache_position are no-ops in 5.8
        # but forwarded so older transformers versions still receive them).
        if output_hidden_states is not None:
            kwargs.setdefault("output_hidden_states", output_hidden_states)
        if output_attentions is not None:
            kwargs.setdefault("output_attentions", output_attentions)
        if return_dict is not None:
            kwargs.setdefault("return_dict", return_dict)
        if cache_position is not None:
            kwargs.setdefault("cache_position", cache_position)
        return super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            image_sizes=image_sizes,
            **kwargs,
        )

