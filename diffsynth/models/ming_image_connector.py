from torch import nn


class MingImageConnector(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import Qwen2Config, Qwen2Model
        config = Qwen2Config(**{
            "architectures": ["Qwen2ForCausalLM"],
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "dtype": "float32",
            "eos_token_id": 151645,
            "hidden_act": "silu",
            "hidden_size": 1536,
            "initializer_range": 0.02,
            "intermediate_size": 8960,
            "layer_types": [
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
                "full_attention",
            ],
            "max_position_embeddings": 32768,
            "max_window_layers": 21,
            "model_type": "qwen2",
            "num_attention_heads": 12,
            "num_hidden_layers": 28,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "rope_theta": 1000000.0,
            "sliding_window": None,
            "tie_word_embeddings": True,
            "transformers_version": "4.57.1",
            "use_cache": True,
            "use_sliding_window": False,
            "vocab_size": 151936,
        })
        self.model = Qwen2Model(config)
        for layer in self.model.layers:
            layer.self_attn.is_causal = False
        self._fp32_buffer_snapshot = [
            (m, n, b.detach().clone())
            for m in self.modules() for n, b in m.named_buffers(recurse=False)
            if b is not None and b.is_floating_point()
        ]

    def forward(self, inputs_embeds, attention_mask=None):
        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.dtype)
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # RoPE inv_freq buffers are not in the checkpoint and must stay fp32
        for m, n, b in self._fp32_buffer_snapshot:
            setattr(m, n, b.to(device=getattr(m, n).device))
        return self
