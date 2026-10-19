import torch
from typing import Optional
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding


class _MOSSImageRotaryEmbedding(Qwen3_5TextRotaryEmbedding):
    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        if self.inv_freq.dtype != torch.float32:
            inv_freq, attention_scaling = self.compute_default_rope_parameters(self.config, None)
            inv_freq = inv_freq.to(self.inv_freq.device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
            self.attention_scaling = attention_scaling
        return self


class MOSSImageTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = Qwen3_5TextConfig(**{
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_output_gate": True,
            "bos_token_id": None,
            "dtype": "bfloat16",
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 12288,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 262144,
            "mlp_only_layers": [],
            "model_type": "qwen3_5_text",
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "num_attention_heads": 16,
            "num_hidden_layers": 32,
            "num_key_value_heads": 4,
            "pad_token_id": None,
            "partial_rotary_factor": 0.25,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
            },
            "tie_word_embeddings": False,
            "use_cache": True,
            "vocab_size": 248320,
        })
        self.model = Qwen3_5TextModel(config)
        self.model.rotary_emb = _MOSSImageRotaryEmbedding(config=config)
        self.config = config

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        # Qwen3.5 reads `conv1d.weight` directly instead of calling the module, so its
        # VRAM wrapper never gets a forward call to trigger lazy loading.
        for layer in self.model.layers:
            conv1d = getattr(getattr(layer, "linear_attn", None), "conv1d", None)
            if conv1d is not None and hasattr(conv1d, "preparing"):
                conv1d.preparing()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )
        return outputs.last_hidden_state
