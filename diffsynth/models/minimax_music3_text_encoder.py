import torch
import torch.nn as nn


class MiniMaxMusic3TextEncoder(nn.Module):

    def __init__(
        self,
        vocab_size: int = 200000,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        max_position_embeddings: int = 10240,
        rope_theta: float = 1000000,
        rms_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = False,
    ):
        super().__init__()
        from transformers import Qwen3Config, Qwen3ForCausalLM

        config = Qwen3Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
        )
        self.model = Qwen3ForCausalLM(config)
        self.config = config

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        past_key_values=None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ):
        return self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
