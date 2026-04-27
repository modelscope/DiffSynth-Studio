import torch


class AceStepTextEncoder(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        from transformers import Qwen3Config, Qwen3Model

        config = Qwen3Config(
            attention_bias=False,
            attention_dropout=0.0,
            bos_token_id=151643,
            dtype="bfloat16",
            eos_token_id=151643,
            head_dim=128,
            hidden_act="silu",
            hidden_size=1024,
            initializer_range=0.02,
            intermediate_size=3072,
            layer_types=["full_attention"] * 28,
            max_position_embeddings=32768,
            max_window_layers=28,
            model_type="qwen3",
            num_attention_heads=16,
            num_hidden_layers=28,
            num_key_value_heads=8,
            pad_token_id=151643,
            rms_norm_eps=1e-06,
            rope_scaling=None,
            rope_theta=1000000,
            sliding_window=None,
            tie_word_embeddings=True,
            use_cache=True,
            use_sliding_window=False,
            vocab_size=151669,
        )

        self.model = Qwen3Model(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state
