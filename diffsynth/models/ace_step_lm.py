import torch


LM_CONFIGS = {
    "acestep-5Hz-lm-0.6B": {
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "layer_types": ["full_attention"] * 28,
        "max_window_layers": 28,
    },
    "acestep-5Hz-lm-1.7B": {
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "layer_types": ["full_attention"] * 28,
        "max_window_layers": 28,
    },
    "acestep-5Hz-lm-4B": {
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "layer_types": ["full_attention"] * 36,
        "max_window_layers": 36,
    },
}


class AceStepLM(torch.nn.Module):
    """
    Language model for ACE-Step.

    Converts natural language prompts into structured parameters
    (caption, lyrics, bpm, keyscale, duration, timesignature, etc.)
    for ACE-Step music generation.

    Wraps a Qwen3ForCausalLM transformers model. Config is manually
    constructed based on variant type, and model weights are loaded
    via DiffSynth's standard mechanism from safetensors files.
    """

    def __init__(
        self,
        variant: str = "acestep-5Hz-lm-1.7B",
    ):
        super().__init__()
        from transformers import Qwen3Config, Qwen3ForCausalLM

        config_params = LM_CONFIGS[variant]

        config = Qwen3Config(
            attention_bias=False,
            attention_dropout=0.0,
            bos_token_id=151643,
            dtype="bfloat16",
            eos_token_id=151645,
            head_dim=128,
            hidden_act="silu",
            initializer_range=0.02,
            max_position_embeddings=40960,
            model_type="qwen3",
            num_key_value_heads=8,
            pad_token_id=151643,
            rms_norm_eps=1e-06,
            rope_scaling=None,
            rope_theta=1000000,
            sliding_window=None,
            tie_word_embeddings=True,
            use_cache=True,
            use_sliding_window=False,
            vocab_size=217204,
            **config_params,
        )

        self.model = Qwen3ForCausalLM(config)
        self.config = config
