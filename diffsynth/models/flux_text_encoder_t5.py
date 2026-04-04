import torch
from transformers import T5EncoderModel, T5Config


class FluxTextEncoderT5(T5EncoderModel):
    def __init__(self):
        config = T5Config(**{
            "architectures": [
                "T5EncoderModel"
            ],
            "classifier_dropout": 0.0,
            "d_ff": 10240,
            "d_kv": 64,
            "d_model": 4096,
            "decoder_start_token_id": 0,
            "dense_act_fn": "gelu_new",
            "dropout_rate": 0.1,
            "dtype": "bfloat16",
            "eos_token_id": 1,
            "feed_forward_proj": "gated-gelu",
            "initializer_factor": 1.0,
            "is_encoder_decoder": True,
            "is_gated_act": True,
            "layer_norm_epsilon": 1e-06,
            "model_type": "t5",
            "num_decoder_layers": 24,
            "num_heads": 64,
            "num_layers": 24,
            "output_past": True,
            "pad_token_id": 0,
            "relative_attention_max_distance": 128,
            "relative_attention_num_buckets": 32,
            "tie_word_embeddings": False,
            "transformers_version": "4.57.1",
            "use_cache": True,
            "vocab_size": 32128
        })
        super().__init__(config)

    def forward(self, input_ids):
        outputs = super().forward(input_ids=input_ids)
        prompt_emb = outputs.last_hidden_state
        return prompt_emb
