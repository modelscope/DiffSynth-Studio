import torch


class SDTextEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=77,
        vocab_size=49408,
        layer_norm_eps=1e-05,
        hidden_act="quick_gelu",
        initializer_factor=1.0,
        initializer_range=0.02,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
        projection_dim=768,
    ):
        super().__init__()
        from transformers import CLIPConfig, CLIPTextModel

        config = CLIPConfig(
            text_config={
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "num_hidden_layers": num_hidden_layers,
                "num_attention_heads": num_attention_heads,
                "max_position_embeddings": max_position_embeddings,
                "vocab_size": vocab_size,
                "layer_norm_eps": layer_norm_eps,
                "hidden_act": hidden_act,
                "initializer_factor": initializer_factor,
                "initializer_range": initializer_range,
                "bos_token_id": bos_token_id,
                "eos_token_id": eos_token_id,
                "pad_token_id": pad_token_id,
                "projection_dim": projection_dim,
                "dropout": 0.0,
            },
            vision_config={
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "num_hidden_layers": num_hidden_layers,
                "num_attention_heads": num_attention_heads,
                "max_position_embeddings": max_position_embeddings,
                "layer_norm_eps": layer_norm_eps,
                "hidden_act": hidden_act,
                "initializer_factor": initializer_factor,
                "initializer_range": initializer_range,
                "projection_dim": projection_dim,
            },
            projection_dim=projection_dim,
        )
        self.model = CLIPTextModel(config.text_config)
        self.config = config

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        output_hidden_states=True,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        if output_hidden_states:
            return outputs.last_hidden_state, outputs.hidden_states
        return outputs.last_hidden_state
