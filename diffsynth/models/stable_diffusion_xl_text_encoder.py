import torch
from .stable_diffusion_text_encoder import CLIPTextModel, CLIPTextTransformer


class CLIPTextModelWithProjection(torch.nn.Module):
    def __init__(self, hidden_size=1280, intermediate_size=5120, num_hidden_layers=32,
                 num_attention_heads=20, max_position_embeddings=77, vocab_size=49408,
                 layer_norm_eps=1e-05, hidden_act="gelu", projection_dim=1280):
        super().__init__()
        self.text_model = CLIPTextTransformer(
            hidden_size=hidden_size, intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings, vocab_size=vocab_size,
            layer_norm_eps=layer_norm_eps, hidden_act=hidden_act,
        )
        self.text_projection = torch.nn.Linear(hidden_size, projection_dim, bias=False)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, output_hidden_states=False):
        last_hidden_state, all_hidden_states = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, output_hidden_states=output_hidden_states,
        )
        text_embeds = self.text_projection(last_hidden_state[:, 0, :])
        return text_embeds, last_hidden_state, all_hidden_states


class SDXLTextEncoder2(torch.nn.Module):
    def __init__(
        self,
        hidden_size=1280,
        intermediate_size=5120,
        num_hidden_layers=32,
        num_attention_heads=20,
        max_position_embeddings=77,
        vocab_size=49408,
        layer_norm_eps=1e-05,
        hidden_act="gelu",
        projection_dim=1280,
    ):
        super().__init__()
        self.model = CLIPTextModelWithProjection(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            vocab_size=vocab_size,
            layer_norm_eps=layer_norm_eps,
            hidden_act=hidden_act,
            projection_dim=projection_dim,
        )
        self.config = type("Config", (), {"projection_dim": projection_dim})()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        output_hidden_states=True,
    ):
        text_embeds, last_hidden_state, all_hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )
        if output_hidden_states:
            return text_embeds, all_hidden_states
        return text_embeds
