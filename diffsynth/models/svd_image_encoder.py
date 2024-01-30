import torch
from .sd_text_encoder import CLIPEncoderLayer


class CLIPVisionEmbeddings(torch.nn.Module):
    def __init__(self, embed_dim=1280, image_size=224, patch_size=14, num_channels=3):
        super().__init__()

        # class_embeds (This is a fixed tensor)
        self.class_embedding = torch.nn.Parameter(torch.randn(1, 1, embed_dim))

        # position_embeds
        self.patch_embedding = torch.nn.Conv2d(in_channels=num_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

        # position_embeds (This is a fixed tensor)
        self.position_embeds = torch.nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2 + 1, embed_dim))

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.repeat(batch_size, 1, 1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1) + self.position_embeds
        return embeddings


class SVDImageEncoder(torch.nn.Module):
    def __init__(self, embed_dim=1280, layer_norm_eps=1e-5, num_encoder_layers=32, encoder_intermediate_size=5120, projection_dim=1024):
        super().__init__()
        self.embeddings = CLIPVisionEmbeddings(embed_dim=embed_dim)
        self.pre_layernorm = torch.nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.encoders = torch.nn.ModuleList([CLIPEncoderLayer(embed_dim, encoder_intermediate_size, num_heads=16, head_dim=80, use_quick_gelu=False) for _ in range(num_encoder_layers)])
        self.post_layernorm = torch.nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.visual_projection = torch.nn.Linear(embed_dim, projection_dim, bias=False)

    def forward(self, pixel_values):
        embeds = self.embeddings(pixel_values)
        embeds = self.pre_layernorm(embeds)
        for encoder_id, encoder in enumerate(self.encoders):
            embeds = encoder(embeds)
        embeds = self.post_layernorm(embeds[:, 0, :])
        embeds = self.visual_projection(embeds)
        return embeds

    def state_dict_converter(self):
        return SVDImageEncoderStateDictConverter()


class SVDImageEncoderStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "vision_model.embeddings.patch_embedding.weight": "embeddings.patch_embedding.weight",
            "vision_model.embeddings.class_embedding": "embeddings.class_embedding",
            "vision_model.embeddings.position_embedding.weight": "embeddings.position_embeds",
            "vision_model.pre_layrnorm.weight": "pre_layernorm.weight",
            "vision_model.pre_layrnorm.bias": "pre_layernorm.bias",
            "vision_model.post_layernorm.weight": "post_layernorm.weight",
            "vision_model.post_layernorm.bias": "post_layernorm.bias",
            "visual_projection.weight": "visual_projection.weight"
        }
        attn_rename_dict = {
            "self_attn.q_proj": "attn.to_q",
            "self_attn.k_proj": "attn.to_k",
            "self_attn.v_proj": "attn.to_v",
            "self_attn.out_proj": "attn.to_out",
            "layer_norm1": "layer_norm1",
            "layer_norm2": "layer_norm2",
            "mlp.fc1": "fc1",
            "mlp.fc2": "fc2",
        }
        state_dict_ = {}
        for name in state_dict:
            if name in rename_dict:
                param = state_dict[name]
                if name == "vision_model.embeddings.class_embedding":
                    param = state_dict[name].view(1, 1, -1)
                elif name == "vision_model.embeddings.position_embedding.weight":
                    param = state_dict[name].view(1, 257, 1280)
                state_dict_[rename_dict[name]] = param
            elif name.startswith("vision_model.encoder.layers."):
                param = state_dict[name]
                names = name.split(".")
                layer_id, layer_type, tail = names[3], ".".join(names[4:-1]), names[-1]
                name_ = ".".join(["encoders", layer_id, attn_rename_dict[layer_type], tail])
                state_dict_[name_] = param
        return state_dict_
