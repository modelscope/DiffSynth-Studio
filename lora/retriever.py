import torch
from diffsynth import SDTextEncoder
from diffsynth.models.sd3_text_encoder import SD3TextEncoder1StateDictConverter
from diffsynth.models.sd_text_encoder import CLIPEncoderLayer


class LoRALayerBlock(torch.nn.Module):
    def __init__(self, L, dim_in):
        super().__init__()
        self.x = torch.nn.Parameter(torch.randn(1, L, dim_in))

    def forward(self, lora_A, lora_B):
        out = self.x @ lora_A.T @ lora_B.T
        return out
    

class LoRAEmbedder(torch.nn.Module):
    def __init__(self, lora_patterns=None, L=1, out_dim=2048):
        super().__init__()
        if lora_patterns is None:
            lora_patterns = self.default_lora_patterns()
            
        model_dict = {}
        for lora_pattern in lora_patterns:
            name, dim = lora_pattern["name"], lora_pattern["dim"][0]
            model_dict[name.replace(".", "___")] = LoRALayerBlock(L, dim)
        self.model_dict = torch.nn.ModuleDict(model_dict)
        
        proj_dict = {}
        for lora_pattern in lora_patterns:
            layer_type, dim = lora_pattern["type"], lora_pattern["dim"][1]
            if layer_type not in proj_dict:
                proj_dict[layer_type.replace(".", "___")] = torch.nn.Linear(dim, out_dim)
        self.proj_dict = torch.nn.ModuleDict(proj_dict)
        
        self.lora_patterns = lora_patterns
        
        
    def default_lora_patterns(self):
        lora_patterns = []
        lora_dict = {
            "attn.a_to_qkv": (3072, 9216), "attn.a_to_out": (3072, 3072), "ff_a.0": (3072, 12288), "ff_a.2": (12288, 3072), "norm1_a.linear": (3072, 18432),
            "attn.b_to_qkv": (3072, 9216), "attn.b_to_out": (3072, 3072), "ff_b.0": (3072, 12288), "ff_b.2": (12288, 3072), "norm1_b.linear": (3072, 18432),
        }
        for i in range(19):
            for suffix in lora_dict:
                lora_patterns.append({
                    "name": f"blocks.{i}.{suffix}",
                    "dim": lora_dict[suffix],
                    "type": suffix,
                })
        lora_dict = {"to_qkv_mlp": (3072, 21504), "proj_out": (15360, 3072), "norm.linear": (3072, 9216)}
        for i in range(38):
            for suffix in lora_dict:
                lora_patterns.append({
                    "name": f"single_blocks.{i}.{suffix}",
                    "dim": lora_dict[suffix],
                    "type": suffix,
                })
        return lora_patterns
        
    def forward(self, lora):
        lora_emb = []
        for lora_pattern in self.lora_patterns:
            name, layer_type = lora_pattern["name"], lora_pattern["type"]
            lora_A = lora[name + ".lora_A.default.weight"]
            lora_B = lora[name + ".lora_B.default.weight"]
            lora_out = self.model_dict[name.replace(".", "___")](lora_A, lora_B)
            lora_out = self.proj_dict[layer_type.replace(".", "___")](lora_out)
            lora_emb.append(lora_out)
        lora_emb = torch.concat(lora_emb, dim=1)
        return lora_emb


class TextEncoder(torch.nn.Module):
    def __init__(self, embed_dim=768, vocab_size=49408, max_position_embeddings=77, num_encoder_layers=12, encoder_intermediate_size=3072):
        super().__init__()

        # token_embedding
        self.token_embedding = torch.nn.Embedding(vocab_size, embed_dim)

        # position_embeds (This is a fixed tensor)
        self.position_embeds = torch.nn.Parameter(torch.zeros(1, max_position_embeddings, embed_dim))

        # encoders
        self.encoders = torch.nn.ModuleList([CLIPEncoderLayer(embed_dim, encoder_intermediate_size) for _ in range(num_encoder_layers)])

        # attn_mask
        self.attn_mask = self.attention_mask(max_position_embeddings)

        # final_layer_norm
        self.final_layer_norm = torch.nn.LayerNorm(embed_dim)

    def attention_mask(self, length):
        mask = torch.empty(length, length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, input_ids, clip_skip=1):
        embeds = self.token_embedding(input_ids) + self.position_embeds
        attn_mask = self.attn_mask.to(device=embeds.device, dtype=embeds.dtype)
        for encoder_id, encoder in enumerate(self.encoders):
            embeds = encoder(embeds, attn_mask=attn_mask)
            if encoder_id + clip_skip == len(self.encoders):
                break
        embeds = self.final_layer_norm(embeds)
        pooled_embeds = embeds[torch.arange(embeds.shape[0]), input_ids.to(dtype=torch.int).argmax(dim=-1)]
        return pooled_embeds
    
    @staticmethod
    def state_dict_converter():
        return SD3TextEncoder1StateDictConverter()
    
    
class LoRAEncoder(torch.nn.Module):
    def __init__(self, embed_dim=768, max_position_embeddings=304, num_encoder_layers=2, encoder_intermediate_size=3072, L=1):
        super().__init__()
        max_position_embeddings *= L
        
        # Embedder
        self.embedder = LoRAEmbedder(L=L, out_dim=embed_dim)

        # position_embeds (This is a fixed tensor)
        self.position_embeds = torch.nn.Parameter(torch.zeros(1, max_position_embeddings, embed_dim))

        # encoders
        self.encoders = torch.nn.ModuleList([CLIPEncoderLayer(embed_dim, encoder_intermediate_size) for _ in range(num_encoder_layers)])

        # attn_mask
        self.attn_mask = self.attention_mask(max_position_embeddings)

        # final_layer_norm
        self.final_layer_norm = torch.nn.LayerNorm(embed_dim)

    def attention_mask(self, length):
        mask = torch.empty(length, length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, lora):
        embeds = self.embedder(lora) + self.position_embeds
        attn_mask = self.attn_mask.to(device=embeds.device, dtype=embeds.dtype)
        for encoder_id, encoder in enumerate(self.encoders):
            embeds = encoder(embeds, attn_mask=attn_mask)
        embeds = self.final_layer_norm(embeds)
        embeds = embeds.mean(dim=1)
        return embeds
