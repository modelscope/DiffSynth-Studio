from .qwen_image_dit import QwenImageTransformerBlock, AdaLayerNorm, TimestepEmbeddings
from einops import rearrange
import torch



class QwenImageAccelerateAdapter(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 1,
    ):
        super().__init__()
        self.proj_latents_in = torch.nn.Linear(64, 3072)
        self.time_text_embed = TimestepEmbeddings(256, 3072, diffusers_compatible_format=True, scale=1000, align_dtype_to_timestep=True)
        self.transformer_blocks = torch.nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=3072,
                    num_attention_heads=24,
                    attention_head_dim=128,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = AdaLayerNorm(3072, single=True)
        self.proj_out = torch.nn.Linear(3072, 64)
        self.proj_latents_out = torch.nn.Linear(64, 64)

    def forward(
        self,
        latents=None,
        image=None,
        text=None,
        image_rotary_emb=None,
        timestep=None,
    ):
        latents = rearrange(latents, "B C (H P) (W Q) -> B (H W) (C P Q)", P=2, Q=2)
        image = image + self.proj_latents_in(latents)
        conditioning = self.time_text_embed(timestep, image.dtype)
        for block in self.transformer_blocks:
            text, image = block(
                image=image,
                text=text,
                temb=conditioning,
                image_rotary_emb=image_rotary_emb,
            )
        image = self.norm_out(image, conditioning)
        image = self.proj_out(image)
        image = image + self.proj_latents_out(latents)
        return image
    
    @staticmethod
    def state_dict_converter():
        return QwenImageAccelerateAdapterStateDictConverter()



class QwenImageAccelerateAdapterStateDictConverter():
    def __init__(self):
        pass

    def from_civitai(self, state_dict):
        return state_dict
