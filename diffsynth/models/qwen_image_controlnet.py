import torch
import torch.nn as nn
from .qwen_image_dit import QwenEmbedRope, QwenImageTransformerBlock
from ..vram_management import gradient_checkpoint_forward
from einops import rearrange
from .sd3_dit import TimestepEmbeddings, RMSNorm


class QwenImageControlNet(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 60,
        num_controlnet_layers: int = 6,
    ):
        super().__init__()

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=[16,56,56], scale_rope=True) 

        self.time_text_embed = TimestepEmbeddings(256, 3072, diffusers_compatible_format=True, scale=1000, align_dtype_to_timestep=True)
        self.txt_norm = RMSNorm(3584, eps=1e-6)

        self.img_in = nn.Linear(64 * 2, 3072)
        self.txt_in = nn.Linear(3584, 3072)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=3072,
                    num_attention_heads=24,
                    attention_head_dim=128,
                )
                for _ in range(num_controlnet_layers)
            ]
        )
        self.proj_out = torch.nn.ModuleList([torch.nn.Linear(3072, 3072) for i in range(num_layers)])
        self.num_layers = num_layers
        self.num_controlnet_layers = num_controlnet_layers
        self.align_map = {i: i // (num_layers // num_controlnet_layers) for i in range(num_layers)}


    def forward(
        self,
        latents=None,
        timestep=None,
        prompt_emb=None,
        prompt_emb_mask=None,
        height=None,
        width=None,
        controlnet_conditioning=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        **kwargs,
    ):
        img_shapes = [(latents.shape[0], latents.shape[2]//2, latents.shape[3]//2)]
        txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()

        image = rearrange(latents, "B C (H P) (W Q) -> B (H W) (C P Q)", H=height//16, W=width//16, P=2, Q=2)
        controlnet_conditioning = rearrange(controlnet_conditioning, "B C (H P) (W Q) -> B (H W) (C P Q)", H=height//16, W=width//16, P=2, Q=2)
        image = torch.concat([image, controlnet_conditioning], dim=-1)

        image = self.img_in(image)
        text = self.txt_in(self.txt_norm(prompt_emb))

        conditioning = self.time_text_embed(timestep, image.dtype)

        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=latents.device)

        outputs = []
        for block in self.transformer_blocks:
            text, image = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                image=image,
                text=text,
                temb=conditioning,
                image_rotary_emb=image_rotary_emb,
            )
            outputs.append(image)

        outputs_aligned = [self.proj_out[i](outputs[self.align_map[i]]) for i in range(self.num_layers)]
        return outputs_aligned

    @staticmethod
    def state_dict_converter():
        return QwenImageControlNetStateDictConverter()



class QwenImageControlNetStateDictConverter():
    def __init__(self):
        pass

    def from_civitai(self, state_dict):
        return state_dict


class BlockWiseControlBlock(torch.nn.Module):
    # [linear, gelu, linear]
    def __init__(self, dim: int = 3072):
        super().__init__()
        self.x_rms = RMSNorm(dim, eps=1e-6)
        self.y_rms = RMSNorm(dim, eps=1e-6)
        self.input_proj = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.output_proj = nn.Linear(dim, dim)
    
    def forward(self, x, y):
        x, y = self.x_rms(x), self.y_rms(y)
        x = self.input_proj(x + y)
        x = self.act(x)
        x = self.output_proj(x)
        return x

    def init_weights(self):
        # zero initialize output_proj
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)


class QwenImageBlockWiseControlNet(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 60,
        in_dim: int = 64,
        dim: int = 3072,
    ):
        super().__init__()
        self.img_in = nn.Linear(in_dim, dim)
        self.controlnet_blocks = nn.ModuleList(
            [
                BlockWiseControlBlock(dim)
                for _ in range(num_layers)
            ]
        )

    def init_weight(self):
        nn.init.zeros_(self.img_in.weight)
        nn.init.zeros_(self.img_in.bias)
        for block in self.controlnet_blocks:
            block.init_weights()

    def process_controlnet_conditioning(self, controlnet_conditioning):
        return self.img_in(controlnet_conditioning)

    def blockwise_forward(self, img, controlnet_conditioning, block_id):
        return self.controlnet_blocks[block_id](img, controlnet_conditioning)

    @staticmethod
    def state_dict_converter():
        return QwenImageBlockWiseControlNetStateDictConverter()


class QwenImageBlockWiseControlNetStateDictConverter():
    def __init__(self):
        pass

    def from_civitai(self, state_dict):
        return state_dict
