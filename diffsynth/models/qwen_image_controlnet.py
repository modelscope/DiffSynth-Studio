import torch
import torch.nn as nn
from .general_modules import RMSNorm


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
        additional_in_dim: int = 0,
        dim: int = 3072,
    ):
        super().__init__()
        self.img_in = nn.Linear(in_dim + additional_in_dim, dim)
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
