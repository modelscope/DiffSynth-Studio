import torch
import torch.nn as nn
from .wan_video_dit import WanModel, precompute_freqs_cis, sinusoidal_embedding_1d
from einops import rearrange
from ..core import gradient_checkpoint_forward

def precompute_freqs_cis_1d(dim: int, end: int = 16384, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim, end, theta)
    return f_freqs_cis.chunk(3, dim=-1)

class MovaAudioDit(WanModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        head_dim = kwargs.get("dim", 1536) // kwargs.get("num_heads", 12)
        self.freqs = precompute_freqs_cis_1d(head_dim)
        self.patch_embedding = nn.Conv1d(
            kwargs.get("in_dim", 128), kwargs.get("dim", 1536), kernel_size=[1], stride=[1]
        )

    def precompute_freqs_cis(self, dim: int, end: int = 16384, theta: float = 10000.0):
        self.f_freqs_cis = precompute_freqs_cis_1d(dim, end, theta)

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        x, (f, ) = self.patchify(x)
        freqs = torch.cat([
            self.freqs[0][:f].view(f, -1).expand(f, -1),
            self.freqs[1][:f].view(f, -1).expand(f, -1),
            self.freqs[2][:f].view(f, -1).expand(f, -1),
        ], dim=-1).reshape(f, 1, -1).to(x.device)

        for block in self.blocks:
            x = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x, context, t_mod, freqs,
            )
        x = self.head(x, t)
        x = self.unpatchify(x, (f, ))
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b f (p c) -> b c (f p)',
            f=grid_size[0],
            p=self.patch_size[0]
        )
