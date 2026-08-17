from __future__ import annotations

import torch

from diffsynth.models.ltx25_diffvae.model.video_vae.transformer.blocks import DiffusionNABlock
from diffsynth.models.ltx25_diffvae.model.video_vae.transformer.combined.attn import full as residual_attn
from diffsynth.models.ltx25_diffvae.model.video_vae.transformer.combined.context import combined as inject_context
from diffsynth.models.ltx25_diffvae.model.video_vae.transformer.combined.mlp import residual_mlp


class CombinedDiffusionNABlock(DiffusionNABlock):
    def forward_combined(
        self,
        context_and_x: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        scale_msa, shift_msa, scale_mlp, shift_mlp = self._modulation(modulation)
        x = inject_context(context_and_x, self.context_proj.weight, self.context_proj.bias)
        x = residual_attn(x, self.attn, self.norm1, scale_msa, shift_msa)
        x = residual_mlp(x, self.mlp, self.norm2, scale_mlp, shift_mlp)
        return x

    def forward(
        self,
        context_and_x: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return self.forward_combined(context_and_x, modulation)
