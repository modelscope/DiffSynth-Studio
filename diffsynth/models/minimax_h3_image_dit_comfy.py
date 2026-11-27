import types

import torch
from torch import nn

from ..core.gradient import gradient_checkpoint_forward
from .minimax_h3_dit import MINIMAX_H3_ADALN_MODALITY_NUM
from .minimax_h3_dit_comfy import MiniMaxH3ComfyPrunedTimeEmbedder, _comfy_attention_forward, _to_comfy_pruned_adaln_proj
from .minimax_h3_image_dit import MiniMaxH3ImageDiT, _parameter_dtype


class MiniMaxH3ImageDiTComfy(MiniMaxH3ImageDiT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for block in self.blocks:
            block.attn.forward = types.MethodType(_comfy_attention_forward, block.attn)
        for block in self.token_refiner.blocks:
            block.attn.forward = types.MethodType(_comfy_attention_forward, block.attn)


class MiniMaxH3ImageDiTComfyPruned(MiniMaxH3ImageDiTComfy):
    """Comfy-Org's AdaLN-pruned MiniMax-H3-Image, with the curve-form AdaLN table and the fused PDD head bank.

    The AdaLN projections take an 8-wide coordinate read off a shared 1025-row curve table instead of a full
    timestep embedding, and `final_layer.video_out` stacks the six per-step heads in one affine layer: rows of block
    0 are the first head, later blocks are offsets from it, so step `i` uses `head_0` for `i == 0` and
    `head_0 + head_i` afterwards.
    """

    def __init__(self, adaln_curve_grid: int = 1025, time_embed_dim: int = 8, pdd_steps: int = 6, **kwargs):
        super().__init__(time_embed_dim=time_embed_dim, **kwargs)
        self.adaln_t_table = nn.Parameter(torch.empty(adaln_curve_grid, time_embed_dim, dtype=torch.float32), requires_grad=False)
        self.time_embedder = MiniMaxH3ComfyPrunedTimeEmbedder(lambda: self.adaln_t_table)
        for block in self.blocks:
            block.adaln_proj = _to_comfy_pruned_adaln_proj(block.adaln_proj)
        self.final_layer.adaln_proj = _to_comfy_pruned_adaln_proj(self.final_layer.adaln_proj)
        patch_dim = self.final_layer.video_out.out_features
        self.final_layer.video_out = nn.Linear(self.hidden_size, patch_dim * pdd_steps)
        self.pdd_steps = pdd_steps

    def forward(
        self,
        x,
        prompt_embeds,
        position_ids,
        unique_timesteps,
        timestep_indices,
        token_tags,
        img_pos,
        text_pos,
        pdd_heads=None,
        pdd_step_index: int = 0,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> torch.Tensor:
        """Same contract as `MiniMaxH3ImageDiT.forward`; `pdd_heads` is ignored because the heads are fused."""
        device = x.device
        seq_len = int(position_ids.shape[1])
        img_pos = img_pos.view(-1).to(device=device, dtype=torch.long)
        text_pos = text_pos.view(-1).to(device=device, dtype=torch.long)
        timestep_indices = timestep_indices.view(-1).to(device=device, dtype=torch.long)
        token_tags = token_tags.view(-1).to(device=device, dtype=torch.long)
        unique_timesteps = unique_timesteps.view(-1).to(device=device)

        rope_freqs = self.rope(position_ids.to(device))
        hidden = self._embed(
            x=x, prompt_embeds=prompt_embeds, seq_len=seq_len,
            img_pos=img_pos, text_pos=text_pos, device=device,
        )

        t_emb = self.time_embedder(unique_timesteps, dtype=hidden.dtype)

        combined_indices = (timestep_indices * MINIMAX_H3_ADALN_MODALITY_NUM + token_tags.clamp(min=0)).to(device)
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

        for block in self.blocks:
            hidden = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                hidden,
                t_emb=t_emb,
                combined_indices=combined_indices,
                rope_freqs=rope_freqs,
                cu_seqlens=cu_seqlens,
                max_seqlen=seq_len,
            )

        hidden = self.final_layer(hidden, t_emb=t_emb, timestep_indices=timestep_indices)

        heads = self.final_layer.video_out(hidden.to(_parameter_dtype(self.final_layer.video_out)))
        heads = heads.view(heads.shape[0], self.pdd_steps, -1)
        output = heads[:, 0] if pdd_step_index == 0 else heads[:, 0] + heads[:, pdd_step_index]
        return output.index_select(0, img_pos)
