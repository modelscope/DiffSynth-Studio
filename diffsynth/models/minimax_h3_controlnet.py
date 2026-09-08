import torch
import torch.nn as nn

from ..core.gradient import gradient_checkpoint_forward
from .minimax_h3_dit import MiniMaxH3DiTBlock, _PATCH_H, _PATCH_T, _PATCH_W


class MiniMaxH3ControlNetBlock(MiniMaxH3DiTBlock):
    def __init__(self, hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size, time_embed_dim, adaln_out_features, norm_eps, qk_norm_eps, block_id=0):
        super().__init__(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size, time_embed_dim, adaln_out_features, norm_eps, qk_norm_eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(hidden_size, hidden_size)
        self.after_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, c, x, *, t_emb, combined_indices, rope_freqs, cu_seqlens, max_seqlen):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = super().forward(c, t_emb=t_emb, combined_indices=combined_indices, rope_freqs=rope_freqs, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        return torch.stack(all_c)


class MiniMaxH3ControlNet(nn.Module):
    _repeated_blocks = ["MiniMaxH3ControlNetBlock"]

    def __init__(
        self,
        control_layers: tuple = (0, 10, 20, 30, 40),
        control_in_dim: int = 49,
        control_apply_audio: bool = False,
        hidden_size: int = 5376,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        ffn_hidden_size: int = 14336,
        time_embed_dim: int = 2688,
        adaln_out_features: int = 96768,
        patch_size: tuple = (1, 2, 2),
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        self.control_layers = tuple(sorted({int(i) for i in control_layers}))
        if not self.control_layers or self.control_layers[0] != 0:
            raise ValueError(f"control_layers must start at layer 0 so the control stream is re-based on the main branch's input embeddings, got {self.control_layers}")
        self.control_in_dim = control_in_dim
        self.control_apply_audio = control_apply_audio
        self.control_layers_mapping = {i: n for n, i in enumerate(self.control_layers)}
        self.blocks = nn.ModuleList([
            MiniMaxH3ControlNetBlock(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size, time_embed_dim, adaln_out_features, norm_eps, qk_norm_eps, block_id=n)
            for n in range(len(self.control_layers))
        ])
        control_patch_dim = control_in_dim * patch_size[0] * patch_size[1] * patch_size[2]
        self.control_patch_proj = nn.Linear(control_patch_dim, hidden_size, bias=True)

    def patch_columns(self):
        return self.control_in_dim * _PATCH_T * _PATCH_H * _PATCH_W

    def forward(
        self,
        embeddings,
        control_rows,
        img_pos,
        audio_pos,
        *,
        t_emb,
        combined_indices,
        rope_freqs,
        cu_seqlens,
        max_seqlen,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        expected = self.patch_columns()
        if control_rows.shape[-1] != expected:
            raise ValueError(f"control_rows carry {control_rows.shape[-1]} columns but control_in_dim={self.control_in_dim} expects {expected}")
        if control_rows.shape[0] != img_pos.shape[0]:
            raise ValueError(f"control_rows hold {control_rows.shape[0]} rows but the packed sequence has {img_pos.shape[0]} video rows; one control row per video row is required, in the same order")
        control_embeds = self.control_patch_proj(control_rows.to(self.control_patch_proj.weight.dtype))
        c = embeddings.index_copy(0, img_pos, control_embeds.to(embeddings.dtype))
        for block in self.blocks:
            c = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                c,
                embeddings,
                t_emb=t_emb,
                combined_indices=combined_indices,
                rope_freqs=rope_freqs,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        hints = torch.unbind(c)[:-1]
        if not self.control_apply_audio and audio_pos.numel():
            keep = torch.ones(embeddings.shape[0], dtype=hints[0].dtype, device=hints[0].device)
            keep[audio_pos] = 0
            hints = tuple(hint * keep.unsqueeze(-1) for hint in hints)
        return hints
