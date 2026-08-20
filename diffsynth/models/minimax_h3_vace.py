import torch
import torch.nn as nn
from .minimax_h3_dit import MiniMaxH3DiTBlock, MINIMAX_H3_ADALN_MODALITY_NUM
from ..core.gradient import gradient_checkpoint_forward


class MiniMaxH3VaceBlock(MiniMaxH3DiTBlock):
    def __init__(self, hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size, time_embed_dim, adaln_out_features, norm_eps, qk_norm_eps, block_id=0):
        super().__init__(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size, time_embed_dim, adaln_out_features, norm_eps, qk_norm_eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(hidden_size, hidden_size)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, *, t_emb, combined_indices, rope_freqs, cu_seqlens, max_seqlen):
        if self.block_id == 0:
            c = self.before_proj(c) + x
        c = super().forward(c, t_emb=t_emb, combined_indices=combined_indices, rope_freqs=rope_freqs, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return self.after_proj(c), c


class MiniMaxH3VaceModel(nn.Module):
    def __init__(
        self,
        vace_layers=(0, 7, 14, 21, 28, 35, 42, 49),
        vace_in_dim=96,
        hidden_size=5376,
        num_attention_heads=56,
        attention_head_dim=128,
        ffn_hidden_size=14336,
        time_embed_dim=2688,
        adaln_out_features=96768,
        norm_eps=1e-5,
        qk_norm_eps=1e-5,
    ):
        super().__init__()
        self.vace_layers = sorted(vace_layers)
        self.vace_in_dim = vace_in_dim
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # vace blocks
        self.vace_blocks = nn.ModuleList([
            MiniMaxH3VaceBlock(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size, time_embed_dim, adaln_out_features, norm_eps, qk_norm_eps, block_id=i)
            for i in self.vace_layers
        ])

        # vace patch embedding
        self.vace_patch_embedding = nn.Linear(vace_in_dim, hidden_size, bias=True)

    def init_from_dit(self, dit):
        """Warm-start the VACE blocks from the corresponding backbone blocks."""
        for layer_id, block in zip(self.vace_layers, self.vace_blocks):
            state_dict = dit.blocks[layer_id].state_dict()
            block.load_state_dict(state_dict, strict=False)

    def forward(
        self,
        x, vace_context, t_emb, combined_indices, rope_freqs, img_pos,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ):
        # The control tokens share the target video's positions in the packed
        # sequence, so RoPE frequencies and AdaLN indices are gathered from there.
        c = self.vace_patch_embedding(vace_context)
        ctrl_pos = img_pos[:c.shape[0]]
        combined_indices = combined_indices[ctrl_pos]
        rope_freqs = rope_freqs[ctrl_pos]
        cu_seqlens = torch.tensor([0, c.shape[0]], dtype=torch.int32, device=c.device)
        max_seqlen = c.shape[0]

        x = x[ctrl_pos]
        hints = []
        for block in self.vace_blocks:
            hint, c = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                c, x,
                t_emb=t_emb,
                combined_indices=combined_indices,
                rope_freqs=rope_freqs,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            hints.append(hint)
        return hints
