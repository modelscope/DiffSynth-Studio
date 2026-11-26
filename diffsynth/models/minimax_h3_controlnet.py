import torch
import torch.nn as nn

from ..core.context_compress import KVCompressor, create_kv_cache_blocks
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
        # The control stream is re-based on the main branch's input embeddings at the first block.
        if self.block_id == 0:
            c = self.before_proj(c) + x
        c = super().forward(c, t_emb=t_emb, combined_indices=combined_indices, rope_freqs=rope_freqs, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return self.after_proj(c), c


def create_hint_cache(compressor: KVCompressor = None) -> dict:
    """The hint cache of one ControlNet, keyed by the main branch's block id.

    `create_kv_cache_blocks` builds one cache block per transformer block for the KV-cache use case.
    ControlNet hints are addressed by the block id they are added to, so a single cache block that
    holds one entry per control layer is enough. With a `compressor` the hints are stored
    compressed and decoded only when the main branch reads them; without one this is a plain dict.
    """
    return create_kv_cache_blocks(1, compressor)[0]


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
        control_scale=1.0,
        hint_cache=None,
    ) -> dict[int, torch.Tensor]:
        """Run the control stream and return `{block_id: hint * control_scale}`.

        With a `hint_cache` (see `create_hint_cache`) the hints are written there instead of being
        returned, each one as soon as its layer produces it, so the caller can drop the dense hint
        right away and read the (possibly compressed) hints back layer by layer.
        """
        expected = self.patch_columns()
        if control_rows.shape[-1] != expected:
            raise ValueError(f"control_rows carry {control_rows.shape[-1]} columns but control_in_dim={self.control_in_dim} expects {expected}")
        if control_rows.shape[0] != img_pos.shape[0]:
            raise ValueError(f"control_rows hold {control_rows.shape[0]} rows but the packed sequence has {img_pos.shape[0]} video rows; one control row per video row is required, in the same order")
        control_embeds = self.control_patch_proj(control_rows)
        c = embeddings.index_copy(0, img_pos, control_embeds.to(embeddings.dtype))

        keep = None
        if not self.control_apply_audio and audio_pos.numel():
            keep = torch.ones(c.shape[0], 1, dtype=c.dtype, device=c.device)
            keep[audio_pos] = 0

        hints = {}
        for layer_id, block in zip(self.control_layers, self.blocks):
            hint, c = gradient_checkpoint_forward(
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
            if keep is not None:
                hint = hint * keep
            hint = hint * control_scale
            if hint_cache is None:
                hints[layer_id] = hint
            else:
                hint_cache[layer_id] = hint
        return hints if hint_cache is None else hint_cache
