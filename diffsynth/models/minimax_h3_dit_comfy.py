import types

import torch
from torch import nn

from .minimax_h3_dit import MiniMaxH3AdalnProj, MiniMaxH3DiT, _apply_rope, _sdpa_varlen_attention


class MiniMaxH3ComfyPrunedAdalnProj(MiniMaxH3AdalnProj):
    def forward(self, t_emb):
        x = self.linear(t_emb)
        m = x.shape[0]
        x = x.view(m * self.modality_num, self.expand_ratio * self.hidden_size)
        return tuple(x.chunk(self.expand_ratio, dim=-1))


def _comfy_attention_forward(self, x, *, rope_freqs, cu_seqlens, max_seqlen=None):
    total = x.shape[0]
    qkv = self.qkv_proj(x)
    qkv = qkv.view(total, 3, self.num_heads, self.head_dim)
    q = qkv[:, 0, :, :]
    k = qkv[:, 1, :, :]
    v = qkv[:, 2, :, :]
    q = self.q_norm(q)
    k = self.k_norm(k)
    if rope_freqs is not None:
        q = _apply_rope(q, rope_freqs)
        k = _apply_rope(k, rope_freqs)
    out = _sdpa_varlen_attention(q, k, v, cu_seqlens=cu_seqlens, softmax_scale=self.softmax_scale)
    out = out.reshape(total, self.num_heads * self.head_dim)
    return self.out_proj(out)


class MiniMaxH3ComfyPrunedTimeEmbedder(nn.Module):
    def __init__(self, table_getter):
        super().__init__()
        self._table_getter = table_getter

    def forward(self, t: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        table = self._table_getter().to(device=t.device, dtype=torch.float32)
        pos = t.to(torch.float32).clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        weight = (pos - i0).unsqueeze(1)
        t_emb = torch.lerp(table[i0], table[i0 + 1], weight)
        return t_emb.to(dtype)


def _to_comfy_pruned_adaln_proj(proj: MiniMaxH3AdalnProj) -> MiniMaxH3ComfyPrunedAdalnProj:
    curve_proj = MiniMaxH3ComfyPrunedAdalnProj(
        proj.hidden_size,
        proj.linear.in_features,
        proj.linear.out_features,
        expand_ratio=proj.expand_ratio,
        modality_num=proj.modality_num,
    )
    curve_proj.linear = proj.linear
    return curve_proj


class MiniMaxH3DiTComfy(MiniMaxH3DiT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for block in self.blocks:
            block.attn.forward = types.MethodType(_comfy_attention_forward, block.attn)
        for block in self.token_refiner.blocks:
            block.attn.forward = types.MethodType(_comfy_attention_forward, block.attn)


class MiniMaxH3DiTComfyPruned(MiniMaxH3DiTComfy):
    def __init__(self, adaln_curve_grid: int = 1025, time_embed_dim: int = 8, **kwargs):
        super().__init__(time_embed_dim=time_embed_dim, **kwargs)
        self.register_buffer(
            "adaln_t_table", torch.empty(adaln_curve_grid, time_embed_dim, dtype=torch.float32)
        )
        self.time_embedder = MiniMaxH3ComfyPrunedTimeEmbedder(lambda: self.adaln_t_table)
        for block in self.blocks:
            block.adaln_proj = _to_comfy_pruned_adaln_proj(block.adaln_proj)
        self.final_layer.adaln_proj = _to_comfy_pruned_adaln_proj(self.final_layer.adaln_proj)
