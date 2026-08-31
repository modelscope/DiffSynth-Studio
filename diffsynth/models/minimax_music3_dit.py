import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.attention import attention_forward
from ..core.gradient import gradient_checkpoint_forward


class MiniMaxMusic3FourierEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(embedding_dim // 2, 1))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * math.pi * timestep.unsqueeze(-1) @ self.weight.T
        return torch.cat((angles.cos(), angles.sin()), dim=-1)


class MiniMaxMusic3TimestepEmbedding(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, out_dim)
        self.linear_2 = nn.Linear(out_dim, out_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(hidden_states)))


class MiniMaxMusic3RotaryEmbedding(nn.Module):

    def __init__(self, rotary_dim: int, theta: float = 10000.0):
        super().__init__()
        self.rotary_dim = rotary_dim
        self.theta = theta

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.rotary_dim, 2, device=device, dtype=torch.float32) / self.rotary_dim))
        steps = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(steps, inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs.cos().contiguous(), freqs.sin().contiguous()


def _apply_partial_rotary_emb(hidden_states: torch.Tensor, rotary_emb: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = rotary_emb
    rotary_dim = cos.shape[-1]
    cos = cos[:, None, :].to(hidden_states.dtype)
    sin = sin[:, None, :].to(hidden_states.dtype)
    rotated = hidden_states[..., :rotary_dim]
    half_first, half_second = rotated.chunk(2, dim=-1)
    rotate_half = torch.cat((-half_second, half_first), dim=-1)
    rotated = rotated * cos + rotate_half * sin
    return torch.cat((rotated, hidden_states[..., rotary_dim:]), dim=-1)


class MiniMaxMusic3Attention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner_dim = heads * head_dim
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=False)])

    def forward(self, hidden_states: torch.Tensor, rotary_emb: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query = self.to_q(hidden_states).view(batch_size, seq_len, self.heads, self.head_dim)
        key = self.to_k(hidden_states).view(batch_size, seq_len, self.heads, self.head_dim)
        value = self.to_v(hidden_states).view(batch_size, seq_len, self.heads, self.head_dim)
        query = _apply_partial_rotary_emb(query, rotary_emb)
        key = _apply_partial_rotary_emb(key, rotary_emb)
        hidden_states = attention_forward(
            query, key, value,
            q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        return self.to_out[0](hidden_states)


class MiniMaxMusic3TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, ff_inner_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MiniMaxMusic3Attention(dim, heads, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff_in = nn.Linear(dim, ff_inner_dim * 2)
        self.ff_out = nn.Linear(ff_inner_dim, dim)

    def forward(self, hidden_states: torch.Tensor, rotary_emb: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), rotary_emb)
        gate_states, gate = self.ff_in(self.norm2(hidden_states)).chunk(2, dim=-1)
        return hidden_states + self.ff_out(gate_states * F.silu(gate))


class MiniMaxMusic3DiT(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        condition_dim: int = 2048,
        num_layers: int = 36,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        ff_inner_dim: int = 8192,
        rotary_dim: int = 32,
        fourier_embedding_dim: int = 256,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        concat_channels = 2 * in_channels + condition_dim

        self.time_proj = MiniMaxMusic3FourierEmbedding(fourier_embedding_dim)
        self.time_embed = MiniMaxMusic3TimestepEmbedding(fourier_embedding_dim, inner_dim)
        self.preprocess_conv = nn.Conv1d(concat_channels, concat_channels, 1, bias=False)
        self.postprocess_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)
        self.proj_in = nn.Linear(concat_channels, inner_dim, bias=False)
        self.rotary_pos_emb = MiniMaxMusic3RotaryEmbedding(rotary_dim)
        self.transformer_blocks = nn.ModuleList([
            MiniMaxMusic3TransformerBlock(inner_dim, num_attention_heads, attention_head_dim, ff_inner_dim)
            for _ in range(num_layers)
        ])
        self.proj_out = nn.Linear(inner_dim, in_channels, bias=False)
        self.use_gradient_checkpointing = False
        self.use_gradient_checkpointing_offload = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        zeros = torch.zeros_like(hidden_states)
        hidden_states = torch.cat((hidden_states, zeros, encoder_hidden_states.transpose(1, 2)), dim=1)
        hidden_states = self.preprocess_conv(hidden_states) + hidden_states
        hidden_states = hidden_states.transpose(1, 2)

        temb = self.time_embed(self.time_proj(timestep))

        hidden_states = self.proj_in(hidden_states)
        hidden_states = torch.cat((temb.unsqueeze(1), hidden_states), dim=1)
        rotary_emb = self.rotary_pos_emb(hidden_states.shape[1], hidden_states.device)

        for block in self.transformer_blocks:
            hidden_states = gradient_checkpoint_forward(
                block,
                self.use_gradient_checkpointing,
                self.use_gradient_checkpointing_offload,
                hidden_states,
                rotary_emb,
            )

        hidden_states = self.proj_out(hidden_states[:, 1:])
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.postprocess_conv(hidden_states) + hidden_states
        return hidden_states
