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


class MiniMaxMusic3LayerNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.eps = eps
        self.dim = dim

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(hidden_states, (self.dim,), self.gamma, self.beta, self.eps)


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
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor, rotary_emb: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query, key, value = self.to_qkv(hidden_states).chunk(3, dim=-1)
        query = query.view(batch_size, seq_len, self.heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.heads, self.head_dim)
        query = _apply_partial_rotary_emb(query, rotary_emb)
        key = _apply_partial_rotary_emb(key, rotary_emb)
        hidden_states = attention_forward(
            query, key, value,
            q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        return self.to_out(hidden_states)


class MiniMaxMusic3SwiGLU(nn.Module):
    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, inner_dim * 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return gate_states * F.silu(gate)


class MiniMaxMusic3FeedForward(nn.Module):
    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        self.ff = nn.Sequential(
            MiniMaxMusic3SwiGLU(dim, inner_dim),
            nn.Dropout(0.0),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.ff(hidden_states)


class MiniMaxMusic3TransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, ff_inner_dim: int):
        super().__init__()
        self.pre_norm = MiniMaxMusic3LayerNorm(dim)
        self.self_attn = MiniMaxMusic3Attention(dim, heads, head_dim)
        self.ff_norm = MiniMaxMusic3LayerNorm(dim)
        self.ff = MiniMaxMusic3FeedForward(dim, ff_inner_dim)

    def forward(self, hidden_states: torch.Tensor, rotary_emb: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.pre_norm(hidden_states), rotary_emb)
        hidden_states = hidden_states + self.ff(self.ff_norm(hidden_states))
        return hidden_states


class MiniMaxMusic3Transformer(nn.Module):
    def __init__(self, inner_dim, concat_channels, in_channels, num_layers, num_attention_heads, attention_head_dim, ff_inner_dim, rotary_dim):
        super().__init__()
        self.project_in = nn.Linear(concat_channels, inner_dim, bias=False)
        self.rotary_pos_emb = MiniMaxMusic3RotaryEmbedding(rotary_dim)
        self.layers = nn.ModuleList([
            MiniMaxMusic3TransformerLayer(inner_dim, num_attention_heads, attention_head_dim, ff_inner_dim)
            for _ in range(num_layers)
        ])
        self.project_out = nn.Linear(inner_dim, in_channels, bias=False)


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

        self.timestep_features = MiniMaxMusic3FourierEmbedding(fourier_embedding_dim)
        self.to_timestep_embed = nn.Sequential(
            nn.Linear(fourier_embedding_dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, inner_dim),
        )
        self.preprocess_conv = nn.Conv1d(concat_channels, concat_channels, 1, bias=False)
        self.postprocess_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)
        self.transformer = MiniMaxMusic3Transformer(
            inner_dim, concat_channels, in_channels, num_layers,
            num_attention_heads, attention_head_dim, ff_inner_dim, rotary_dim,
        )
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

        temb = self.to_timestep_embed(self.timestep_features(timestep))

        hidden_states = self.transformer.project_in(hidden_states)
        hidden_states = torch.cat((temb.unsqueeze(1), hidden_states), dim=1)
        rotary_emb = self.transformer.rotary_pos_emb(hidden_states.shape[1], hidden_states.device)

        for layer in self.transformer.layers:
            hidden_states = gradient_checkpoint_forward(
                layer,
                self.use_gradient_checkpointing,
                self.use_gradient_checkpointing_offload,
                hidden_states,
                rotary_emb,
            )

        hidden_states = self.transformer.project_out(hidden_states[:, 1:])
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.postprocess_conv(hidden_states) + hidden_states
        return hidden_states
