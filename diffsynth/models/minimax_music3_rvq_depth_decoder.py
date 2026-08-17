import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.attention import attention_forward


class MiniMaxMusic3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class MiniMaxMusic3DepthAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query = self.to_q(hidden_states).view(batch_size, seq_len, self.heads, self.head_dim)
        key = self.to_k(hidden_states).view(batch_size, seq_len, self.heads, self.head_dim)
        value = self.to_v(hidden_states).view(batch_size, seq_len, self.heads, self.head_dim)
        hidden_states = attention_forward(
            query, key, value,
            q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
            is_causal=True,
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        return self.to_out(hidden_states)


class MiniMaxMusic3DepthDecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, intermediate_size: int):
        super().__init__()
        self.input_layernorm = MiniMaxMusic3RMSNorm(dim)
        self.attn = MiniMaxMusic3DepthAttention(dim, heads)
        self.post_attention_layernorm = MiniMaxMusic3RMSNorm(dim)
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.input_layernorm(hidden_states))
        norm_states = self.post_attention_layernorm(hidden_states)
        return hidden_states + self.down_proj(F.silu(self.gate_proj(norm_states)) * self.up_proj(norm_states))


class MiniMaxMusic3RVQDepthDecoder(nn.Module):

    def __init__(
        self,
        hidden_size: int = 4096,
        num_layers: int = 4,
        num_attention_heads: int = 16,
        intermediate_size: int = 6144,
        audio_vocab_size: int = 1024,
        num_codebooks: int = 8,
        max_position_embeddings: int = 16,
    ):
        super().__init__()
        self.audio_embeddings = nn.Embedding(audio_vocab_size * (num_codebooks - 1), hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pos_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.layers = nn.ModuleList([
            MiniMaxMusic3DepthDecoderBlock(hidden_size, num_attention_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.norm = MiniMaxMusic3RMSNorm(hidden_size)
        self.audio_heads = nn.ModuleList(
            [nn.Linear(hidden_size, audio_vocab_size, bias=False) for _ in range(num_codebooks - 1)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        hidden_states = inputs_embeds + self.pos_embedding(positions).unsqueeze(0)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)
