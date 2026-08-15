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
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch_size, seq_len, self.heads, self.head_dim)
        key = self.k_proj(hidden_states).view(batch_size, seq_len, self.heads, self.head_dim)
        value = self.v_proj(hidden_states).view(batch_size, seq_len, self.heads, self.head_dim)
        hidden_states = attention_forward(
            query, key, value,
            q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
            is_causal=True,
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        return self.o_proj(hidden_states)


class MiniMaxMusic3DepthMLP(nn.Module):
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class MiniMaxMusic3DepthDecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, intermediate_size: int):
        super().__init__()
        self.input_layernorm = MiniMaxMusic3RMSNorm(dim)
        self.self_attn = MiniMaxMusic3DepthAttention(dim, heads)
        self.post_attention_layernorm = MiniMaxMusic3RMSNorm(dim)
        self.mlp = MiniMaxMusic3DepthMLP(dim, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states))
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class MiniMaxMusic3AudioDecoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_attention_heads, intermediate_size, audio_vocab_size, num_codebooks, max_position_embeddings):
        super().__init__()
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
        self.audio_extra_embedding = nn.Embedding(audio_vocab_size * (num_codebooks - 1), hidden_size)
        self.audio_decoder = MiniMaxMusic3AudioDecoder(
            hidden_size, num_layers, num_attention_heads, intermediate_size,
            audio_vocab_size, num_codebooks, max_position_embeddings,
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        hidden_states = inputs_embeds + self.audio_decoder.pos_embedding(positions).unsqueeze(0)
        for layer in self.audio_decoder.layers:
            hidden_states = layer(hidden_states)
        return self.audio_decoder.norm(hidden_states)
