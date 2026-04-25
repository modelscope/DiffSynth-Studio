import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from einops import repeat


class ActionEncoder(nn.Module, ABC):
    """Base interface for all action sequence encoders."""

    @abstractmethod
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        actions : (B, T, action_dim)

        Returns
        -------
        tokens : (B, N, embed_dim)
        """


# ── Perceiver / Resampler ──────────────────────────────────────────────────────

class PerceiverLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_mult: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        kv = self.norm_kv(context)
        latents = latents + self.cross_attn(self.norm_q(latents), kv, kv)[0]
        latents = latents + self.ff(self.norm_ff(latents))
        return latents


class PerceiverActionEncoder(ActionEncoder):
    """
    Resampler-style encoder (cf. EVAC / Flamingo).

    Learned latent queries cross-attend to the projected action sequence,
    mapping (B, T, action_dim) to a fixed (B, num_queries, embed_dim).
    """

    def __init__(
        self,
        action_dim: int,
        embed_dim: int,
        num_queries: int = 16,
        depth: int = 4,
        num_heads: int = 8,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.input_proj = nn.Linear(action_dim, embed_dim)
        self.latents = nn.Parameter(torch.randn(1, num_queries, embed_dim) / embed_dim ** 0.5)
        self.layers = nn.ModuleList([
            PerceiverLayer(embed_dim, num_heads, ff_mult) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        B = actions.shape[0]
        context = self.input_proj(actions)                         # (B, T, embed_dim)
        latents = repeat(self.latents, "1 n d -> b n d", b=B)     # (B, num_queries, embed_dim)
        for layer in self.layers:
            latents = layer(latents, context)
        return self.norm(latents)                                   # (B, num_queries, embed_dim)


# ── MLP ───────────────────────────────────────────────────────────────────────

class MLPActionEncoder(ActionEncoder):
    """
    Simple per-timestep MLP encoder.

    Maps (B, T, action_dim) -> (B, T, embed_dim); sequence length is preserved.
    Suitable for AdaLN injection (the wrapper mean-pools before projection).
    """

    def __init__(self, action_dim: int, embed_dim: int, num_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(action_dim, embed_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(embed_dim, embed_dim), nn.GELU()]
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.norm(self.mlp(actions))   # (B, T, embed_dim)
