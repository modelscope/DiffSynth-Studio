import torch
import torch.nn as nn
from abc import ABC
from .encoders import ActionEncoder


class BaseInjector(nn.Module, ABC):
    """Marker base class for all conditioning injectors."""


class CrossAttnInjector(BaseInjector):
    """
    Injects conditioning as extra tokens appended to the cross-attention
    context sequence.

    Projects encoder output (B, N, embed_dim) -> (B, N, backbone_dim) so
    the tokens can be concatenated directly with projected text tokens.
    """

    def __init__(self, encoder: ActionEncoder, embed_dim: int, backbone_dim: int):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Linear(embed_dim, backbone_dim)

    def get_context_tokens(self, condition: torch.Tensor) -> torch.Tensor:
        """condition: raw input passed to encoder -> (B, N, backbone_dim)"""
        tokens = self.encoder(condition)   # (B, N, embed_dim)
        return self.proj(tokens)           # (B, N, backbone_dim)


class InputConcatInjector(BaseInjector):
    """
    Injects conditioning by concatenating pre-encoded latents as extra input
    channels before the backbone's patch embedding.

    No learnable parameters here; the extended patch_embedding (created by
    ActionConditionedDiT) handles the projection.
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels

    def get_extra_channels(self, latent: torch.Tensor) -> torch.Tensor:
        """latent: VAE-encoded (B, C, T, H, W) -- returned as-is."""
        return latent


class AdaLNInjector(BaseInjector):
    """
    Injects conditioning via adaptive layer norm modulation.

    Pools encoder output to a single vector and projects it to backbone_dim
    for addition to the time embedding inside the backbone forward pass.
    """

    def __init__(self, encoder: ActionEncoder, embed_dim: int, backbone_dim: int):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, backbone_dim),
            nn.SiLU(),
            nn.Linear(backbone_dim, backbone_dim),
        )

    def get_time_delta(self, condition: torch.Tensor) -> torch.Tensor:
        """condition: raw action input -> (B, backbone_dim) to add to time emb."""
        tokens = self.encoder(condition)   # (B, N, embed_dim)
        pooled = tokens.mean(dim=1)        # (B, embed_dim)
        return self.proj(pooled)           # (B, backbone_dim)
