from __future__ import annotations

import torch
import torch.nn.functional as F


def combined(
    context_and_x: torch.Tensor,
    w_proj: torch.Tensor,
    b_proj: torch.Tensor | None,
) -> torch.Tensor:
    context_channels = w_proj.shape[1]
    latent_context = context_and_x[..., :context_channels]
    x = context_and_x[..., context_channels:]
    return x + F.linear(latent_context, w_proj, b_proj)
