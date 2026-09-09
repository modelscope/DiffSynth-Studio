from __future__ import annotations

import torch
from torch import nn


class QKVProjections(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.to_q(x), self.to_k(x), self.to_v(x)
