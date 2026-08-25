import torch
import torch.nn as nn
import torch.nn.functional as F


class FluxReduxImageEncoder(nn.Module):
    def __init__(self, redux_dim: int = 1152, txt_in_features: int = 4096):
        super().__init__()
        self.redux_up = nn.Linear(redux_dim, txt_in_features * 3)
        self.redux_down = nn.Linear(txt_in_features * 3, txt_in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.redux_down(F.silu(self.redux_up(x)))
