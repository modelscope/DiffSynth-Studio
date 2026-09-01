import torch
import torch.nn.functional as F
from torch import nn


_DEFAULT_TOKEN_CHUNK = 16_384


def swiglu(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    if x.dtype != w_gate.dtype:
        x = x.to(w_gate.dtype)
    leading, dim = x.shape[:-1], x.shape[-1]
    flat = x.reshape(-1, dim).contiguous()
    output = torch.empty_like(flat)
    for start in range(0, flat.shape[0], _DEFAULT_TOKEN_CHUNK):
        end = min(start + _DEFAULT_TOKEN_CHUNK, flat.shape[0])
        tokens = flat[start:end]
        workspace = torch.empty((end - start, w_gate.shape[0]), dtype=x.dtype, device=x.device)
        torch.mm(tokens, w_gate.t(), out=workspace)
        F.silu(workspace, inplace=True)
        workspace.mul_(F.linear(tokens, w_up))
        torch.mm(workspace, w_down.t(), out=output[start:end])
    return output.view(*leading, dim)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swiglu(x, self.w_gate.weight, self.w_up.weight, self.w_down.weight)


def plain_mlp(x: torch.Tensor, mlp: SwiGLU, norm: nn.RMSNorm) -> torch.Tensor:
    return x + mlp(norm(x))
