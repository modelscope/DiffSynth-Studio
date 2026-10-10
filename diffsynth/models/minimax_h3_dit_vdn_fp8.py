"""W8A8 FP8 (e4m3) for the MiniMax-H3 / VDN-H3 DiT's wide Linears.

Ported from VDN-H3: https://github.com/OpenVDN/vdn-minimax-h3/blob/main/src/models/ops/fp8_linear.py
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl
from dataclasses import dataclass

from ..core.quant import BackendConfig, QuantBackend, register_quant_backend, register_quant_method

FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(FP8_DTYPE).max
MIN_WIDTH = 4096

_PER_TENSOR = None


def per_tensor_gemm():
    global _PER_TENSOR
    if _PER_TENSOR is None:
        _PER_TENSOR = (torch.cuda.is_available()
                       and torch.cuda.get_device_capability(0)[0] >= 10)
    return _PER_TENSOR


@triton.jit
def _quantize_rows_kernel(X, Y, S, K, FP8_MAX: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    amax = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(X + row * K + cols, mask=cols < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(x))
    scale = tl.maximum(tl.max(amax, axis=0) / FP8_MAX, 1e-12)
    tl.store(S + row, scale)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(X + row * K + cols, mask=cols < K, other=0.0).to(tl.float32)
        y = tl.minimum(tl.maximum(x / scale, -FP8_MAX), FP8_MAX)
        tl.store(Y + row * K + cols, y.to(Y.dtype.element_ty), mask=cols < K)


@triton.jit
def _absmax_kernel(X, OUT, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs, mask=offs < N, other=0.0).to(tl.float32)
    tl.atomic_max(OUT, tl.max(tl.abs(x), axis=0))


@triton.jit
def _cast_scaled_kernel(X, Y, S, N, FP8_MAX: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    scale = tl.load(S)
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.minimum(tl.maximum(x / scale, -FP8_MAX), FP8_MAX)
    tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)


def _rows(x):
    if not x.is_cuda:
        raise ValueError("fp8 quantiser requires CUDA tensors")
    if x.dim() != 2:
        raise ValueError(f"expected [M, K], got {tuple(x.shape)}")
    return x.contiguous()


def quantize_rows(x):
    x = _rows(x)
    M, K = x.shape
    y = torch.empty_like(x, dtype=FP8_DTYPE)
    scale = torch.empty(M, 1, device=x.device, dtype=torch.float32)
    _quantize_rows_kernel[(M,)](x, y, scale, K, FP8_MAX=_FP8_MAX, BLOCK_K=1024, num_warps=4)
    return y, scale


def quantize_tensor(x):
    x = _rows(x)
    n = x.numel()
    amax = torch.zeros(1, device=x.device, dtype=torch.float32)
    _absmax_kernel[(triton.cdiv(n, 8192),)](x.view(-1), amax, n, BLOCK=8192, num_warps=8)
    scale = (amax / _FP8_MAX).clamp_min(1e-12).reshape(1, 1)
    y = torch.empty_like(x, dtype=FP8_DTYPE)
    _cast_scaled_kernel[(triton.cdiv(n, 8192),)](
        x.view(-1), y.view(-1), scale, n,
        FP8_MAX=_FP8_MAX, BLOCK=8192, num_warps=8)
    return y, scale


def quantize_activation(x):
    return quantize_tensor(x) if per_tensor_gemm() else quantize_rows(x)


class MiniMaxH3VdnFp8Linear(nn.Linear):
    dtype_guarded_tensor_names: tuple = ("weight_fp8", "weight_scale")

    def _apply(self, fn, recurse=True):
        protected = {id(tensor) for name in self.dtype_guarded_tensor_names
                     if (tensor := getattr(self, name, None)) is not None}

        def guard(tensor):
            converted = fn(tensor)
            if id(tensor) in protected and converted.dtype != tensor.dtype:
                return tensor.to(device=converted.device)
            return converted

        return super()._apply(guard, recurse)

    def __init__(self, linear: nn.Linear):
        super().__init__(linear.in_features, linear.out_features,
                         bias=linear.bias is not None,
                         device=linear.weight.device, dtype=linear.weight.dtype)
        weight = linear.weight.data
        if per_tensor_gemm():
            scale = (weight.abs().amax().float() / _FP8_MAX).clamp_min(1e-12)
            self.register_buffer("weight_fp8", (weight / scale.to(weight.dtype)).to(FP8_DTYPE))
            self.register_buffer("weight_scale", scale.reshape(1, 1).contiguous())
        else:
            scale = (weight.abs().amax(dim=1, keepdim=True).float() / _FP8_MAX).clamp_min(1e-12)
            self.register_buffer("weight_fp8", (weight / scale.to(weight.dtype)).to(FP8_DTYPE))
            self.register_buffer("weight_scale", scale.reshape(1, -1).contiguous())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None
        self.weight = nn.Parameter(
            torch.empty(0, device=weight.device, dtype=weight.dtype),
            requires_grad=False,
        )

    def forward_quantized(self, x_fp8, x_scale, out_dtype=torch.bfloat16):
        out = torch._scaled_mm(
            x_fp8, self.weight_fp8.t(),
            scale_a=x_scale, scale_b=self.weight_scale,
            out_dtype=out_dtype, use_fast_accum=True,
        )
        if self.bias is not None:
            out = out + self.bias
        return out

    def forward(self, x):
        shape = x.shape
        rows = x.reshape(-1, shape[-1])
        out = self.forward_quantized(*quantize_activation(rows), out_dtype=rows.dtype)
        return out.reshape(*shape[:-1], -1)


@register_quant_backend("minimax_h3_vdn_fp8")
class MiniMaxH3VdnFp8QuantBackend(QuantBackend):
    project_url = "https://github.com/OpenVDN/vdn-minimax-h3"

    def capabilities(self):
        return {
            "is_serializable": False,
            "is_differentiable": False,
            "is_compileable": False,
            "requires_calibration": False,
        }

    def validate_environment(self):
        if not torch.cuda.is_available():
            raise RuntimeError("minimax_h3_vdn_fp8 requires CUDA")
        major = torch.cuda.get_device_capability(0)[0]
        if major < 9:
            raise RuntimeError(f"minimax_h3_vdn_fp8 requires sm90+ (got sm{major}x)")

    def create_quantized_linear(self, linear, compute_device=None, model_device=None):
        min_width = self.config.min_width if self.config else MIN_WIDTH
        if linear.in_features < min_width or linear.out_features < min_width:
            return linear
        if compute_device is not None:
            linear = linear.to(compute_device)
        q = MiniMaxH3VdnFp8Linear(linear)
        if model_device is not None:
            q = q.to(model_device)
        return q

    def quantized_linear_classes(self):
        return (MiniMaxH3VdnFp8Linear,)


@dataclass
class MiniMaxH3VdnFp8Config(BackendConfig):
    min_width: int = MIN_WIDTH


register_quant_method(
    "minimax_h3_vdn_fp8",
    "minimax_h3_vdn_fp8",
    MiniMaxH3VdnFp8Config.from_kwargs,
    label="8bit, fp8 e4m3 W8A8 (rowwise act + per-channel weight on sm90, per-tensor on sm100+), inference only",
)
