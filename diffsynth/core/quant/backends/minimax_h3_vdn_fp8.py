"""W8A8 FP8 (e4m3) backend for MiniMax-H3 / VDN-H3 DiT inference.

Ported from the target library's src/models/ops/fp8_linear.py. Quantises both weights
(once at construction, per-output-channel on sm90 / per-tensor on sm100+) and activations
(per row on sm90 / per tensor on sm100+) to float8_e4m3fn, then runs the GEMM via
torch._scaled_mm with use_fast_accum=True.

OPT-IN, NEVER DEFAULT. FP8 does not degrade the output — it CHANGES it (cosine ~0.998
per step vs bf16). The denoising trajectory is chaotic: the final clip is a different
(not worse) sample of the same prompt.

Only Linears at or above `min_width` on BOTH sides are quantised (qkv projections, to_out,
FF GEMMs, to_out_linear). Narrow modules (gates, beta_proj, alpha, adaln_proj) stay bf16.
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..base import BackendConfig, QuantBackend, register_quant_backend
from ..config import register_quant_method

# NOTE the half order. The kernel upstream computes `a * silu(g)` (diffusers' SwiGLU:
# `hidden_states, gate = chunk(2); hidden_states * activation(gate)`). DiffSynth's
# MiniMaxH3MLP computes `silu(gate) * up` and its state_dict converters store fc1 with the
# two halves SWAPPED (see swap_swiglu_halves in state_dict_converters/minimax_h3_controlnet.py),
# so the equivalent expression here is `a * silu(a) * ... ` -> silu(first) * second.
# Porting the kernel verbatim silently halves the FF into the wrong gate: the run still
# produces a plausible video, but the single-step velocity cosine against bf16 drops to
# ~0.53 instead of ~0.998.

FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(FP8_DTYPE).max
MIN_WIDTH = 4096
SKIP_END_BLOCKS = 0

_PER_TENSOR = None


def per_tensor_gemm():
    global _PER_TENSOR
    if _PER_TENSOR is None:
        _PER_TENSOR = (torch.cuda.is_available()
                       and torch.cuda.get_device_capability(0)[0] >= 10)
    return _PER_TENSOR


# ---------------------------------------------------------------------------------------
# Triton quantisation kernels
# ---------------------------------------------------------------------------------------

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
def _swiglu_quantize_kernel(H, Y, S, K, FP8_MAX: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    amax = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(H + row * 2 * K + cols, mask=cols < K, other=0.0).to(tl.float32)
        g = tl.load(H + row * 2 * K + K + cols, mask=cols < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(a * tl.sigmoid(a) * g))
    scale = tl.maximum(tl.max(amax, axis=0) / FP8_MAX, 1e-12)
    tl.store(S + row, scale)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(H + row * 2 * K + cols, mask=cols < K, other=0.0).to(tl.float32)
        g = tl.load(H + row * 2 * K + K + cols, mask=cols < K, other=0.0).to(tl.float32)
        y = tl.minimum(tl.maximum(a * tl.sigmoid(a) * g / scale, -FP8_MAX), FP8_MAX)
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


@triton.jit
def _swiglu_rowmax_kernel(H, Y, RM, K, BLOCK_K: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    amax = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        mask = cols < K
        a = tl.load(H + row * 2 * K + cols, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(H + row * 2 * K + K + cols, mask=mask, other=0.0).to(tl.float32)
        act = (a * tl.sigmoid(a) * g).to(Y.dtype.element_ty)
        amax = tl.maximum(amax, tl.abs(act.to(tl.float32)))
        tl.store(Y + row * K + cols, act, mask=mask)
    tl.store(RM + row, tl.max(amax, axis=0))


# ---------------------------------------------------------------------------------------
# Python-level quantisation entry points
# ---------------------------------------------------------------------------------------

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
    _cast_scaled_kernel[(triton.cdiv(n, 8192),)](x.view(-1), y.view(-1), scale, n,
                                                 FP8_MAX=_FP8_MAX, BLOCK=8192, num_warps=8)
    return y, scale


def swiglu_quantize(h):
    h = _rows(h)
    M, K2 = h.shape
    K = K2 // 2
    y = torch.empty(M, K, device=h.device, dtype=FP8_DTYPE)
    scale = torch.empty(M, 1, device=h.device, dtype=torch.float32)
    _swiglu_quantize_kernel[(M,)](h, y, scale, K, FP8_MAX=_FP8_MAX, BLOCK_K=2048, num_warps=16)
    return y, scale


def swiglu_quantize_tensor(h):
    h = _rows(h)
    M, K2 = h.shape
    K = K2 // 2
    act = torch.empty(M, K, device=h.device, dtype=h.dtype)
    rowmax = torch.empty(M, device=h.device, dtype=torch.float32)
    _swiglu_rowmax_kernel[(M,)](h, act, rowmax, K, BLOCK_K=2048, num_warps=16)
    scale = (rowmax.amax() / _FP8_MAX).clamp_min(1e-12).reshape(1, 1)
    y = torch.empty_like(act, dtype=FP8_DTYPE)
    n = act.numel()
    _cast_scaled_kernel[(triton.cdiv(n, 8192),)](act.view(-1), y.view(-1), scale, n,
                                                 FP8_MAX=_FP8_MAX, BLOCK=8192, num_warps=8)
    return y, scale


def quantize_activation(x):
    return quantize_tensor(x) if per_tensor_gemm() else quantize_rows(x)


def swiglu_quantize_activation(h):
    return swiglu_quantize_tensor(h) if per_tensor_gemm() else swiglu_quantize(h)


def quantize_rows_reference(x):
    scale = (x.float().abs().amax(dim=1, keepdim=True) / _FP8_MAX).clamp_min(1e-12)
    return (x.float() / scale).to(FP8_DTYPE), scale


# ---------------------------------------------------------------------------------------
# The quantised Linear
# ---------------------------------------------------------------------------------------

class MiniMaxH3VdnFp8Linear(nn.Linear):
    """nn.Linear subclass whose GEMM runs in fp8 e4m3 via torch._scaled_mm.

    Weight is quantised once at construction; activation is quantised per forward call
    (or shared across q/k/v via forward_quantized). Only the fp8 weight copy and bias are
    kept — the bf16 weight is released."""

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
        # Release the bf16 weight — replace the Parameter with a meta sentinel so
        # nn.Linear's __init__ contract is satisfied but no memory is held.
        self.weight = nn.Parameter(torch.empty(0, device=weight.device, dtype=weight.dtype),
                                   requires_grad=False)

    def forward_quantized(self, x_fp8, x_scale, out_dtype=torch.bfloat16):
        out = torch._scaled_mm(x_fp8, self.weight_fp8.t(),
                               scale_a=x_scale, scale_b=self.weight_scale,
                               out_dtype=out_dtype, use_fast_accum=True)
        if self.bias is not None:
            out = out + self.bias
        return out

    def forward(self, x):
        shape = x.shape
        rows = x.reshape(-1, shape[-1])
        out = self.forward_quantized(*quantize_activation(rows), out_dtype=rows.dtype)
        return out.reshape(*shape[:-1], -1)


# ---------------------------------------------------------------------------------------
# Backend registration
# ---------------------------------------------------------------------------------------

@register_quant_backend("minimax_h3_vdn_fp8")
class MiniMaxH3VdnFp8QuantBackend(QuantBackend):
    project_url = "https://github.com/OpenVDN/vdn-minimax-h3"

    def capabilities(self):
        return {**super().capabilities(),
                "is_serializable": False,
                "is_differentiable": False,
                "is_compileable": False,
                "requires_calibration": False}

    def validate_environment(self):
        if not torch.cuda.is_available():
            raise RuntimeError("minimax_h3_vdn_fp8 requires CUDA")
        major = torch.cuda.get_device_capability(0)[0]
        if major < 9:
            raise RuntimeError(f"minimax_h3_vdn_fp8 requires sm89+ (got sm{major}x)")

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


from dataclasses import dataclass, field


@dataclass
class MiniMaxH3VdnFp8Config(BackendConfig):
    min_width: int = MIN_WIDTH
    skip_end_blocks: int = SKIP_END_BLOCKS


register_quant_method(
    "minimax_h3_vdn_fp8",
    "minimax_h3_vdn_fp8",
    MiniMaxH3VdnFp8Config.from_kwargs,
    label="8bit, fp8 e4m3 W8A8 (rowwise act + per-channel weight on sm90, per-tensor on sm100+), inference only",
)
