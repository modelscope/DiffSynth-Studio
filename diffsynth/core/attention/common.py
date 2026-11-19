import inspect
import os

import torch
from einops import rearrange

try:
    TORCH_SUPPORT_GQA = "enable_gqa" in inspect.signature(torch.nn.functional.scaled_dot_product_attention).parameters
except Exception:
    TORCH_SUPPORT_GQA = False

_PRE_SM80_CACHE = {}


def is_pre_sm80_cuda(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    idx = device.index if device.index is not None else torch.cuda.current_device()
    cached = _PRE_SM80_CACHE.get(idx)
    if cached is None:
        cached = torch.cuda.get_device_capability(idx)[0] < 8
        _PRE_SM80_CACHE[idx] = cached
    return cached


def is_bf16_attn_bridge_disabled() -> bool:
    val = os.environ.get("DIFFSYNTH_DISABLE_BF16_ATTN_BRIDGE")
    return val is not None and val.strip().lower() in ("1", "true", "yes", "on")


def rearrange_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", required_in_pattern="b n s d", dims=None):
    dims = {} if dims is None else dims
    if q_pattern != required_in_pattern:
        q = rearrange(q, f"{q_pattern} -> {required_in_pattern}", **dims)
    if k_pattern != required_in_pattern:
        k = rearrange(k, f"{k_pattern} -> {required_in_pattern}", **dims)
    if v_pattern != required_in_pattern:
        v = rearrange(v, f"{v_pattern} -> {required_in_pattern}", **dims)
    return q, k, v


def rearrange_out(out: torch.Tensor, out_pattern="b n s d", required_out_pattern="b n s d", dims=None):
    dims = {} if dims is None else dims
    if out_pattern != required_out_pattern:
        out = rearrange(out, f"{required_out_pattern} -> {out_pattern}", **dims)
    return out
