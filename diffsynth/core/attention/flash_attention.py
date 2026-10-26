import os

from .common import rearrange_qkv, rearrange_out
from .registry import register_attention_implementation

if os.environ.get("DIFFSYNTH_FLASH_ATTN_KERNEL_REPO_ID") is not None:
    try:
        from kernels import get_kernel
        kernel_module = get_kernel(os.environ.get("DIFFSYNTH_FLASH_ATTN_KERNEL_REPO_ID"), version=int(os.environ.get("DIFFSYNTH_FLASH_ATTN_KERNEL_VERSION")))
        customized_flash_attn_func = kernel_module.flash_attn_func
        CUSTOMIZED_FA_KERNEL_AVAILABLE = True
    except Exception:
        CUSTOMIZED_FA_KERNEL_AVAILABLE = False
else:
    CUSTOMIZED_FA_KERNEL_AVAILABLE = False

try:
    from flash_attn.cute import flash_attn_func as flash_attn_func_cute
    FLASH_ATTN_4_AVAILABLE = True
except Exception:
    FLASH_ATTN_4_AVAILABLE = False

try:
    from flash_attn_3 import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except Exception:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = hasattr(flash_attn, "flash_attn_func")
except Exception:
    FLASH_ATTN_2_AVAILABLE = False


def _require(flag, name):
    if not flag:
        raise RuntimeError(f"{name} is not available.")


def flash_attention_4(q, k, v, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None, is_causal=False, window_size=None):
    _require(FLASH_ATTN_4_AVAILABLE, "FlashAttention-4")
    required_in_pattern, required_out_pattern = "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    window_size = (window_size, window_size) if window_size is not None else (-1, -1)
    out = flash_attn_func_cute(q, k, v, softmax_scale=scale, causal=is_causal, window_size=window_size)
    if isinstance(out, tuple):
        out = out[0]
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def flash_attention_3(q, k, v, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None, is_causal=False, window_size=None):
    _require(FLASH_ATTN_3_AVAILABLE, "FlashAttention-3")
    required_in_pattern, required_out_pattern = "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    window_size = (window_size, window_size) if window_size is not None else (-1, -1)
    out = flash_attn_interface.flash_attn_func(q, k, v, softmax_scale=scale, causal=is_causal, window_size=window_size)
    if isinstance(out, tuple):
        out = out[0]
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def flash_attention_2(q, k, v, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None, is_causal=False, window_size=None):
    _require(FLASH_ATTN_2_AVAILABLE, "FlashAttention-2")
    required_in_pattern, required_out_pattern = "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    window_size = (window_size, window_size) if window_size is not None else (-1, -1)
    out = flash_attn.flash_attn_func(q, k, v, softmax_scale=scale, causal=is_causal, window_size=window_size)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def flash_attention_customized_kernel(q, k, v, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None, is_causal=False, window_size=None):
    _require(CUSTOMIZED_FA_KERNEL_AVAILABLE, "customized FlashAttention kernel")
    required_in_pattern, required_out_pattern = "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    window_size = (window_size, window_size) if window_size is not None else (-1, -1)
    out = customized_flash_attn_func(q, k, v, softmax_scale=scale, causal=is_causal, window_size=window_size)
    if isinstance(out, tuple):
        out = out[0]
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


if CUSTOMIZED_FA_KERNEL_AVAILABLE:
    register_attention_implementation(
        "customized_fa_kernel",
        flash_attention_customized_kernel,
        supports_causal=True,
        supports_window=True,
        supports_gqa=True,
    )
if FLASH_ATTN_4_AVAILABLE:
    register_attention_implementation(
        "flash_attention_4",
        flash_attention_4,
        supports_causal=True,
        supports_window=True,
        supports_gqa=True,
    )
if FLASH_ATTN_3_AVAILABLE:
    register_attention_implementation(
        "flash_attention_3",
        flash_attention_3,
        supports_causal=True,
        supports_window=True,
        supports_gqa=True,
    )
if FLASH_ATTN_2_AVAILABLE:
    register_attention_implementation(
        "flash_attention_2",
        flash_attention_2,
        supports_causal=True,
        supports_window=True,
        supports_gqa=True,
    )
