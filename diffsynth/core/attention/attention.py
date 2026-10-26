import os
from .registry import (
    get_attention_implementation,
    list_attention_implementations,
    register_attention_implementation,
)
from .flash_attention import *
from .sage_attention import *
from .xformers_attention import *
from .torch_attention import torch_sdpa, torch_sdpa_sliding_window
from .flex_attention import flex_attention


def initialize_attention_priority():
    if os.environ.get('DIFFSYNTH_ATTENTION_IMPLEMENTATION') is not None:
        return os.environ.get('DIFFSYNTH_ATTENTION_IMPLEMENTATION').strip().lower()
    else:
        for name in ["customized_fa_kernel", "flash_attention_4", "flash_attention_3", "flash_attention_2", "sage_attention", "xformers"]:
            if name in list_attention_implementations():
                return name
        return "torch"


ATTENTION_IMPLEMENTATION = initialize_attention_priority()


def attention_forward(q, k, v, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, attn_mask=None, scale=None, is_causal=False, compatibility_mode=False, window_size=None, use_flex=False, score_mod=None):
    if use_flex or score_mod is not None:
        return flex_attention(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, attn_mask=attn_mask, scale=scale, score_mod=score_mod)

    if compatibility_mode:
        return torch_sdpa(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, attn_mask=attn_mask, scale=scale, is_causal=is_causal, window_size=window_size)

    backend = get_attention_implementation(ATTENTION_IMPLEMENTATION)

    if attn_mask is not None and not backend.supports_mask:
        return torch_sdpa(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, attn_mask=attn_mask, scale=scale, is_causal=is_causal)
    if window_size is not None and not backend.supports_window:
        assert is_causal == False and attn_mask is None
        return torch_sdpa_sliding_window(q, k, v, window_size, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
    if is_causal and not backend.supports_causal:
        return torch_sdpa(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, attn_mask=attn_mask, scale=scale, is_causal=is_causal)

    kwargs = {"scale": scale}
    if backend.supports_causal:
        kwargs["is_causal"] = is_causal
    if backend.supports_mask:
        kwargs["attn_mask"] = attn_mask
    if backend.supports_window:
        kwargs["window_size"] = window_size
    return backend.func(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, **kwargs)
