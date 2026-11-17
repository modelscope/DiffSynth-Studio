import os

import torch

from .registry import (
    get_attention_implementation,
    list_attention_implementations,
    register_attention_implementation,
)
from .flash_attention import CUSTOMIZED_FA_KERNEL_AVAILABLE, FLASH_ATTN_4_AVAILABLE, FLASH_ATTN_3_AVAILABLE, FLASH_ATTN_2_AVAILABLE
from .sage_attention import SAGE_ATTN_AVAILABLE
from .xformers_attention import XFORMERS_AVAILABLE
from .torch_attention import torch_sdpa, torch_sdpa_sliding_window
from .flex_attention import flex_attention


def initialize_attention_priority():
    if os.environ.get('DIFFSYNTH_ATTENTION_IMPLEMENTATION') is not None:
        return os.environ.get('DIFFSYNTH_ATTENTION_IMPLEMENTATION').strip().lower()
    else:
        for name in ["customized_fa_kernel", "flash_attention_4", "flash_attention_3", "flash_attention_2", "xformers"]:
            if name in list_attention_implementations():
                return name
        return "torch"


ATTENTION_IMPLEMENTATION = initialize_attention_priority()


def local_attention_forward(q, k, v, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, attn_mask=None, scale=None, is_causal=False, compatibility_mode=False, window_size=None, use_flex=False, score_mod=None, cu_seqlens=None):
    if cu_seqlens is not None:
        # Packed stream: attend within each segment so the loop stays inside the
        # sequence-parallel exchange and sees the global segment boundaries.
        bounds = cu_seqlens.tolist()
        # Token axis of each layout, e.g. "b n s d" -> 2.
        q_dim, k_dim, v_dim, out_dim = (pattern.split().index("s") for pattern in (q_pattern, k_pattern, v_pattern, out_pattern))
        # Allocate in the input layout: segments may not cover a padded tail.
        out = torch.empty_like(q)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if stop == start:
                continue
            q_seg, k_seg, v_seg = (t.narrow(dim, start, stop - start) for t, dim in zip((q, k, v), (q_dim, k_dim, v_dim)))
            seg_out = local_attention_forward(q_seg, k_seg, v_seg, q_pattern, k_pattern, v_pattern, out_pattern,
                                              dims, attn_mask, scale, is_causal, compatibility_mode, window_size, use_flex, score_mod)
            out.narrow(out_dim, start, stop - start).copy_(seg_out)
        return out

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


def attention_forward(q, k, v, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, attn_mask=None, scale=None, is_causal=False, compatibility_mode=False, window_size=None, use_flex=False, score_mod=None, cu_seqlens=None):
    from ..distributed.sp_state import current_sequence_parallel_state

    attn_kwargs = {
        "dims": dims, "attn_mask": attn_mask, "scale": scale, "is_causal": is_causal,
        "compatibility_mode": compatibility_mode, "window_size": window_size,
        "use_flex": use_flex, "score_mod": score_mod, "cu_seqlens": cu_seqlens,
    }
    sp_state = current_sequence_parallel_state()
    if sp_state is None or not sp_state.attention_parallel:
        # No sharded stack, or a replicated-K/V submodule inside one: run the local kernel.
        return local_attention_forward(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, **attn_kwargs)

    return sp_state.backend.forward(
        q, k, v,
        local_attention=local_attention_forward,
        q_pattern=q_pattern, k_pattern=k_pattern, v_pattern=v_pattern, out_pattern=out_pattern,
        **attn_kwargs,
    )
