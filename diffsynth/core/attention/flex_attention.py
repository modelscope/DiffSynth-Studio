import torch

from .common import rearrange_qkv, rearrange_out

try:
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention_func
    FLEX_ATTN_AVAILABLE = True
except Exception:
    FLEX_ATTN_AVAILABLE = False

# Compile flex attention lazily
_compiled_flex_attention_func = None


def get_flex_attention_func():
    global _compiled_flex_attention_func
    if _compiled_flex_attention_func is None:
        _compiled_flex_attention_func = torch.compile(_flex_attention_func)
    return _compiled_flex_attention_func


def flex_attention(q, k, v, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, attn_mask=None, scale=None, score_mod=None):
    assert FLEX_ATTN_AVAILABLE, "Flex Attention is not available. Please upgrade torch to 2.5.0 or later."
    required_in_pattern, required_out_pattern = "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = get_flex_attention_func()(query=q, key=k, value=v, block_mask=attn_mask, scale=scale, score_mod=score_mod)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out
