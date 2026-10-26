from .common import rearrange_qkv, rearrange_out
from .registry import register_attention_implementation

try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except Exception:
    XFORMERS_AVAILABLE = False


def xformers_attention(q, k, v, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    if not XFORMERS_AVAILABLE:
        raise RuntimeError("xformers is not available.")
    required_in_pattern, required_out_pattern = "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = xops.memory_efficient_attention(q, k, v, scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


if XFORMERS_AVAILABLE:
    register_attention_implementation("xformers", xformers_attention)
