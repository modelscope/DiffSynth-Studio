from .common import rearrange_qkv, rearrange_out
from .registry import register_attention_implementation

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except Exception:
    SAGE_ATTN_AVAILABLE = False

def sage_attention(q, k, v, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    if not SAGE_ATTN_AVAILABLE:
        raise RuntimeError("SageAttention is not available.")
    required_in_pattern, required_out_pattern = "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = sageattn(q, k, v, sm_scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out

if SAGE_ATTN_AVAILABLE:
    register_attention_implementation("sage_attention", sage_attention)
