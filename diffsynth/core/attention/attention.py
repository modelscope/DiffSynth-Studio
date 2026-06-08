import torch, os, inspect
from einops import rearrange, repeat


try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False

try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ModuleNotFoundError:
    XFORMERS_AVAILABLE = False


def initialize_attention_priority():
    if os.environ.get('DIFFSYNTH_ATTENTION_IMPLEMENTATION') is not None:
        return os.environ.get('DIFFSYNTH_ATTENTION_IMPLEMENTATION').lower()
    elif FLASH_ATTN_3_AVAILABLE:
        return "flash_attention_3"
    elif FLASH_ATTN_2_AVAILABLE:
        return "flash_attention_2"
    elif SAGE_ATTN_AVAILABLE:
        return "sage_attention"
    elif XFORMERS_AVAILABLE:
        return "xformers"
    else:
        return "torch"


ATTENTION_IMPLEMENTATION = initialize_attention_priority()


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


def torch_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, attn_mask=None, scale=None, is_causal=False):
    required_in_pattern, required_out_pattern= "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        # Grouped Query Attention
        if "enable_gqa" in inspect.signature(torch.nn.functional.scaled_dot_product_attention).parameters:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask, scale=scale, is_causal=is_causal, enable_gqa=True)
        else:
            # In low-version torch, `enable_gqa` is not supported.
            k = repeat(k, "b n s d -> b (n m) s d", m=q.shape[1]//k.shape[1])
            v = repeat(v, "b n s d -> b (n m) s d", m=q.shape[1]//v.shape[1])
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask, scale=scale, is_causal=is_causal)
    else:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask, scale=scale, is_causal=is_causal)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def torch_sdpa_sliding_window(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sliding_window: int, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    required_in_pattern, required_out_pattern = "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)

    B, N, S, D = q.shape
    W = sliding_window
    chunk_size = W
    num_chunks = (S + chunk_size - 1) // chunk_size

    output = torch.empty_like(q)
    dtype = q.dtype
    device = q.device
    min_val = torch.finfo(dtype).min

    for i in range(num_chunks):
        q_start = i * chunk_size
        q_end = min(q_start + chunk_size, S)
        actual_chunk_size = q_end - q_start

        kv_start = max(0, q_start - W)
        kv_end = min(S, q_end + W)

        q_chunk = q[:, :, q_start:q_end, :]
        k_chunk = k[:, :, kv_start:kv_end, :]
        v_chunk = v[:, :, kv_start:kv_end, :]

        q_indices = torch.arange(q_start, q_end, device=device)
        k_indices = torch.arange(kv_start, kv_end, device=device)
        diff = q_indices.unsqueeze(1) - k_indices.unsqueeze(0)
        valid = diff.abs() <= W

        local_mask = torch.zeros(actual_chunk_size, kv_end - kv_start, dtype=dtype, device=device)
        local_mask.masked_fill_(~valid, min_val)
        local_mask = local_mask.unsqueeze(0).unsqueeze(0)

        out_chunk = torch_sdpa(
            q_chunk, k_chunk, v_chunk, attn_mask=local_mask, scale=scale
        )
        output[:, :, q_start:q_end, :] = out_chunk

    output = rearrange_out(output, out_pattern, required_out_pattern, dims)
    return output


def flash_attention_3(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None, is_causal=False, window_size=None):
    required_in_pattern, required_out_pattern= "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    window_size = (window_size, window_size) if window_size is not None else (-1, -1)
    out = flash_attn_interface.flash_attn_func(q, k, v, softmax_scale=scale, window_size=window_size)
    if isinstance(out, tuple):
        out = out[0]
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def flash_attention_2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None, is_causal=False, window_size=None):
    required_in_pattern, required_out_pattern= "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    window_size = (window_size, window_size) if window_size is not None else (-1, -1)
    out = flash_attn.flash_attn_func(q, k, v, softmax_scale=scale, causal=is_causal, window_size=window_size)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def sage_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    required_in_pattern, required_out_pattern= "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = sageattn(q, k, v, sm_scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def xformers_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    required_in_pattern, required_out_pattern= "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = xops.memory_efficient_attention(q, k, v, scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def attention_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, attn_mask=None, scale=None, is_causal=False, compatibility_mode=False, window_size=None):
    if compatibility_mode or (attn_mask is not None) or ATTENTION_IMPLEMENTATION == "torch":
        if window_size is None:
            return torch_sdpa(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, attn_mask=attn_mask, scale=scale, is_causal=is_causal)
        else:
            # Sliding Window Attention is not compatible with `is_causal` and `attn_mask`.
            assert is_causal == False and attn_mask is None
            return torch_sdpa_sliding_window(q, k, v, window_size, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
    elif ATTENTION_IMPLEMENTATION == "flash_attention_3":
        return flash_attention_3(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale, is_causal=is_causal, window_size=window_size)
    elif ATTENTION_IMPLEMENTATION == "flash_attention_2":
        return flash_attention_2(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale, is_causal=is_causal, window_size=window_size)
    elif ATTENTION_IMPLEMENTATION == "sage_attention":
        if window_size is not None or is_causal: return attention_forward(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, attn_mask, scale, is_causal, compatibility_mode=True, window_size=window_size)
        return sage_attention(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
    elif ATTENTION_IMPLEMENTATION == "xformers":
        if window_size is not None or is_causal: return attention_forward(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, attn_mask, scale, is_causal, compatibility_mode=True, window_size=window_size)
        return xformers_attention(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
    else:
        raise NotImplementedError("No available attention implementation.")
