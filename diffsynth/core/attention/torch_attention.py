import torch
from einops import repeat

from .common import TORCH_SUPPORT_GQA, rearrange_qkv, rearrange_out
from .registry import register_attention_implementation


def torch_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, attn_mask=None, scale=None, is_causal=False, window_size=None):
    if window_size is not None:
        if is_causal or attn_mask is not None:
            raise ValueError("Sliding window attention does not support `is_causal=True` or `attn_mask`.")
        return torch_sdpa_sliding_window(q, k, v, window_size, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)

    required_in_pattern, required_out_pattern = "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        # Grouped Query Attention
        if TORCH_SUPPORT_GQA:
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


register_attention_implementation(
    "torch",
    torch_sdpa,
    supports_window=True,
    supports_causal=True,
    supports_mask=True,
    supports_gqa=True,
)
