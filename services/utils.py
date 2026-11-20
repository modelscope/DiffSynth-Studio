def register_attention():
    import os
    os.environ["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] = "sage_attention_h20_fast"
    from sageattention import sageattn_qk_int8_pv_fp8_cuda_sm90
    from diffsynth.core.attention.common import rearrange_out, rearrange_qkv
    from diffsynth.core.attention.registry import register_attention_implementation

    def sage_attention_h20_fast(q, k, v, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
        required_in_pattern, required_out_pattern = "b s n d", "b s n d"
        q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        out = sageattn_qk_int8_pv_fp8_cuda_sm90(
            q, k, v,
            tensor_layout="NHD",
            qk_quant_gran="per_warp",
            pv_accum_dtype="fp32+fp32",
            smooth_k=False,
            is_causal=False,
            sm_scale=scale,
        )
        out = rearrange_out(out, out_pattern, required_out_pattern, dims)
        return out

    register_attention_implementation(
        "sage_attention_h20_fast",
        sage_attention_h20_fast,
        supports_causal=False,
        supports_window=False,
        supports_mask=False,
    )
