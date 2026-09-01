import torch

CONTROL_PREFIXES = ("control_blocks.", "control_proj_in.")
WRAPPER_PREFIXES = ("_fsdp_wrapped_module.", "module.", "_orig_mod.")
NUM_ATTENTION_HEADS = 56
ATTENTION_HEAD_DIM = 128


def unwrap_key(key):
    changed = True
    while changed:
        changed = False
        for prefix in WRAPPER_PREFIXES:
            if key.startswith(prefix):
                key, changed = key[len(prefix):], True
    return key


def interleave_qkv(query, key, value, num_attention_heads=NUM_ATTENTION_HEADS, attention_head_dim=ATTENTION_HEAD_DIM):
    """Fuse separate q/k/v weights into the per-head interleaved layout `MiniMaxH3Attention` reads.

    `qkv_proj` output rows are consumed as `view(total, heads, 3, head_dim)`, i.e. rows are grouped
    per head as `[head0: q k v, head1: q k v, ...]`. This is the inverse of the reorder the source
    checkpoint's `[q_all; k_all; v_all]` layout went through when it was split.
    """
    inner_dim = num_attention_heads * attention_head_dim
    for name, tensor in (("to_q", query), ("to_k", key), ("to_v", value)):
        if tensor.shape[0] != inner_dim:
            raise ValueError(f"{name} has {tensor.shape[0]} rows, expected {inner_dim} = {num_attention_heads} heads * {attention_head_dim}")
    per_head = [t.reshape(num_attention_heads, attention_head_dim, *t.shape[1:]) for t in (query, key, value)]
    fused = torch.cat(per_head, dim=1)
    return fused.reshape(num_attention_heads * 3 * attention_head_dim, *query.shape[1:]).contiguous()


def swap_swiglu_halves(weight):
    """Swap the two halves of a fused gated-FFN weight.

    `MiniMaxH3MLP.fc1` computes `silu(gate) * up` from a `[gate; up]` fusion, while the source
    checkpoint stores the diffusers `SwiGLU` order `[up; gate]`.
    """
    up, gate = weight.chunk(2, dim=0)
    return torch.cat([gate, up], dim=0).contiguous()


def MiniMaxH3ControlNetStateDictConverter(state_dict):
    source = {}
    for key in state_dict:
        bare_key = unwrap_key(key)
        if bare_key.startswith(CONTROL_PREFIXES):
            source[bare_key] = state_dict[key]
    if not source:
        return {}

    state_dict_ = {}
    for key in ("control_proj_in.weight", "control_proj_in.bias"):
        if key in source:
            state_dict_[key.replace("control_proj_in.", "control_patch_proj.")] = source[key]

    block_ids = sorted({int(key.split(".")[1]) for key in source if key.startswith("control_blocks.")})
    for block_id in block_ids:
        prefix = f"control_blocks.{block_id}."
        target_prefix = f"blocks.{block_id}."
        for suffix in ("norm1.weight", "norm2.weight", "before_proj.weight", "before_proj.bias", "after_proj.weight", "after_proj.bias", "adaln_proj.linear.weight", "adaln_proj.linear.bias"):
            if prefix + suffix in source:
                state_dict_[target_prefix + suffix] = source[prefix + suffix]
        for source_suffix, target_suffix in (("attn.norm_q.weight", "attn.q_norm.weight"), ("attn.norm_k.weight", "attn.k_norm.weight"), ("attn.to_out.0.weight", "attn.out_proj.weight"), ("ff.net.2.weight", "mlp.fc2.weight")):
            if prefix + source_suffix in source:
                state_dict_[target_prefix + target_suffix] = source[prefix + source_suffix]
        if prefix + "attn.to_q.weight" in source:
            state_dict_[target_prefix + "attn.qkv_proj.weight"] = interleave_qkv(
                source[prefix + "attn.to_q.weight"], source[prefix + "attn.to_k.weight"], source[prefix + "attn.to_v.weight"],
            )
        if prefix + "ff.net.0.proj.weight" in source:
            state_dict_[target_prefix + "mlp.fc1.weight"] = swap_swiglu_halves(source[prefix + "ff.net.0.proj.weight"])
    return state_dict_