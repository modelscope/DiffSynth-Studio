"""
State dict converter for ACE-Step LLM (Qwen3-based).

The safetensors file stores Qwen3 model weights. Different checkpoints
may have different key formats:
- Qwen3ForCausalLM format: model.embed_tokens.weight, model.layers.0.*
- Qwen3Model format: embed_tokens.weight, layers.0.*

Qwen3ForCausalLM wraps a .model attribute (Qwen3Model), so its
state_dict() has keys:
    model.model.embed_tokens.weight
    model.model.layers.0.self_attn.q_proj.weight
    model.model.norm.weight
    model.lm_head.weight (tied to model.model.embed_tokens)

This converter normalizes all keys to the Qwen3ForCausalLM format.

Example mapping:
    model.embed_tokens.weight -> model.model.embed_tokens.weight
    embed_tokens.weight -> model.model.embed_tokens.weight
    model.layers.0.self_attn.q_proj.weight -> model.model.layers.0.self_attn.q_proj.weight
    layers.0.self_attn.q_proj.weight -> model.model.layers.0.self_attn.q_proj.weight
    model.norm.weight -> model.model.norm.weight
    norm.weight -> model.model.norm.weight
"""


def ace_step_lm_converter(state_dict):
    """
    Convert ACE-Step LLM checkpoint keys to match Qwen3ForCausalLM state dict.

    参数 state_dict 是 DiskMap 类型。
    遍历时，key 是 key 名，state_dict[key] 获取实际值。
    """
    new_state_dict = {}
    model_prefix = "model."
    nested_prefix = "model.model."

    for key in state_dict:
        if key.startswith(nested_prefix):
            # Already has model.model., keep as is
            new_key = key
        elif key.startswith(model_prefix):
            # Has model., add another model.
            new_key = "model." + key
        else:
            # No prefix, add model.model.
            new_key = "model.model." + key
        new_state_dict[new_key] = state_dict[key]

    # Handle tied word embeddings: lm_head.weight shares with embed_tokens
    if "model.model.embed_tokens.weight" in new_state_dict:
        new_state_dict["model.lm_head.weight"] = new_state_dict["model.model.embed_tokens.weight"]

    return new_state_dict
