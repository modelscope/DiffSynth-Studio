"""
State dict converter for ACE-Step Text Encoder (Qwen3-Embedding-0.6B).

The safetensors stores Qwen3Model weights with keys:
    embed_tokens.weight
    layers.0.self_attn.q_proj.weight
    norm.weight

AceStepTextEncoder wraps a .model attribute (Qwen3Model), so its
state_dict() has keys with 'model.' prefix:
    model.embed_tokens.weight
    model.layers.0.self_attn.q_proj.weight
    model.norm.weight

This converter adds 'model.' prefix to match the nested structure.
"""


def ace_step_text_encoder_converter(state_dict):
    """
    Convert ACE-Step Text Encoder checkpoint keys to match Qwen3Model wrapped state dict.

    参数 state_dict 是 DiskMap 类型。
    遍历时，key 是 key 名，state_dict[key] 获取实际值。
    """
    new_state_dict = {}
    prefix = "model."
    nested_prefix = "model.model."

    for key in state_dict:
        if key.startswith(nested_prefix):
            new_key = key
        elif key.startswith(prefix):
            new_key = "model." + key
        else:
            new_key = "model." + key
        new_state_dict[new_key] = state_dict[key]

    return new_state_dict
