"""
State dict converter for ACE-Step DiT model.

The original checkpoint stores all model weights in a single file
(nested in AceStepConditionGenerationModel). The DiT weights are
prefixed with 'decoder.'.

This converter extracts only keys starting with 'decoder.' and strips
the prefix to match the standalone AceStepDiTModel in DiffSynth.
"""


def ace_step_dit_converter(state_dict):
    """
    Convert ACE-Step DiT checkpoint keys to DiffSynth format.

    参数 state_dict 是 DiskMap 类型。
    遍历时，key 是 key 名，state_dict[key] 获取实际值。

    Original checkpoint contains all model weights under prefixes:
    - decoder.* (DiT)
    - encoder.* (Conditioner)
    - tokenizer.* (Audio Tokenizer)
    - detokenizer.* (Audio Detokenizer)
    - null_condition_emb (CFG null embedding)

    This extracts only 'decoder.' keys and strips the prefix.

    Example mapping:
        decoder.layers.0.self_attn.q_proj.weight -> layers.0.self_attn.q_proj.weight
        decoder.proj_in.0.linear_1.weight -> proj_in.0.linear_1.weight
        decoder.time_embed.linear_1.weight -> time_embed.linear_1.weight
        decoder.rotary_emb.inv_freq -> rotary_emb.inv_freq
    """
    new_state_dict = {}
    prefix = "decoder."

    for key in state_dict:
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = state_dict[key]

    return new_state_dict
