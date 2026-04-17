"""
State dict converter for ACE-Step Conditioner model.

The original checkpoint stores all model weights in a single file
(nested in AceStepConditionGenerationModel). The Conditioner weights are
prefixed with 'encoder.'.

This converter extracts only keys starting with 'encoder.' and strips
the prefix to match the standalone AceStepConditionEncoder in DiffSynth.
"""


def ace_step_conditioner_converter(state_dict):
    """
    Convert ACE-Step Conditioner checkpoint keys to DiffSynth format.

    参数 state_dict 是 DiskMap 类型。
    遍历时，key 是 key 名，state_dict[key] 获取实际值。

    Original checkpoint contains all model weights under prefixes:
    - decoder.* (DiT)
    - encoder.* (Conditioner)
    - tokenizer.* (Audio Tokenizer)
    - detokenizer.* (Audio Detokenizer)
    - null_condition_emb (CFG null embedding)

    This extracts only 'encoder.' keys and strips the prefix.

    Example mapping:
        encoder.lyric_encoder.layers.0.self_attn.q_proj.weight -> lyric_encoder.layers.0.self_attn.q_proj.weight
        encoder.attention_pooler.layers.0.self_attn.q_proj.weight -> attention_pooler.layers.0.self_attn.q_proj.weight
        encoder.timbre_encoder.layers.0.self_attn.q_proj.weight -> timbre_encoder.layers.0.self_attn.q_proj.weight
        encoder.audio_tokenizer.audio_acoustic_proj.weight -> audio_tokenizer.audio_acoustic_proj.weight
        encoder.detokenizer.layers.0.self_attn.q_proj.weight -> detokenizer.layers.0.self_attn.q_proj.weight
    """
    new_state_dict = {}
    prefix = "encoder."

    for key in state_dict:
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = state_dict[key]

    # Extract null_condition_emb from top level (used for CFG negative condition)
    if "null_condition_emb" in state_dict:
        new_state_dict["null_condition_emb"] = state_dict["null_condition_emb"]

    return new_state_dict
