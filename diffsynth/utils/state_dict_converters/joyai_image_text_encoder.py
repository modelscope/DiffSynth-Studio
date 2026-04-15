def JoyAIImageTextEncoderStateDictConverter(state_dict):
    """Convert HuggingFace Qwen3VL checkpoint keys to DiffSynth wrapper keys.

    Mapping (checkpoint -> wrapper):
    - lm_head.weight -> model.lm_head.weight
    - model.language_model.* -> model.model.language_model.*
    - model.visual.* -> model.model.visual.*
    """
    state_dict_ = {}
    for key in state_dict:
        if key == "lm_head.weight":
            new_key = "model.lm_head.weight"
        elif key.startswith("model.language_model."):
            new_key = "model.model." + key[len("model."):]
        elif key.startswith("model.visual."):
            new_key = "model.model." + key[len("model."):]
        else:
            new_key = key
        state_dict_[new_key] = state_dict[key]
    return state_dict_
