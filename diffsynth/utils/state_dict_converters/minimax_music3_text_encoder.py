def MiniMaxMusic3TextEncoderStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("model.audio_extra_embedding") or name.startswith("model.audio_decoder."):
            continue
        state_dict_["model." + name] = state_dict[name]
    return state_dict_
