def MiniMaxMusic3RVQDepthDecoderStateDictConverter(state_dict):
    prefix = "model."
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("model.audio_extra_embedding.") or name.startswith("model.audio_decoder."):
            state_dict_[name[len(prefix):]] = state_dict[name]
    return state_dict_
