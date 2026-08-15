def MiniMaxMusic3VocoderStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("dec_in_proj.") or name.startswith("decoder.model."):
            state_dict_[name] = state_dict[name]
    return state_dict_
