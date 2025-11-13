def WanImageEncoderStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("textual."):
            continue
        name_ = "model." + name
        state_dict_[name_] = state_dict[name]
    return state_dict_