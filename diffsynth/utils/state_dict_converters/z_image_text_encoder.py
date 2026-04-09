def ZImageTextEncoderStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name != "lm_head.weight":
            state_dict_[name] = state_dict[name]
    return state_dict_