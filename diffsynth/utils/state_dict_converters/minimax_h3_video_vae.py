def MiniMaxH3VideoVAEStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name == "decoder.mask_token":
            continue
        if name in ["decoder.register_tokens"] or name.endswith(".scale1") or name.endswith(".scale2"):
            name_ = name + ".weight"
        else:
            name_ = name
        state_dict_[name_] = state_dict[name]
    return state_dict_
