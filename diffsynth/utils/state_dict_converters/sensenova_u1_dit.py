def SenseNovaU1DiTStateDictConverter(state_dict):
    # The checkpoint holds the DiT and the understanding-branch vision encoder together;
    # drop the vision encoder keys so the strict load only sees this model's parameters.
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("vision_model."):
            continue
        state_dict_[name] = state_dict[name]
    return state_dict_
