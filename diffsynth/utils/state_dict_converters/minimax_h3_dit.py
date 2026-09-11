def MiniMaxH3DiTSingularityStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("model.diffusion_model."):
            new_name = name.replace("model.diffusion_model.", "")
            state_dict_[new_name] = state_dict[name]
    return state_dict_
