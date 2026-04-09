def ZImageDiTStateDictConverter(state_dict):
    state_dict_ = {name.replace("model.diffusion_model.", ""): state_dict[name] for name in state_dict}
    return state_dict_
