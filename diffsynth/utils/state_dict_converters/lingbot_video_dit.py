def LingBotVideoDiTStateDictConverter(state_dict):
    prefixes = ["model.diffusion_model.", "_orig_mod.", "module.", "transformer."]
    state_dict_ = {}
    for name in state_dict:
        new_name = name
        for prefix in prefixes:
            if new_name.startswith(prefix):
                new_name = new_name[len(prefix):]
                break
        state_dict_[new_name] = state_dict[name]
    return state_dict_
