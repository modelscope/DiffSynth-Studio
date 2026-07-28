def LingBotVideoTextEncoderStateDictConverter(state_dict):
    prefixes = ["_orig_mod.", "module."]
    state_dict_ = {}
    for name in state_dict:
        new_name = name
        for prefix in prefixes:
            if new_name.startswith(prefix):
                new_name = new_name[len(prefix):]
                break
        state_dict_[new_name] = state_dict[name]
    return state_dict_
