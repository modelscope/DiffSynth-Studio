def MiniMaxMusic3DiTStateDictConverter(state_dict):
    prefix = "diffusion_transformer."
    state_dict_ = {}
    for name in state_dict:
        if name.startswith(prefix) and not name.endswith("rotary_pos_emb.inv_freq"):
            state_dict_[name[len(prefix):]] = state_dict[name]
    return state_dict_
