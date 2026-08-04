def MiniMaxH3AudioVAEStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name in ["pre_block.attn.q_bias", "pre_block.attn.v_bias", "pre_block.attn.zero_k_bias"]:
            name_ = name + ".weight"
        else:
            name_ = name
        state_dict_[name_] = state_dict[name]
    return state_dict_
