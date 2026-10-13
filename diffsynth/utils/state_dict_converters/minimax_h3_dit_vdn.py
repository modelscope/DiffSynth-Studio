def MiniMaxH3VDNBranchStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        value = state_dict[name]
        if name.startswith("transformer_blocks."):
            name = "blocks." + name[len("transformer_blocks."):]
        state_dict_[name] = value
    return state_dict_
