def Qwen2ConnectorStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("connector."):
            name_ = name[len("connector."):]
            state_dict_[name_] = state_dict[name]
    return state_dict_