def VaceWanModelDictConverter(state_dict):
    state_dict_ = {name: state_dict[name] for name in state_dict if name.startswith("vace")}
    return state_dict_
