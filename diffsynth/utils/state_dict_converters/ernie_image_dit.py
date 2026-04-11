def ErnieImageDiTStateDictConverter(state_dict):
    state_dict_ = {name: state_dict[name] for name in state_dict}
    return state_dict_
