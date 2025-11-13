def WanS2VAudioEncoderStateDictConverter(state_dict):
    state_dict = {'model.' + k: state_dict[k] for k in state_dict}
    return state_dict
