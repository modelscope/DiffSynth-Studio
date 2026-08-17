def MiniMaxMusic3TextEncoderStateDictConverter(state_dict):
    return {"model." + name: state_dict[name] for name in state_dict}
