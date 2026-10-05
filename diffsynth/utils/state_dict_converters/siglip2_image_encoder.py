def Siglip2ImageEncoderStateDictConverter(state_dict):
    return {"vision_model." + key: state_dict[key] for key in state_dict}
