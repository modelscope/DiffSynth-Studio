def SenseNovaU1VisionEncoderStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if not name.startswith("vision_model."):
            continue
        state_dict_[name.replace("vision_model.", "", 1)] = state_dict[name]
    return state_dict_
