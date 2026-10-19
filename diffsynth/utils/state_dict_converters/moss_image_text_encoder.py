def MOSSImageTextEncoderStateDictConverter(state_dict):
    new_state_dict = {}
    for key in state_dict:
        if key.startswith("language_model."):
            new_key = key.replace("language_model.", "model.", 1)
            new_state_dict[new_key] = state_dict[key]
    return new_state_dict
