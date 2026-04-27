def SDTextEncoderStateDictConverter(state_dict):
    new_state_dict = {}
    for key in state_dict:
        if key.startswith("text_model.") and "position_ids" not in key:
            new_key = "model." + key
            new_state_dict[new_key] = state_dict[key]
    return new_state_dict
