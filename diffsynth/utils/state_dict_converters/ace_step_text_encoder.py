def AceStepTextEncoderStateDictConverter(state_dict):
    new_state_dict = {}
    prefix = "model."
    nested_prefix = "model.model."

    for key in state_dict:
        if key.startswith(nested_prefix):
            new_key = key
        elif key.startswith(prefix):
            new_key = "model." + key
        else:
            new_key = "model." + key
        new_state_dict[new_key] = state_dict[key]

    return new_state_dict
