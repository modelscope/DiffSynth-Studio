def AceStepDiTModelStateDictConverter(state_dict):
    new_state_dict = {}
    prefix = "decoder."

    for key in state_dict:
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = state_dict[key]

    return new_state_dict
