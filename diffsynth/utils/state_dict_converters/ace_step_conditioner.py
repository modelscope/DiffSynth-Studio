def AceStepConditionEncoderStateDictConverter(state_dict):
    new_state_dict = {}
    prefix = "encoder."

    for key in state_dict:
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = state_dict[key]

    if "null_condition_emb" in state_dict:
        new_state_dict["null_condition_emb"] = state_dict["null_condition_emb"]

    return new_state_dict
