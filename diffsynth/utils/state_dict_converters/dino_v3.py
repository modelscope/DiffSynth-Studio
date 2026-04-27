def DINOv3StateDictConverter(state_dict):
    new_state_dict = {}
    for key in state_dict:
        value = state_dict[key]
        if key.startswith("layer"):
            new_state_dict["model." + key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict
