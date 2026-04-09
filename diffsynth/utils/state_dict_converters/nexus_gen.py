def NexusGenAutoregressiveModelStateDictConverter(state_dict):
    new_state_dict = {}
    for key in state_dict:
        value = state_dict[key]
        new_state_dict["model." + key] = value
    return new_state_dict