def SDVAEStateDictConverter(state_dict):
    new_state_dict = {}
    for key in state_dict:
        if ".query." in key:
            new_key = key.replace(".query.", ".to_q.")
            new_state_dict[new_key] = state_dict[key]
        elif ".key." in key:
            new_key = key.replace(".key.", ".to_k.")
            new_state_dict[new_key] = state_dict[key]
        elif ".value." in key:
            new_key = key.replace(".value.", ".to_v.")
            new_state_dict[new_key] = state_dict[key]
        elif ".proj_attn." in key:
            new_key = key.replace(".proj_attn.", ".to_out.0.")
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict
