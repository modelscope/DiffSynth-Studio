def QwenVideoEditDiTStateDictConverter(state_dict):
    return {k[len("pipe.dit."):]: state_dict[k] for k in state_dict if k.startswith("pipe.dit.")}


def QwenVideoEditAdapterStateDictConverter(state_dict):
    return {k: state_dict[k] for k in state_dict if k.startswith(("in_proj.", "out_proj."))}
