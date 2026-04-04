def FluxTextEncoderT5StateDictConverter(state_dict):
    state_dict_ = {i: state_dict[i] for i in state_dict}
    state_dict_["encoder.embed_tokens.weight"] = state_dict["shared.weight"]
    return state_dict_
