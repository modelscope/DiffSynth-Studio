def MiniMaxMusic3VocoderStateDictConverter(state_dict):
    """Select the decoder weights from ``dav.pth`` (``dec_in_proj.*`` and ``decoder.model.*``).

    Drops the unused encoder and normalizing-flow weights. No parameter is renamed.
    """
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("dec_in_proj.") or name.startswith("decoder.model."):
            state_dict_[name] = state_dict[name]
    return state_dict_
