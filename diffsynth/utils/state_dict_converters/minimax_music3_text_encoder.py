def MiniMaxMusic3TextEncoderStateDictConverter(state_dict):
    """Select the Qwen3 backbone weights from the qwen checkpoint for the wrapper.

    Drops the RVQ depth-decoder and audio-embedding weights, then prepends the ``model.`` wrapper prefix
    (the backbone lives under ``self.model`` of the wrapper). Inner Qwen3 parameter names are unchanged.
    """
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("model.audio_extra_embedding") or name.startswith("model.audio_decoder."):
            continue
        state_dict_["model." + name] = state_dict[name]
    return state_dict_
