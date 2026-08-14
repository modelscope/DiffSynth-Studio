def MiniMaxMusic3RVQDepthDecoderStateDictConverter(state_dict):
    """Select the RVQ depth-decoder weights from the qwen checkpoint and strip the leading ``model.`` prefix.

    Keeps ``audio_extra_embedding.*`` and ``audio_decoder.*``; drops the global language-model weights.
    No inner parameter is renamed.
    """
    prefix = "model."
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("model.audio_extra_embedding.") or name.startswith("model.audio_decoder."):
            state_dict_[name[len(prefix):]] = state_dict[name]
    return state_dict_
