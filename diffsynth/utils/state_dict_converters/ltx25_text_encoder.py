def LTX25TextEncoderStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("model."):
            new_name = "model.model.language_model." + name.removeprefix("model.")
        elif name.startswith("vision_model."):
            new_name = "model.model.embed_vision." + name.removeprefix("vision_model.")
        elif name.startswith("multi_modal_projector."):
            new_name = "model.model.embed_vision.multimodal_embedder." + name.removeprefix("multi_modal_projector.")
        elif name.startswith("audio_projector."):
            new_name = "model.model.embed_audio." + name.removeprefix("audio_projector.")
        else:
            continue
        state_dict_[new_name] = state_dict[name]
    state_dict_["model.lm_head.weight"] = state_dict_["model.model.language_model.embed_tokens.weight"]
    return state_dict_
