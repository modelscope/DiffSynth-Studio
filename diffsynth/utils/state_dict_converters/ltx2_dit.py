def LTXModelStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("model.diffusion_model."):
            new_name = name.replace("model.diffusion_model.", "")
            if new_name.startswith("audio_embeddings_connector.") or new_name.startswith("video_embeddings_connector."):
                continue
            state_dict_[new_name] = state_dict[name]
    return state_dict_
