def LTX2TextEncoderStateDictConverter(state_dict):
    state_dict_ = {}
    for key in state_dict:
        if key.startswith("language_model.model."):
            new_key = key.replace("language_model.model.", "model.language_model.")
        elif key.startswith("vision_tower."):
            new_key = key.replace("vision_tower.", "model.vision_tower.")
        elif key.startswith("multi_modal_projector."):
            new_key = key.replace("multi_modal_projector.", "model.multi_modal_projector.")
        elif key.startswith("language_model.lm_head."):
            new_key = key.replace("language_model.lm_head.", "lm_head.")
        else:
            continue
        state_dict_[new_key] = state_dict[key]
    state_dict_["lm_head.weight"] = state_dict_.get("model.language_model.embed_tokens.weight")
    return state_dict_


def LTX2TextEncoderPostModulesStateDictConverter(state_dict):
    state_dict_ = {}
    for key in state_dict:
        if key.startswith("text_embedding_projection."):
            new_key = key.replace("text_embedding_projection.", "feature_extractor_linear.")
        elif key.startswith("model.diffusion_model.video_embeddings_connector."):
            new_key = key.replace("model.diffusion_model.video_embeddings_connector.", "embeddings_connector.")
        elif key.startswith("model.diffusion_model.audio_embeddings_connector."):
            new_key = key.replace("model.diffusion_model.audio_embeddings_connector.", "audio_embeddings_connector.")
        else:
            continue
        state_dict_[new_key] = state_dict[key]
    return state_dict_
