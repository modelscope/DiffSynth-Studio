import torch


def Krea2TextEncoderStateDictConverter(state_dict):
    state_dict_ = {}
    for key in state_dict:
        if key.startswith("model.language_model."):
            new_key = "model.model." + key[len("model."):]
        elif key.startswith("model.visual."):
            new_key = "model.model." + key[len("model."):]
        else:
            new_key = key
        state_dict_[new_key] = state_dict[key]
    if "model.lm_head.weight" not in state_dict_:
        state_dict_["model.lm_head.weight"] = state_dict_.get(
            "model.model.language_model.embed_tokens.weight",
            torch.zeros(151936, 2560),
        )
    return state_dict_
