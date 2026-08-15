def MiniMaxMusic3ConditionEncoderStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("cond_layer_logits") or name.startswith("cond_layer_scale") or name.startswith("latent_conditioners."):
            state_dict_[name] = state_dict[name]
    return state_dict_
