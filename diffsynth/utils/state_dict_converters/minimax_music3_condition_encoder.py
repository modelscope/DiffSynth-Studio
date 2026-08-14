def MiniMaxMusic3ConditionEncoderStateDictConverter(state_dict):
    """Select the condition-encoder weights from ``flowmatching_vae.pth``.

    Keeps ``cond_layer_logits``, ``cond_layer_scale`` and ``latent_conditioners.*`` unchanged;
    drops the DiT weights sharing the same file. No parameter is renamed.
    """
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("cond_layer_logits") or name.startswith("cond_layer_scale") or name.startswith("latent_conditioners."):
            state_dict_[name] = state_dict[name]
    return state_dict_
