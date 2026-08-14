def MiniMaxMusic3DiTStateDictConverter(state_dict):
    """Select the DiT weights from ``flowmatching_vae.pth`` and strip the ``diffusion_transformer.`` prefix.

    No inner parameter is renamed; the condition-encoder weights sharing the same file are dropped, as
    is the stored RoPE ``inv_freq``, which the rotary embedding derives from its own ``theta`` instead.
    """
    prefix = "diffusion_transformer."
    state_dict_ = {}
    for name in state_dict:
        if name.startswith(prefix) and not name.endswith("rotary_pos_emb.inv_freq"):
            state_dict_[name[len(prefix):]] = state_dict[name]
    return state_dict_
