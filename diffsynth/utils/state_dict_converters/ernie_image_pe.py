def ErnieImagePEStateDictConverter(state_dict):
    """
    Maps checkpoint keys to Ministral3ForCausalLM format.

    Checkpoint keys:
        model.layers.0.self_attn.q_proj.weight  → model.model.layers.0.self_attn.q_proj.weight
        model.embed_tokens.weight               → model.model.embed_tokens.weight
        model.norm.weight                       → model.model.norm.weight
        lm_head.weight                          → model.lm_head.weight
    """
    new_state_dict = {}
    for key in state_dict:
        if key.startswith("model.layers.") or key.startswith("model.embed") or key.startswith("model.norm"):
            new_key = key.replace("model.", "model.model.", 1)
            new_state_dict[new_key] = state_dict[key]
        elif key.startswith("lm_head."):
            new_key = "model." + key
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict
