def FluxTextEncoderClipStateDictConverter(state_dict):
    rename_dict = {
        "text_model.embeddings.token_embedding.weight": "token_embedding.weight",
        "text_model.embeddings.position_embedding.weight": "position_embeds",
        "text_model.final_layer_norm.weight": "final_layer_norm.weight",
        "text_model.final_layer_norm.bias": "final_layer_norm.bias",
    }
    attn_rename_dict = {
        "self_attn.q_proj": "attn.to_q",
        "self_attn.k_proj": "attn.to_k",
        "self_attn.v_proj": "attn.to_v",
        "self_attn.out_proj": "attn.to_out",
        "layer_norm1": "layer_norm1",
        "layer_norm2": "layer_norm2",
        "mlp.fc1": "fc1",
        "mlp.fc2": "fc2",
    }
    state_dict_ = {}
    for name in state_dict:
        if name in rename_dict:
            param = state_dict[name]
            if name == "text_model.embeddings.position_embedding.weight":
                param = param.reshape((1, param.shape[0], param.shape[1]))
            state_dict_[rename_dict[name]] = param
        elif name.startswith("text_model.encoder.layers."):
            param = state_dict[name]
            names = name.split(".")
            layer_id, layer_type, tail = names[3], ".".join(names[4:-1]), names[-1]
            name_ = ".".join(["encoders", layer_id, attn_rename_dict[layer_type], tail])
            state_dict_[name_] = param
    return state_dict_
