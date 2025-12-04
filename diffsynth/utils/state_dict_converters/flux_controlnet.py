import torch


def FluxControlNetStateDictConverter(state_dict):
    global_rename_dict = {
        "context_embedder": "context_embedder",
        "x_embedder": "x_embedder",
        "time_text_embed.timestep_embedder.linear_1": "time_embedder.timestep_embedder.0",
        "time_text_embed.timestep_embedder.linear_2": "time_embedder.timestep_embedder.2",
        "time_text_embed.guidance_embedder.linear_1": "guidance_embedder.timestep_embedder.0",
        "time_text_embed.guidance_embedder.linear_2": "guidance_embedder.timestep_embedder.2",
        "time_text_embed.text_embedder.linear_1": "pooled_text_embedder.0",
        "time_text_embed.text_embedder.linear_2": "pooled_text_embedder.2",
        "norm_out.linear": "final_norm_out.linear",
        "proj_out": "final_proj_out",
    }
    rename_dict = {
        "proj_out": "proj_out",
        "norm1.linear": "norm1_a.linear",
        "norm1_context.linear": "norm1_b.linear",
        "attn.to_q": "attn.a_to_q",
        "attn.to_k": "attn.a_to_k",
        "attn.to_v": "attn.a_to_v",
        "attn.to_out.0": "attn.a_to_out",
        "attn.add_q_proj": "attn.b_to_q",
        "attn.add_k_proj": "attn.b_to_k",
        "attn.add_v_proj": "attn.b_to_v",
        "attn.to_add_out": "attn.b_to_out",
        "ff.net.0.proj": "ff_a.0",
        "ff.net.2": "ff_a.2",
        "ff_context.net.0.proj": "ff_b.0",
        "ff_context.net.2": "ff_b.2",
        "attn.norm_q": "attn.norm_q_a",
        "attn.norm_k": "attn.norm_k_a",
        "attn.norm_added_q": "attn.norm_q_b",
        "attn.norm_added_k": "attn.norm_k_b",
    }
    rename_dict_single = {
        "attn.to_q": "a_to_q",
        "attn.to_k": "a_to_k",
        "attn.to_v": "a_to_v",
        "attn.norm_q": "norm_q_a",
        "attn.norm_k": "norm_k_a",
        "norm.linear": "norm.linear",
        "proj_mlp": "proj_in_besides_attn",
        "proj_out": "proj_out",
    }
    state_dict_ = {}

    for name in state_dict:
        param = state_dict[name]
        if name.endswith(".weight") or name.endswith(".bias"):
            suffix = ".weight" if name.endswith(".weight") else ".bias"
            prefix = name[:-len(suffix)]
            if prefix in global_rename_dict:
                state_dict_[global_rename_dict[prefix] + suffix] = param
            elif prefix.startswith("transformer_blocks."):
                names = prefix.split(".")
                names[0] = "blocks"
                middle = ".".join(names[2:])
                if middle in rename_dict:
                    name_ = ".".join(names[:2] + [rename_dict[middle]] + [suffix[1:]])
                    state_dict_[name_] = param
            elif prefix.startswith("single_transformer_blocks."):
                names = prefix.split(".")
                names[0] = "single_blocks"
                middle = ".".join(names[2:])
                if middle in rename_dict_single:
                    name_ = ".".join(names[:2] + [rename_dict_single[middle]] + [suffix[1:]])
                    state_dict_[name_] = param
                else:
                    state_dict_[name] = param
            else:
                state_dict_[name] = param
    for name in list(state_dict_.keys()):
        if ".proj_in_besides_attn." in name:
            name_ = name.replace(".proj_in_besides_attn.", ".to_qkv_mlp.")
            param = torch.concat([
                state_dict_[name.replace(".proj_in_besides_attn.", f".a_to_q.")],
                state_dict_[name.replace(".proj_in_besides_attn.", f".a_to_k.")],
                state_dict_[name.replace(".proj_in_besides_attn.", f".a_to_v.")],
                state_dict_[name],
            ], dim=0)
            state_dict_[name_] = param
            state_dict_.pop(name.replace(".proj_in_besides_attn.", f".a_to_q."))
            state_dict_.pop(name.replace(".proj_in_besides_attn.", f".a_to_k."))
            state_dict_.pop(name.replace(".proj_in_besides_attn.", f".a_to_v."))
            state_dict_.pop(name)
    for name in list(state_dict_.keys()):
        for component in ["a", "b"]:
            if f".{component}_to_q." in name:
                name_ = name.replace(f".{component}_to_q.", f".{component}_to_qkv.")
                param = torch.concat([
                    state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_q.")],
                    state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_k.")],
                    state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_v.")],
                ], dim=0)
                state_dict_[name_] = param
                state_dict_.pop(name.replace(f".{component}_to_q.", f".{component}_to_q."))
                state_dict_.pop(name.replace(f".{component}_to_q.", f".{component}_to_k."))
                state_dict_.pop(name.replace(f".{component}_to_q.", f".{component}_to_v."))
    
    return state_dict_