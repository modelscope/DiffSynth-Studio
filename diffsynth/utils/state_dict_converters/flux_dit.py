import torch


def FluxDiTStateDictConverter(state_dict):
    is_nexus_gen = sum([key.startswith("pipe.dit.") for key in state_dict]) > 0
    if is_nexus_gen:
        dit_state_dict = {}
        for key in state_dict:
            if key.startswith('pipe.dit.'):
                param = state_dict[key]
                new_key = key.replace("pipe.dit.", "")
                if new_key.startswith("final_norm_out.linear."):
                    param = torch.concat([param[3072:], param[:3072]], dim=0)
                dit_state_dict[new_key] = param
        return dit_state_dict

    rename_dict = {
        "time_in.in_layer.bias": "time_embedder.timestep_embedder.0.bias",
        "time_in.in_layer.weight": "time_embedder.timestep_embedder.0.weight",
        "time_in.out_layer.bias": "time_embedder.timestep_embedder.2.bias",
        "time_in.out_layer.weight": "time_embedder.timestep_embedder.2.weight",
        "txt_in.bias": "context_embedder.bias",
        "txt_in.weight": "context_embedder.weight",
        "vector_in.in_layer.bias": "pooled_text_embedder.0.bias",
        "vector_in.in_layer.weight": "pooled_text_embedder.0.weight",
        "vector_in.out_layer.bias": "pooled_text_embedder.2.bias",
        "vector_in.out_layer.weight": "pooled_text_embedder.2.weight",
        "final_layer.linear.bias": "final_proj_out.bias",
        "final_layer.linear.weight": "final_proj_out.weight",
        "guidance_in.in_layer.bias": "guidance_embedder.timestep_embedder.0.bias",
        "guidance_in.in_layer.weight": "guidance_embedder.timestep_embedder.0.weight",
        "guidance_in.out_layer.bias": "guidance_embedder.timestep_embedder.2.bias",
        "guidance_in.out_layer.weight": "guidance_embedder.timestep_embedder.2.weight",
        "img_in.bias": "x_embedder.bias",
        "img_in.weight": "x_embedder.weight",
        "final_layer.adaLN_modulation.1.weight": "final_norm_out.linear.weight",
        "final_layer.adaLN_modulation.1.bias": "final_norm_out.linear.bias",
    }
    suffix_rename_dict = {
        "img_attn.norm.key_norm.scale": "attn.norm_k_a.weight",
        "img_attn.norm.query_norm.scale": "attn.norm_q_a.weight",
        "img_attn.proj.bias": "attn.a_to_out.bias",
        "img_attn.proj.weight": "attn.a_to_out.weight",
        "img_attn.qkv.bias": "attn.a_to_qkv.bias",
        "img_attn.qkv.weight": "attn.a_to_qkv.weight",
        "img_mlp.0.bias": "ff_a.0.bias",
        "img_mlp.0.weight": "ff_a.0.weight",
        "img_mlp.2.bias": "ff_a.2.bias",
        "img_mlp.2.weight": "ff_a.2.weight",
        "img_mod.lin.bias": "norm1_a.linear.bias",
        "img_mod.lin.weight": "norm1_a.linear.weight",
        "txt_attn.norm.key_norm.scale": "attn.norm_k_b.weight",
        "txt_attn.norm.query_norm.scale": "attn.norm_q_b.weight",
        "txt_attn.proj.bias": "attn.b_to_out.bias",
        "txt_attn.proj.weight": "attn.b_to_out.weight",
        "txt_attn.qkv.bias": "attn.b_to_qkv.bias",
        "txt_attn.qkv.weight": "attn.b_to_qkv.weight",
        "txt_mlp.0.bias": "ff_b.0.bias",
        "txt_mlp.0.weight": "ff_b.0.weight",
        "txt_mlp.2.bias": "ff_b.2.bias",
        "txt_mlp.2.weight": "ff_b.2.weight",
        "txt_mod.lin.bias": "norm1_b.linear.bias",
        "txt_mod.lin.weight": "norm1_b.linear.weight",

        "linear1.bias": "to_qkv_mlp.bias",
        "linear1.weight": "to_qkv_mlp.weight",
        "linear2.bias": "proj_out.bias",
        "linear2.weight": "proj_out.weight",
        "modulation.lin.bias": "norm.linear.bias",
        "modulation.lin.weight": "norm.linear.weight",
        "norm.key_norm.scale": "norm_k_a.weight",
        "norm.query_norm.scale": "norm_q_a.weight",
    }
    state_dict_ = {}
    for name in state_dict:
        original_name = name
        if name.startswith("model.diffusion_model."):
            name = name[len("model.diffusion_model."):]
        names = name.split(".")
        if name in rename_dict:
            rename = rename_dict[name]
            state_dict_[rename] = state_dict[original_name]
        elif names[0] == "double_blocks":
            rename = f"blocks.{names[1]}." + suffix_rename_dict[".".join(names[2:])]
            state_dict_[rename] = state_dict[original_name]
        elif names[0] == "single_blocks":
            if ".".join(names[2:]) in suffix_rename_dict:
                rename = f"single_blocks.{names[1]}." + suffix_rename_dict[".".join(names[2:])]
                state_dict_[rename] = state_dict[original_name]
        else:
            pass
    return state_dict_


def FluxDiTStateDictConverterFromDiffusers(state_dict):
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
                if global_rename_dict[prefix] == "final_norm_out.linear":
                    param = torch.concat([param[3072:], param[:3072]], dim=0)
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
                    pass
            else:
                pass
    for name in list(state_dict_.keys()):
        if "single_blocks." in name and ".a_to_q." in name:
            mlp = state_dict_.get(name.replace(".a_to_q.", ".proj_in_besides_attn."), None)
            if mlp is None:
                mlp = torch.zeros(4 * state_dict_[name].shape[0],
                                    *state_dict_[name].shape[1:],
                                    dtype=state_dict_[name].dtype)
            else:
                state_dict_.pop(name.replace(".a_to_q.", ".proj_in_besides_attn."))
            param = torch.concat([
                state_dict_.pop(name),
                state_dict_.pop(name.replace(".a_to_q.", ".a_to_k.")),
                state_dict_.pop(name.replace(".a_to_q.", ".a_to_v.")),
                mlp,
            ], dim=0)
            name_ = name.replace(".a_to_q.", ".to_qkv_mlp.")
            state_dict_[name_] = param
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