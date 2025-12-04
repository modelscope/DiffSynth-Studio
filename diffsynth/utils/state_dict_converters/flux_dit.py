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