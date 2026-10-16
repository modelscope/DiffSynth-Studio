def Flux3DiTFromDiffusers(state_dict):
    rename_dict = {
        "context_embedder.weight": "txt_in.weight",
        "x_embedders.image.weight": "emb_in.image.weight",
        "x_embedders.image_cond.weight": "emb_in.image_cond.weight",
        "timestep_embedder.linear_1.weight": "time_in.in_layer.weight",
        "timestep_embedder.linear_2.weight": "time_in.out_layer.weight",
        "target_timestep_embedder.linear_1.weight": "ref_time_in.in_layer.weight",
        "target_timestep_embedder.linear_2.weight": "ref_time_in.out_layer.weight",
        "norm_out.linear.weight": "final_layer.image.adaLN_modulation.1.weight",
        "proj_out.weight": "final_layer.image.linear.weight",
    }
    for mode in ("image", "image_cond", "txt"):
        rename_dict[f"early_stream_modulations.{mode}.linear.weight"] = f"early_stream_modulations.{mode}.lin.weight"
        rename_dict[f"single_stream_modulations.{mode}.linear.weight"] = f"single_stream_modulations.{mode}.lin.weight"
    block_names = [(f"mode_blocks.{mode}.{i}", f"content_mode_blocks.{mode}.{i}") for mode in ("image", "image_cond") for i in range(5)]
    block_names += [(f"txt_mode_blocks.{i}", f"txt_mode_blocks.{i}") for i in range(5)]
    block_names += [(f"single_transformer_blocks.{i}", f"single_blocks.{i}") for i in range(28)]
    for name, name_ in block_names:
        rename_dict.update({
            f"{name}.attn.to_q.weight": f"{name_}.q_proj.weight",
            f"{name}.attn.to_k.weight": f"{name_}.k_proj.weight",
            f"{name}.attn.to_v.weight": f"{name_}.v_proj.weight",
            f"{name}.attn.to_out.0.weight": f"{name_}.attn_out.weight",
            f"{name}.attn.norm_q.weight": f"{name_}.norm.query_norm.scale",
            f"{name}.attn.norm_k.weight": f"{name_}.norm.key_norm.scale",
            f"{name}.ff.linear_in.weight": f"{name_}.mlp_in.weight",
            f"{name}.ff.linear_out.weight": f"{name_}.mlp_out.weight",
        })
    state_dict_ = {}
    for name in state_dict:
        state_dict_[rename_dict.get(name, name)] = state_dict[name]
    return state_dict_
