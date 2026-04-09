def WanVideoDiTFromDiffusers(state_dict):
    rename_dict = {
        "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
        "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
        "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
        "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
        "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
        "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
        "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
        "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
        "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
        "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
        "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
        "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
        "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
        "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
        "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
        "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
        "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
        "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
        "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
        "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
        "blocks.0.attn2.add_k_proj.bias":"blocks.0.cross_attn.k_img.bias",
        "blocks.0.attn2.add_k_proj.weight":"blocks.0.cross_attn.k_img.weight",
        "blocks.0.attn2.add_v_proj.bias":"blocks.0.cross_attn.v_img.bias",
        "blocks.0.attn2.add_v_proj.weight":"blocks.0.cross_attn.v_img.weight",
        "blocks.0.attn2.norm_added_k.weight":"blocks.0.cross_attn.norm_k_img.weight",
        "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
        "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
        "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
        "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
        "blocks.0.norm2.bias": "blocks.0.norm3.bias",
        "blocks.0.norm2.weight": "blocks.0.norm3.weight",
        "blocks.0.scale_shift_table": "blocks.0.modulation",
        "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
        "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
        "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
        "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
        "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
        "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
        "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
        "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
        "condition_embedder.time_proj.bias": "time_projection.1.bias",
        "condition_embedder.time_proj.weight": "time_projection.1.weight",
        "condition_embedder.image_embedder.ff.net.0.proj.bias":"img_emb.proj.1.bias",
        "condition_embedder.image_embedder.ff.net.0.proj.weight":"img_emb.proj.1.weight",
        "condition_embedder.image_embedder.ff.net.2.bias":"img_emb.proj.3.bias",
        "condition_embedder.image_embedder.ff.net.2.weight":"img_emb.proj.3.weight",
        "condition_embedder.image_embedder.norm1.bias":"img_emb.proj.0.bias",
        "condition_embedder.image_embedder.norm1.weight":"img_emb.proj.0.weight",
        "condition_embedder.image_embedder.norm2.bias":"img_emb.proj.4.bias",
        "condition_embedder.image_embedder.norm2.weight":"img_emb.proj.4.weight",
        "patch_embedding.bias": "patch_embedding.bias",
        "patch_embedding.weight": "patch_embedding.weight",
        "scale_shift_table": "head.modulation",
        "proj_out.bias": "head.head.bias",
        "proj_out.weight": "head.head.weight",
    }
    state_dict_ = {}
    for name in state_dict:
        if name in rename_dict:
            state_dict_[rename_dict[name]] = state_dict[name]
        else:
            name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
            if name_ in rename_dict:
                name_ = rename_dict[name_]
                name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                state_dict_[name_] = state_dict[name]
    return state_dict_


def WanVideoDiTStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("vace"):
            continue
        if name.split(".")[0] in ["pose_patch_embedding", "face_adapter", "face_encoder", "motion_encoder"]:
            continue
        name_ = name
        if name_.startswith("model."):
            name_ = name_[len("model."):]
        state_dict_[name_] = state_dict[name]
    return state_dict_
