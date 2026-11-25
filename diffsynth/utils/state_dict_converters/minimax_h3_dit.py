import torch


_DWD_PREFIXES = ("dwd_norm_out.", "dwd_pixel_refiner.")


def MiniMaxH3DiTSingularityStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("model.diffusion_model."):
            new_name = name.replace("model.diffusion_model.", "")
            state_dict_[new_name] = state_dict[name]
    return state_dict_


def minimax_h3_diffusers_parameter_name(name):
    projections = {
        "proj_in": "video_patch_proj",
        "audio_proj_in": "audio_patch_proj",
        "context_embedder": "condition_proj",
        "proj_out": "final_layer.video_out",
        "audio_proj_out": "final_layer.audio_out",
    }
    prefix, _, tail = name.partition(".")
    if prefix in projections:
        return f"{projections[prefix]}.{tail}"
    if name == "time_embedder.table":
        return "adaln_t_table"
    if prefix == "norm_out":
        tail = tail if tail.startswith("norm.") else "adaln_proj." + tail
        return "final_layer." + tail.replace("folded_bias", "linear.bias")

    name = name.replace("transformer_blocks.", "blocks.", 1)
    name = name.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.", 1)
    for source, target in (
        ("adaln_proj.folded_bias", "adaln_proj.linear.bias"),
        ("attn.to_q.", "attn.qkv_proj."),
        ("attn.to_out.0", "attn.out_proj"),
        ("attn.norm_q", "attn.q_norm"),
        ("attn.norm_k", "attn.k_norm"),
        ("ff.net.0.proj", "mlp.fc1"),
        ("ff.net.2", "mlp.fc2"),
    ):
        name = name.replace(source, target)
    return name


def MiniMaxH3DiTPrunedDiffusersStateDictConverter(state_dict):
    table = state_dict["time_embedder.table"]
    result = {}
    for name in state_dict:
        if (
            name in ("adaln_basis", "adaln_mean")
            or name.startswith(_DWD_PREFIXES)
            or name.endswith((".attn.to_k.weight", ".attn.to_v.weight"))
            or ".lora_" in name
        ):
            continue
        if name.endswith(".folded_bias") and name.removesuffix("folded_bias") + "linear.bias" in state_dict:
            continue
        value = table if name == "time_embedder.table" else state_dict[name]
        if name.endswith(".attn.to_q.weight"):
            prefix = name.removesuffix("to_q.weight")
            value = torch.cat((value, state_dict[prefix + "to_k.weight"], state_dict[prefix + "to_v.weight"]), dim=0)
        elif name.endswith(".ff.net.0.proj.weight"):
            up, gate = value.chunk(2, dim=0)
            value = torch.cat((gate, up), dim=0)
        result[minimax_h3_diffusers_parameter_name(name)] = value
    steps = torch.arange(16, dtype=torch.float32, device=table.device)
    result["rope.inv_freq"] = 1.0 / (10000.0 ** (steps / 16))
    return result


def MiniMaxH3PiDWDStateDictConverter(state_dict):
    return {name: state_dict[name] for name in state_dict if name.startswith(_DWD_PREFIXES)}
