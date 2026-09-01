def LTX25DiffusionVideoDecoderStateDictConverter(state_dict):
    converted = {}
    for source_name in state_dict:
        if source_name.startswith("decoder."):
            name = source_name.removeprefix("decoder.")
        elif source_name.startswith("per_channel_statistics."):
            name = source_name
        else:
            continue

        if name == "type_emb" or name.startswith("coarse_") or name.endswith((".gate_msa", ".gate_mlp", ".gate_ctx")):
            continue
        name = name.replace("t_embedder.mlp.0.", "t_embedder.timestep_embedder.linear_1.")
        name = name.replace("t_embedder.mlp.2.", "t_embedder.timestep_embedder.linear_2.")
        value = state_dict[source_name]
        if name.endswith(".attn.qkv.weight") or name.endswith(".attn.qkv.bias"):
            if value.shape[0] % 3 != 0:
                raise ValueError(f"Fused QKV tensor has invalid leading dimension: {source_name} {tuple(value.shape)}")
            leaf = "weight" if name.endswith(".weight") else "bias"
            prefix = name[: -len(leaf)]
            q, k, v = value.chunk(3, dim=0)
            converted[f"{prefix}to_q.{leaf}"] = q
            converted[f"{prefix}to_k.{leaf}"] = k
            converted[f"{prefix}to_v.{leaf}"] = v
        else:
            converted[name] = value
    return converted
