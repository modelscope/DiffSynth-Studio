def LTX25DiffusionVideoDecoderStateDictConverter(state_dict):
    """Select DiffVAE weights and rename only legacy timestep-MLP keys.

    The flat decoder keeps fused ``qkv`` parameters, so this converter is safe for
    lazy ``DiskMap`` inputs and never reads tensor metadata or values.
    """
    converted = {}
    for source_name in state_dict:
        if source_name.startswith("decoder."):
            name = source_name.removeprefix("decoder.")
        elif source_name.startswith("per_channel_statistics."):
            name = source_name
        else:
            continue

        if name.startswith("coarse_") or name.endswith((".gate_msa", ".gate_mlp", ".gate_ctx")):
            continue
        name = name.replace("t_embedder.mlp.0.", "t_embedder.timestep_embedder.linear_1.")
        name = name.replace("t_embedder.mlp.2.", "t_embedder.timestep_embedder.linear_2.")
        converted[name] = state_dict[source_name]
    return converted
