def MiniMaxH3VDNBranchStateDictConverter(state_dict):
    # The released linear_branch checkpoint is keyed by the post-transform diffusers
    # parameter names (transformer_blocks.{L}.attn.*); the DiffSynth DiT nests its blocks
    # under `blocks`. `state_dict` may be a lazy DiskMap, so iterate keys and index.
    state_dict_ = {}
    for name in state_dict:
        value = state_dict[name]
        if name.startswith("transformer_blocks."):
            name = "blocks." + name[len("transformer_blocks."):]
        state_dict_[name] = value
    return state_dict_
