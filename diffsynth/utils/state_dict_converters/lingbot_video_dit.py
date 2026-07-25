def LingBotVideoDiTStateDictConverter(state_dict):
    # The LingBot-Video DiT checkpoint (transformer/diffusion_pytorch_model.safetensors)
    # is saved by diffusers with module-hierarchy keys that already match
    # `LingBotVideoDiT` exactly (patch_embedder / time_embedder / time_modulation /
    # text_embedder / blocks.N.* / norm_out_modulation / proj_out). The only work
    # needed is stripping any wrapper prefix a repackaged/compiled checkpoint may add.
    prefixes = ["model.diffusion_model.", "_orig_mod.", "module.", "transformer."]
    state_dict_ = {}
    for name in state_dict:
        new_name = name
        for prefix in prefixes:
            if new_name.startswith(prefix):
                new_name = new_name[len(prefix):]
                break
        state_dict_[new_name] = state_dict[name]
    return state_dict_
