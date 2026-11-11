def MiniMaxH3ImageVAEStateDictConverter(state_dict):
    # The released MiniMax-H3-Image VAE checkpoint already uses the "original" MiniMax key names,
    # which is what `MiniMaxH3ImageVAE` uses. Only two adjustments are needed:
    #   * `decoder.mask_token` is never read (the decoder appends an all-zero token instead), so drop it;
    #   * `decoder.register_tokens` and the per-block `scale1` / `scale2` are bare `nn.Parameter`s in
    #     the checkpoint but `WarpedTensor` modules (i.e. `....weight`) in DiffSynth.
    state_dict_ = {}
    for name in state_dict:
        if name == "decoder.mask_token":
            continue
        if name in ["decoder.register_tokens"] or name.endswith(".scale1") or name.endswith(".scale2"):
            name_ = name + ".weight"
        else:
            name_ = name
        state_dict_[name_] = state_dict[name]
    return state_dict_
