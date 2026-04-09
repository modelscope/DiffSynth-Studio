def JoyAIImageDiTStateDictConverter(state_dict):
    """Convert JoyAI-Image DiT checkpoint to model state dict.

    Handle:
    1. "model." prefix stripping from checkpoint
    2. FeedForward key mapping: diffusers uses "net.0.proj" / "net.2"
       while DiffSynth uses "proj" / "out_proj"
    """
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("model."):
            name = name[len("model."):]

        # Map diffusers FeedForward keys to DiffSynth keys
        # Pattern: double_blocks.N.{img_mlp|txt_mlp}.net.0.proj.* -> double_blocks.N.{img_mlp|txt_mlp}.proj.*
        new_name = name
        if ".net.0.proj." in name:
            new_name = name.replace(".net.0.proj.", ".proj.")
        elif ".net.2." in name:
            new_name = name.replace(".net.2.", ".out_proj.")

        state_dict_[new_name] = state_dict[name]

    return state_dict_
