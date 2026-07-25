def LingBotVideoTextEncoderStateDictConverter(state_dict):
    # The LingBot-Video text encoder checkpoint stores a Qwen3VLForConditionalGeneration
    # with keys already under `model.language_model.*` and `model.visual.*`, which is
    # exactly the layout of the wrapped `Qwen3VLModel` (assigned to `self.model`). The
    # tied `lm_head` is not stored, so no key mapping is required — this is an identity
    # converter that only strips any wrapper prefix a repackaged checkpoint may add.
    prefixes = ["_orig_mod.", "module."]
    state_dict_ = {}
    for name in state_dict:
        new_name = name
        for prefix in prefixes:
            if new_name.startswith(prefix):
                new_name = new_name[len(prefix):]
                break
        state_dict_[new_name] = state_dict[name]
    return state_dict_
