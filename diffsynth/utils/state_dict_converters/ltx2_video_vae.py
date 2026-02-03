def LTX2VideoEncoderStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("vae.encoder."):
            new_name = name.replace("vae.encoder.", "")
            state_dict_[new_name] = state_dict[name]
        elif name.startswith("vae.per_channel_statistics."):
            new_name = name.replace("vae.per_channel_statistics.", "per_channel_statistics.")
            state_dict_[new_name] = state_dict[name]
    return state_dict_


def LTX2VideoDecoderStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("vae.decoder."):
            new_name = name.replace("vae.decoder.", "")
            state_dict_[new_name] = state_dict[name]
        elif name.startswith("vae.per_channel_statistics."):
            new_name = name.replace("vae.per_channel_statistics.", "per_channel_statistics.")
            state_dict_[new_name] = state_dict[name]
    return state_dict_