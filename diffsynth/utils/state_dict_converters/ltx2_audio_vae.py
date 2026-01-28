def LTX2AudioEncoderStateDictConverter(state_dict):
    # Not used
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("audio_vae.encoder."):
            new_name = name.replace("audio_vae.encoder.", "")
            state_dict_[new_name] = state_dict[name]
        elif name.startswith("audio_vae.per_channel_statistics."):
            new_name = name.replace("audio_vae.per_channel_statistics.", "per_channel_statistics.")
            state_dict_[new_name] = state_dict[name]
    return state_dict_


def LTX2AudioDecoderStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("audio_vae.decoder."):
            new_name = name.replace("audio_vae.decoder.", "")
            state_dict_[new_name] = state_dict[name]
        elif name.startswith("audio_vae.per_channel_statistics."):
            new_name = name.replace("audio_vae.per_channel_statistics.", "per_channel_statistics.")
            state_dict_[new_name] = state_dict[name]
    return state_dict_


def LTX2VocoderStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("vocoder."):
            new_name = name.replace("vocoder.", "")
            state_dict_[new_name] = state_dict[name]
    return state_dict_
