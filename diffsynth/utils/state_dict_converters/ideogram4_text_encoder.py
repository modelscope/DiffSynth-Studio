def Ideogram4TextEncoderStateDictConverter(state_dict):
    # `Ideogram4TextEncoder` holds the checkpoint's model under `self.model`.
    return {"model." + key: state_dict[key] for key in state_dict}
