from .sd_vae_decoder import SDVAEDecoder, SDVAEDecoderStateDictConverter


class SDXLVAEDecoder(SDVAEDecoder):
    def __init__(self, upcast_to_float32=True):
        super().__init__()
        self.scaling_factor = 0.13025

    @staticmethod
    def state_dict_converter():
        return SDXLVAEDecoderStateDictConverter()
    

class SDXLVAEDecoderStateDictConverter(SDVAEDecoderStateDictConverter):
    def __init__(self):
        super().__init__()

    def from_diffusers(self, state_dict):
        state_dict = super().from_diffusers(state_dict)
        return state_dict, {"upcast_to_float32": True}
    
    def from_civitai(self, state_dict):
        state_dict = super().from_civitai(state_dict)
        return state_dict, {"upcast_to_float32": True}
