import torch
from transformers import T5EncoderModel, T5Config
from .sd_text_encoder import SDTextEncoder



class FluxTextEncoder2(T5EncoderModel):
    def __init__(self, config):
        super().__init__(config)
        self.eval()

    def forward(self, input_ids):
        outputs = super().forward(input_ids=input_ids)
        prompt_emb = outputs.last_hidden_state
        return prompt_emb

    @staticmethod
    def state_dict_converter():
        return FluxTextEncoder2StateDictConverter()



class FluxTextEncoder2StateDictConverter():
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        state_dict_ = state_dict
        return state_dict_

    def from_civitai(self, state_dict):
        return self.from_diffusers(state_dict)
