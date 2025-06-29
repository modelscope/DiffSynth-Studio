import torch
from diffsynth.lora import GeneralLoRALoader
from diffsynth.models.lora import FluxLoRAFromCivitai


class FluxLoRALoader(GeneralLoRALoader):
    def __init__(self, device="cpu", torch_dtype=torch.float32):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.loader = FluxLoRAFromCivitai()

    def load(self, model: torch.nn.Module, state_dict_lora, alpha=1.0):
        lora_prefix, model_resource = self.loader.match(model, state_dict_lora)
        self.loader.load(model, state_dict_lora, lora_prefix, alpha=alpha, model_resource=model_resource)