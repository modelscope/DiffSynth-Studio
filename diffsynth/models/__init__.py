import torch
from safetensors import safe_open

from .sd_text_encoder import SDTextEncoder
from .sd_unet import SDUNet
from .sd_vae_encoder import SDVAEEncoder
from .sd_vae_decoder import SDVAEDecoder

from .sdxl_text_encoder import SDXLTextEncoder, SDXLTextEncoder2
from .sdxl_unet import SDXLUNet
from .sdxl_vae_decoder import SDXLVAEDecoder
from .sdxl_vae_encoder import SDXLVAEEncoder


class ModelManager:
    def __init__(self, torch_type=torch.float16, device="cuda"):
        self.torch_type = torch_type
        self.device = device
        self.model = {}

    def is_stabe_diffusion_xl(self, state_dict):
        param_name = "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight"
        return param_name in state_dict

    def is_stable_diffusion(self, state_dict):
        return True
    
    def load_stable_diffusion(self, state_dict, components=None):
        component_dict = {
            "text_encoder": SDTextEncoder,
            "unet": SDUNet,
            "vae_decoder": SDVAEDecoder,
            "vae_encoder": SDVAEEncoder,
            "refiner": SDXLUNet,
        }
        if components is None:
            components = ["text_encoder", "unet", "vae_decoder", "vae_encoder"]
        for component in components:
            self.model[component] = component_dict[component]()
            self.model[component].load_state_dict(self.model[component].state_dict_converter().from_civitai(state_dict))
            self.model[component].to(self.torch_type).to(self.device)

    def load_stable_diffusion_xl(self, state_dict, components=None):
        component_dict = {
            "text_encoder": SDXLTextEncoder,
            "text_encoder_2": SDXLTextEncoder2,
            "unet": SDXLUNet,
            "vae_decoder": SDXLVAEDecoder,
            "vae_encoder": SDXLVAEEncoder,
            "refiner": SDXLUNet,
        }
        if components is None:
            components = ["text_encoder", "text_encoder_2", "unet", "vae_decoder", "vae_encoder"]
        for component in components:
            self.model[component] = component_dict[component]()
            self.model[component].load_state_dict(self.model[component].state_dict_converter().from_civitai(state_dict))
            if component in ["vae_decoder", "vae_encoder"]:
                # These two model will output nan when float16 is enabled.
                # The precision problem happens in the last three resnet blocks.
                # I do not know how to solve this problem.
                self.model[component].to(torch.float32).to(self.device)
            else:
                self.model[component].to(self.torch_type).to(self.device)
        
    def load_from_safetensors(self, file_path, components=None):
        state_dict = load_state_dict_from_safetensors(file_path)
        if self.is_stabe_diffusion_xl(state_dict):
            self.load_stable_diffusion_xl(state_dict, components=components)
        elif self.is_stable_diffusion(state_dict):
            self.load_stable_diffusion(state_dict, components=components)

    def to(self, device):
        for component in self.model:
            self.model[component].to(device)
    
    def __getattr__(self, __name):
        if __name in self.model:
            return self.model[__name]
        else:
            return super.__getattribute__(__name)


def load_state_dict_from_safetensors(file_path):
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    return state_dict


def load_state_dict_from_bin(file_path):
    return torch.load(file_path, map_location="cpu")


def search_parameter(param, state_dict):
    for name, param_ in state_dict.items():
        if param.numel() == param_.numel():
            if param.shape == param_.shape:
                if torch.dist(param, param_) < 1e-6:
                    return name
            else:
                if torch.dist(param.flatten(), param_.flatten()) < 1e-6:
                    return name
    return None


def build_rename_dict(source_state_dict, target_state_dict, split_qkv=False):
    matched_keys = set()
    with torch.no_grad():
        for name in source_state_dict:
            rename = search_parameter(source_state_dict[name], target_state_dict)
            if rename is not None:
                print(f'"{name}": "{rename}",')
                matched_keys.add(rename)
            elif split_qkv and len(source_state_dict[name].shape)>=1 and source_state_dict[name].shape[0]%3==0:
                length = source_state_dict[name].shape[0] // 3
                rename = []
                for i in range(3):
                    rename.append(search_parameter(source_state_dict[name][i*length: i*length+length], target_state_dict))
                if None not in rename:
                    print(f'"{name}": {rename},')
                    for rename_ in rename:
                        matched_keys.add(rename_)
    for name in target_state_dict:
        if name not in matched_keys:
            print("Cannot find", name, target_state_dict[name].shape)
