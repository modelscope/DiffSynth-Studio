import os, torch, json, importlib
from typing import List

from .downloader import download_models, download_customized_models, Preset_model_id, Preset_model_website

from .sd_text_encoder import SDTextEncoder
from .sd_unet import SDUNet
from .sd_vae_encoder import SDVAEEncoder
from .sd_vae_decoder import SDVAEDecoder
from .lora import get_lora_loaders

from .sdxl_text_encoder import SDXLTextEncoder, SDXLTextEncoder2
from .sdxl_unet import SDXLUNet
from .sdxl_vae_decoder import SDXLVAEDecoder
from .sdxl_vae_encoder import SDXLVAEEncoder

from .sd3_text_encoder import SD3TextEncoder1, SD3TextEncoder2, SD3TextEncoder3
from .sd3_dit import SD3DiT
from .sd3_vae_decoder import SD3VAEDecoder
from .sd3_vae_encoder import SD3VAEEncoder

from .sd_controlnet import SDControlNet
from .sdxl_controlnet import SDXLControlNetUnion

from .sd_motion import SDMotionModel
from .sdxl_motion import SDXLMotionModel

from .svd_image_encoder import SVDImageEncoder
from .svd_unet import SVDUNet
from .svd_vae_decoder import SVDVAEDecoder
from .svd_vae_encoder import SVDVAEEncoder

from .sd_ipadapter import SDIpAdapter, IpAdapterCLIPImageEmbedder
from .sdxl_ipadapter import SDXLIpAdapter, IpAdapterXLCLIPImageEmbedder

from .hunyuan_dit_text_encoder import HunyuanDiTCLIPTextEncoder, HunyuanDiTT5TextEncoder
from .hunyuan_dit import HunyuanDiT
from .hunyuan_video_vae_decoder import HunyuanVideoVAEDecoder
from .hunyuan_video_vae_encoder import HunyuanVideoVAEEncoder

from .flux_dit import FluxDiT
from .flux_text_encoder import FluxTextEncoder2
from .flux_vae import FluxVAEEncoder, FluxVAEDecoder
from .flux_ipadapter import FluxIpAdapter

from .cog_vae import CogVAEEncoder, CogVAEDecoder
from .cog_dit import CogDiT

from ..extensions.RIFE import IFNet
from ..extensions.ESRGAN import RRDBNet

from ..configs.model_config import model_loader_configs, huggingface_model_loader_configs, patch_model_loader_configs
from .utils import load_state_dict, init_weights_on_device, hash_state_dict_keys, split_state_dict_with_prefix


def load_model_from_single_file(state_dict, model_names, model_classes, model_resource, torch_dtype, device):
    loaded_model_names, loaded_models = [], []
    for model_name, model_class in zip(model_names, model_classes):
        print(f"    model_name: {model_name} model_class: {model_class.__name__}")
        state_dict_converter = model_class.state_dict_converter()
        if model_resource == "civitai":
            state_dict_results = state_dict_converter.from_civitai(state_dict)
        elif model_resource == "diffusers":
            state_dict_results = state_dict_converter.from_diffusers(state_dict)
        if isinstance(state_dict_results, tuple):
            model_state_dict, extra_kwargs = state_dict_results
            print(f"        This model is initialized with extra kwargs: {extra_kwargs}")
        else:
            model_state_dict, extra_kwargs = state_dict_results, {}
        torch_dtype = torch.float32 if extra_kwargs.get("upcast_to_float32", False) else torch_dtype
        with init_weights_on_device():
            model = model_class(**extra_kwargs)
        if hasattr(model, "eval"):
            model = model.eval()
        model.load_state_dict(model_state_dict, assign=True)
        model = model.to(dtype=torch_dtype, device=device)
        loaded_model_names.append(model_name)
        loaded_models.append(model)
    return loaded_model_names, loaded_models


def load_model_from_huggingface_folder(file_path, model_names, model_classes, torch_dtype, device):
    loaded_model_names, loaded_models = [], []
    for model_name, model_class in zip(model_names, model_classes):
        if torch_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            model = model_class.from_pretrained(file_path, torch_dtype=torch_dtype).eval()
        else:
            model = model_class.from_pretrained(file_path).eval().to(dtype=torch_dtype)
        if torch_dtype == torch.float16 and hasattr(model, "half"):
            model = model.half()
        try:
            model = model.to(device=device)
        except:
            pass
        loaded_model_names.append(model_name)
        loaded_models.append(model)
    return loaded_model_names, loaded_models


def load_single_patch_model_from_single_file(state_dict, model_name, model_class, base_model, extra_kwargs, torch_dtype, device):
    print(f"    model_name: {model_name} model_class: {model_class.__name__} extra_kwargs: {extra_kwargs}")
    base_state_dict = base_model.state_dict()
    base_model.to("cpu")
    del base_model
    model = model_class(**extra_kwargs)
    model.load_state_dict(base_state_dict, strict=False)
    model.load_state_dict(state_dict, strict=False)
    model.to(dtype=torch_dtype, device=device)
    return model


def load_patch_model_from_single_file(state_dict, model_names, model_classes, extra_kwargs, model_manager, torch_dtype, device):
    loaded_model_names, loaded_models = [], []
    for model_name, model_class in zip(model_names, model_classes):
        while True:
            for model_id in range(len(model_manager.model)):
                base_model_name = model_manager.model_name[model_id]
                if base_model_name == model_name:
                    base_model_path = model_manager.model_path[model_id]
                    base_model = model_manager.model[model_id]
                    print(f"    Adding patch model to {base_model_name} ({base_model_path})")
                    patched_model = load_single_patch_model_from_single_file(
                        state_dict, model_name, model_class, base_model, extra_kwargs, torch_dtype, device)
                    loaded_model_names.append(base_model_name)
                    loaded_models.append(patched_model)
                    model_manager.model.pop(model_id)
                    model_manager.model_path.pop(model_id)
                    model_manager.model_name.pop(model_id)
                    break
            else:
                break
    return loaded_model_names, loaded_models



class ModelDetectorTemplate:
    def __init__(self):
        pass

    def match(self, file_path="", state_dict={}):
        return False
    
    def load(self, file_path="", state_dict={}, device="cuda", torch_dtype=torch.float16, **kwargs):
        return [], []
    


class ModelDetectorFromSingleFile:
    def __init__(self, model_loader_configs=[]):
        self.keys_hash_with_shape_dict = {}
        self.keys_hash_dict = {}
        for metadata in model_loader_configs:
            self.add_model_metadata(*metadata)


    def add_model_metadata(self, keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource):
        self.keys_hash_with_shape_dict[keys_hash_with_shape] = (model_names, model_classes, model_resource)
        if keys_hash is not None:
            self.keys_hash_dict[keys_hash] = (model_names, model_classes, model_resource)


    def match(self, file_path="", state_dict={}):
        if isinstance(file_path, str) and os.path.isdir(file_path):
            return False
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            return True
        keys_hash = hash_state_dict_keys(state_dict, with_shape=False)
        if keys_hash in self.keys_hash_dict:
            return True
        return False


    def load(self, file_path="", state_dict={}, device="cuda", torch_dtype=torch.float16, **kwargs):
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)

        # Load models with strict matching
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            model_names, model_classes, model_resource = self.keys_hash_with_shape_dict[keys_hash_with_shape]
            loaded_model_names, loaded_models = load_model_from_single_file(state_dict, model_names, model_classes, model_resource, torch_dtype, device)
            return loaded_model_names, loaded_models

        # Load models without strict matching
        # (the shape of parameters may be inconsistent, and the state_dict_converter will modify the model architecture)
        keys_hash = hash_state_dict_keys(state_dict, with_shape=False)
        if keys_hash in self.keys_hash_dict:
            model_names, model_classes, model_resource = self.keys_hash_dict[keys_hash]
            loaded_model_names, loaded_models = load_model_from_single_file(state_dict, model_names, model_classes, model_resource, torch_dtype, device)
            return loaded_model_names, loaded_models

        return loaded_model_names, loaded_models



class ModelDetectorFromSplitedSingleFile(ModelDetectorFromSingleFile):
    def __init__(self, model_loader_configs=[]):
        super().__init__(model_loader_configs)


    def match(self, file_path="", state_dict={}):
        if isinstance(file_path, str) and os.path.isdir(file_path):
            return False
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)
        splited_state_dict = split_state_dict_with_prefix(state_dict)
        for sub_state_dict in splited_state_dict:
            if super().match(file_path, sub_state_dict):
                return True
        return False


    def load(self, file_path="", state_dict={}, device="cuda", torch_dtype=torch.float16, **kwargs):
        # Split the state_dict and load from each component
        splited_state_dict = split_state_dict_with_prefix(state_dict)
        valid_state_dict = {}
        for sub_state_dict in splited_state_dict:
            if super().match(file_path, sub_state_dict):
                valid_state_dict.update(sub_state_dict)
        if super().match(file_path, valid_state_dict):
            loaded_model_names, loaded_models = super().load(file_path, valid_state_dict, device, torch_dtype)
        else:
            loaded_model_names, loaded_models = [], []
            for sub_state_dict in splited_state_dict:
                if super().match(file_path, sub_state_dict):
                    loaded_model_names_, loaded_models_ = super().load(file_path, valid_state_dict, device, torch_dtype)
                    loaded_model_names += loaded_model_names_
                    loaded_models += loaded_models_
        return loaded_model_names, loaded_models
    


class ModelDetectorFromHuggingfaceFolder:
    def __init__(self, model_loader_configs=[]):
        self.architecture_dict = {}
        for metadata in model_loader_configs:
            self.add_model_metadata(*metadata)


    def add_model_metadata(self, architecture, huggingface_lib, model_name, redirected_architecture):
        self.architecture_dict[architecture] = (huggingface_lib, model_name, redirected_architecture)


    def match(self, file_path="", state_dict={}):
        if not isinstance(file_path, str) or os.path.isfile(file_path):
            return False
        file_list = os.listdir(file_path)
        if "config.json" not in file_list:
            return False
        with open(os.path.join(file_path, "config.json"), "r") as f:
            config = json.load(f)
        if "architectures" not in config and "_class_name" not in config:
            return False
        return True


    def load(self, file_path="", state_dict={}, device="cuda", torch_dtype=torch.float16, **kwargs):
        with open(os.path.join(file_path, "config.json"), "r") as f:
            config = json.load(f)
        loaded_model_names, loaded_models = [], []
        architectures = config["architectures"] if "architectures" in config else [config["_class_name"]]
        for architecture in architectures:
            huggingface_lib, model_name, redirected_architecture = self.architecture_dict[architecture]
            if redirected_architecture is not None:
                architecture = redirected_architecture
            model_class = importlib.import_module(huggingface_lib).__getattribute__(architecture)
            loaded_model_names_, loaded_models_ = load_model_from_huggingface_folder(file_path, [model_name], [model_class], torch_dtype, device)
            loaded_model_names += loaded_model_names_
            loaded_models += loaded_models_
        return loaded_model_names, loaded_models
    


class ModelDetectorFromPatchedSingleFile:
    def __init__(self, model_loader_configs=[]):
        self.keys_hash_with_shape_dict = {}
        for metadata in model_loader_configs:
            self.add_model_metadata(*metadata)


    def add_model_metadata(self, keys_hash_with_shape, model_name, model_class, extra_kwargs):
        self.keys_hash_with_shape_dict[keys_hash_with_shape] = (model_name, model_class, extra_kwargs)


    def match(self, file_path="", state_dict={}):
        if not isinstance(file_path, str) or os.path.isdir(file_path):
            return False
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            return True
        return False


    def load(self, file_path="", state_dict={}, device="cuda", torch_dtype=torch.float16, model_manager=None, **kwargs):
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)

        # Load models with strict matching
        loaded_model_names, loaded_models = [], []
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            model_names, model_classes, extra_kwargs = self.keys_hash_with_shape_dict[keys_hash_with_shape]
            loaded_model_names_, loaded_models_ = load_patch_model_from_single_file(
                state_dict, model_names, model_classes, extra_kwargs, model_manager, torch_dtype, device)
            loaded_model_names += loaded_model_names_
            loaded_models += loaded_models_
        return loaded_model_names, loaded_models



class ModelManager:
    def __init__(
        self,
        torch_dtype=torch.float16,
        device="cuda",
        model_id_list: List[Preset_model_id] = [],
        downloading_priority: List[Preset_model_website] = ["ModelScope", "HuggingFace"],
        file_path_list: List[str] = [],
    ):
        self.torch_dtype = torch_dtype
        self.device = device
        self.model = []
        self.model_path = []
        self.model_name = []
        downloaded_files = download_models(model_id_list, downloading_priority) if len(model_id_list) > 0 else []
        self.model_detector = [
            ModelDetectorFromSingleFile(model_loader_configs),
            ModelDetectorFromSplitedSingleFile(model_loader_configs),
            ModelDetectorFromHuggingfaceFolder(huggingface_model_loader_configs),
            ModelDetectorFromPatchedSingleFile(patch_model_loader_configs),
        ]
        self.load_models(downloaded_files + file_path_list)


    def load_model_from_single_file(self, file_path="", state_dict={}, model_names=[], model_classes=[], model_resource=None):
        print(f"Loading models from file: {file_path}")
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)
        model_names, models = load_model_from_single_file(state_dict, model_names, model_classes, model_resource, self.torch_dtype, self.device)
        for model_name, model in zip(model_names, models):
            self.model.append(model)
            self.model_path.append(file_path)
            self.model_name.append(model_name)
        print(f"    The following models are loaded: {model_names}.")


    def load_model_from_huggingface_folder(self, file_path="", model_names=[], model_classes=[]):
        print(f"Loading models from folder: {file_path}")
        model_names, models = load_model_from_huggingface_folder(file_path, model_names, model_classes, self.torch_dtype, self.device)
        for model_name, model in zip(model_names, models):
            self.model.append(model)
            self.model_path.append(file_path)
            self.model_name.append(model_name)
        print(f"    The following models are loaded: {model_names}.")


    def load_patch_model_from_single_file(self, file_path="", state_dict={}, model_names=[], model_classes=[], extra_kwargs={}):
        print(f"Loading patch models from file: {file_path}")
        model_names, models = load_patch_model_from_single_file(
            state_dict, model_names, model_classes, extra_kwargs, self, self.torch_dtype, self.device)
        for model_name, model in zip(model_names, models):
            self.model.append(model)
            self.model_path.append(file_path)
            self.model_name.append(model_name)
        print(f"    The following patched models are loaded: {model_names}.")


    def load_lora(self, file_path="", state_dict={}, lora_alpha=1.0):
        if isinstance(file_path, list):
            for file_path_ in file_path:
                self.load_lora(file_path_, state_dict=state_dict, lora_alpha=lora_alpha)
        else:
            print(f"Loading LoRA models from file: {file_path}")
            is_loaded = False
            if len(state_dict) == 0:
                state_dict = load_state_dict(file_path)
            for model_name, model, model_path in zip(self.model_name, self.model, self.model_path):
                for lora in get_lora_loaders():
                    match_results = lora.match(model, state_dict)
                    if match_results is not None:
                        print(f"    Adding LoRA to {model_name} ({model_path}).")
                        lora_prefix, model_resource = match_results
                        lora.load(model, state_dict, lora_prefix, alpha=lora_alpha, model_resource=model_resource)
                        is_loaded = True
                        break
            if not is_loaded:
                print(f"    Cannot load LoRA: {file_path}")


    def load_model(self, file_path, model_names=None, device=None, torch_dtype=None):
        print(f"Loading models from: {file_path}")
        if device is None: device = self.device
        if torch_dtype is None: torch_dtype = self.torch_dtype
        if isinstance(file_path, list):
            state_dict = {}
            for path in file_path:
                state_dict.update(load_state_dict(path))
        elif os.path.isfile(file_path):
            state_dict = load_state_dict(file_path)
        else:
            state_dict = None
        for model_detector in self.model_detector:
            if model_detector.match(file_path, state_dict):
                model_names, models = model_detector.load(
                    file_path, state_dict,
                    device=device, torch_dtype=torch_dtype,
                    allowed_model_names=model_names, model_manager=self
                )
                for model_name, model in zip(model_names, models):
                    self.model.append(model)
                    self.model_path.append(file_path)
                    self.model_name.append(model_name)
                print(f"    The following models are loaded: {model_names}.")
                break
        else:
            print(f"    We cannot detect the model type. No models are loaded.")
        

    def load_models(self, file_path_list, model_names=None, device=None, torch_dtype=None):
        for file_path in file_path_list:
            self.load_model(file_path, model_names, device=device, torch_dtype=torch_dtype)

    
    def fetch_model(self, model_name, file_path=None, require_model_path=False):
        fetched_models = []
        fetched_model_paths = []
        for model, model_path, model_name_ in zip(self.model, self.model_path, self.model_name):
            if file_path is not None and file_path != model_path:
                continue
            if model_name == model_name_:
                fetched_models.append(model)
                fetched_model_paths.append(model_path)
        if len(fetched_models) == 0:
            print(f"No {model_name} models available.")
            return None
        if len(fetched_models) == 1:
            print(f"Using {model_name} from {fetched_model_paths[0]}.")
        else:
            print(f"More than one {model_name} models are loaded in model manager: {fetched_model_paths}. Using {model_name} from {fetched_model_paths[0]}.")
        if require_model_path:
            return fetched_models[0], fetched_model_paths[0]
        else:
            return fetched_models[0]
        

    def to(self, device):
        for model in self.model:
            model.to(device)

