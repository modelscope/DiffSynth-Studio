import torch, os
from safetensors import safe_open
from typing_extensions import Literal, TypeAlias
from typing import List

from .downloader import download_from_huggingface, download_from_modelscope

from .sd_text_encoder import SDTextEncoder
from .sd_unet import SDUNet
from .sd_vae_encoder import SDVAEEncoder
from .sd_vae_decoder import SDVAEDecoder
from .sd_lora import SDLoRA

from .sdxl_text_encoder import SDXLTextEncoder, SDXLTextEncoder2
from .sdxl_unet import SDXLUNet
from .sdxl_vae_decoder import SDXLVAEDecoder
from .sdxl_vae_encoder import SDXLVAEEncoder

from .sd_controlnet import SDControlNet

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


preset_models_on_huggingface = {
    "HunyuanDiT": [
        ("Tencent-Hunyuan/HunyuanDiT", "t2i/clip_text_encoder/pytorch_model.bin", "models/HunyuanDiT/t2i/clip_text_encoder"),
        ("Tencent-Hunyuan/HunyuanDiT", "t2i/mt5/pytorch_model.bin", "models/HunyuanDiT/t2i/mt5"),
        ("Tencent-Hunyuan/HunyuanDiT", "t2i/model/pytorch_model_ema.pt", "models/HunyuanDiT/t2i/model"),
        ("Tencent-Hunyuan/HunyuanDiT", "t2i/sdxl-vae-fp16-fix/diffusion_pytorch_model.bin", "models/HunyuanDiT/t2i/sdxl-vae-fp16-fix"),
    ],
    "stable-video-diffusion-img2vid-xt": [
        ("stabilityai/stable-video-diffusion-img2vid-xt", "svd_xt.safetensors", "models/stable_video_diffusion"),
    ],
    "ExVideo-SVD-128f-v1": [
        ("ECNU-CILab/ExVideo-SVD-128f-v1", "model.fp16.safetensors", "models/stable_video_diffusion"),
    ],
}
preset_models_on_modelscope = {
    "HunyuanDiT": [
        ("modelscope/HunyuanDiT", "t2i/clip_text_encoder/pytorch_model.bin", "models/HunyuanDiT/t2i/clip_text_encoder"),
        ("modelscope/HunyuanDiT", "t2i/mt5/pytorch_model.bin", "models/HunyuanDiT/t2i/mt5"),
        ("modelscope/HunyuanDiT", "t2i/model/pytorch_model_ema.pt", "models/HunyuanDiT/t2i/model"),
        ("modelscope/HunyuanDiT", "t2i/sdxl-vae-fp16-fix/diffusion_pytorch_model.bin", "models/HunyuanDiT/t2i/sdxl-vae-fp16-fix"),
    ],
    "stable-video-diffusion-img2vid-xt": [
        ("AI-ModelScope/stable-video-diffusion-img2vid-xt", "svd_xt.safetensors", "models/stable_video_diffusion"),
    ],
    "ExVideo-SVD-128f-v1": [
        ("ECNU-CILab/ExVideo-SVD-128f-v1", "model.fp16.safetensors", "models/stable_video_diffusion"),
    ],
}
Preset_model_id: TypeAlias = Literal[
    "HunyuanDiT",
    "stable-video-diffusion-img2vid-xt",
    "ExVideo-SVD-128f-v1"
]
Preset_model_website: TypeAlias = Literal[
    "HuggingFace",
    "ModelScope",
]
website_to_preset_models = {
    "HuggingFace": preset_models_on_huggingface,
    "ModelScope": preset_models_on_modelscope,
}
website_to_download_fn = {
    "HuggingFace": download_from_huggingface,
    "ModelScope": download_from_modelscope,
}


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
        self.model = {}
        self.model_path = {}
        self.textual_inversion_dict = {}
        downloaded_files = self.download_models(model_id_list, downloading_priority)
        self.load_models(downloaded_files + file_path_list)

    def download_models(
        self,
        model_id_list: List[Preset_model_id] = [],
        downloading_priority: List[Preset_model_website] = ["ModelScope", "HuggingFace"],
    ):
        downloaded_files = []
        for model_id in model_id_list:
            for website in downloading_priority:
                if model_id in website_to_preset_models[website]:
                    for model_id, origin_file_path, local_dir in website_to_preset_models[website][model_id]:
                        # Check if the file is downloaded.
                        file_to_download = os.path.join(local_dir, os.path.basename(origin_file_path))
                        if file_to_download in downloaded_files:
                            continue
                        # Download
                        website_to_download_fn[website](model_id, origin_file_path, local_dir)
                        if os.path.basename(origin_file_path) in os.listdir(local_dir):
                            downloaded_files.append(file_to_download)
        return downloaded_files

    def is_stable_video_diffusion(self, state_dict):
        param_name = "model.diffusion_model.output_blocks.9.1.time_stack.0.norm_in.weight"
        return param_name in state_dict

    def is_RIFE(self, state_dict):
        param_name = "block_tea.convblock3.0.1.weight"
        return param_name in state_dict or ("module." + param_name) in state_dict

    def is_beautiful_prompt(self, state_dict):
        param_name = "transformer.h.9.self_attention.query_key_value.weight"
        return param_name in state_dict

    def is_stabe_diffusion_xl(self, state_dict):
        param_name = "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight"
        return param_name in state_dict

    def is_stable_diffusion(self, state_dict):
        if self.is_stabe_diffusion_xl(state_dict):
            return False
        param_name = "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm3.weight"
        return param_name in state_dict
    
    def is_controlnet(self, state_dict):
        param_name = "control_model.time_embed.0.weight"
        param_name_2 = "mid_block.resnets.1.time_emb_proj.weight" # For controlnets in diffusers format
        return param_name in state_dict or param_name_2 in state_dict
    
    def is_animatediff(self, state_dict):
        param_name = "mid_block.motion_modules.0.temporal_transformer.proj_out.weight"
        return param_name in state_dict
    
    def is_animatediff_xl(self, state_dict):
        param_name = "up_blocks.2.motion_modules.2.temporal_transformer.transformer_blocks.0.ff_norm.weight"
        return param_name in state_dict
    
    def is_sd_lora(self, state_dict):
        param_name = "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_ff_net_2.lora_up.weight"
        return param_name in state_dict
    
    def is_translator(self, state_dict):
        param_name = "model.encoder.layers.5.self_attn_layer_norm.weight"
        return param_name in state_dict and len(state_dict) == 254
    
    def is_ipadapter(self, state_dict):
        return "image_proj" in state_dict and "ip_adapter" in state_dict and state_dict["image_proj"]["proj.weight"].shape == torch.Size([3072, 1024])
    
    def is_ipadapter_image_encoder(self, state_dict):
        param_name = "vision_model.encoder.layers.31.self_attn.v_proj.weight"
        return param_name in state_dict and len(state_dict) == 521
    
    def is_ipadapter_xl(self, state_dict):
        return "image_proj" in state_dict and "ip_adapter" in state_dict and state_dict["image_proj"]["proj.weight"].shape == torch.Size([8192, 1280])
    
    def is_ipadapter_xl_image_encoder(self, state_dict):
        param_name = "vision_model.encoder.layers.47.self_attn.v_proj.weight"
        return param_name in state_dict and len(state_dict) == 777
    
    def is_hunyuan_dit_clip_text_encoder(self, state_dict):
        param_name = "bert.encoder.layer.23.attention.output.dense.weight"
        return param_name in state_dict
    
    def is_hunyuan_dit_t5_text_encoder(self, state_dict):
        param_name = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
        return param_name in state_dict
    
    def is_hunyuan_dit(self, state_dict):
        param_name = "final_layer.adaLN_modulation.1.weight"
        return param_name in state_dict
    
    def is_diffusers_vae(self, state_dict):
        param_name = "quant_conv.weight"
        return param_name in state_dict
    
    def is_ExVideo_StableVideoDiffusion(self, state_dict):
        param_name = "blocks.185.positional_embedding.embeddings"
        return param_name in state_dict
    
    def load_stable_video_diffusion(self, state_dict, components=None, file_path="", add_positional_conv=None):
        component_dict = {
            "image_encoder": SVDImageEncoder,
            "unet": SVDUNet,
            "vae_decoder": SVDVAEDecoder,
            "vae_encoder": SVDVAEEncoder,
        }
        if components is None:
            components = ["image_encoder", "unet", "vae_decoder", "vae_encoder"]
        for component in components:
            if component == "unet":
                self.model[component] = component_dict[component](add_positional_conv=add_positional_conv)
                self.model[component].load_state_dict(self.model[component].state_dict_converter().from_civitai(state_dict, add_positional_conv=add_positional_conv), strict=False)
            else:
                self.model[component] = component_dict[component]()
                self.model[component].load_state_dict(self.model[component].state_dict_converter().from_civitai(state_dict))
            self.model[component].to(self.torch_dtype).to(self.device)
            self.model_path[component] = file_path
    
    def load_stable_diffusion(self, state_dict, components=None, file_path=""):
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
            if component == "text_encoder":
                # Add additional token embeddings to text encoder
                token_embeddings = [state_dict["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"]]
                for keyword in self.textual_inversion_dict:
                    _, embeddings = self.textual_inversion_dict[keyword]
                    token_embeddings.append(embeddings.to(dtype=token_embeddings[0].dtype))
                token_embeddings = torch.concat(token_embeddings, dim=0)
                state_dict["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"] = token_embeddings
                self.model[component] = component_dict[component](vocab_size=token_embeddings.shape[0])
                self.model[component].load_state_dict(self.model[component].state_dict_converter().from_civitai(state_dict))
                self.model[component].to(self.torch_dtype).to(self.device)
            else:
                self.model[component] = component_dict[component]()
                self.model[component].load_state_dict(self.model[component].state_dict_converter().from_civitai(state_dict))
                self.model[component].to(self.torch_dtype).to(self.device)
            self.model_path[component] = file_path

    def load_stable_diffusion_xl(self, state_dict, components=None, file_path=""):
        component_dict = {
            "text_encoder": SDXLTextEncoder,
            "text_encoder_2": SDXLTextEncoder2,
            "unet": SDXLUNet,
            "vae_decoder": SDXLVAEDecoder,
            "vae_encoder": SDXLVAEEncoder,
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
                self.model[component].to(self.torch_dtype).to(self.device)
            self.model_path[component] = file_path

    def load_controlnet(self, state_dict, file_path=""):
        component = "controlnet"
        if component not in self.model:
            self.model[component] = []
            self.model_path[component] = []
        model = SDControlNet()
        model.load_state_dict(model.state_dict_converter().from_civitai(state_dict))
        model.to(self.torch_dtype).to(self.device)
        self.model[component].append(model)
        self.model_path[component].append(file_path)

    def load_animatediff(self, state_dict, file_path=""):
        component = "motion_modules"
        model = SDMotionModel()
        model.load_state_dict(model.state_dict_converter().from_civitai(state_dict))
        model.to(self.torch_dtype).to(self.device)
        self.model[component] = model
        self.model_path[component] = file_path

    def load_animatediff_xl(self, state_dict, file_path=""):
        component = "motion_modules_xl"
        model = SDXLMotionModel()
        model.load_state_dict(model.state_dict_converter().from_civitai(state_dict))
        model.to(self.torch_dtype).to(self.device)
        self.model[component] = model
        self.model_path[component] = file_path

    def load_beautiful_prompt(self, state_dict, file_path=""):
        component = "beautiful_prompt"
        from transformers import AutoModelForCausalLM
        model_folder = os.path.dirname(file_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_folder, state_dict=state_dict, local_files_only=True, torch_dtype=self.torch_dtype
        ).to(self.device).eval()
        self.model[component] = model
        self.model_path[component] = file_path

    def load_RIFE(self, state_dict, file_path=""):
        component = "RIFE"
        from ..extensions.RIFE import IFNet
        model = IFNet().eval()
        model.load_state_dict(model.state_dict_converter().from_civitai(state_dict))
        model.to(torch.float32).to(self.device)
        self.model[component] = model
        self.model_path[component] = file_path

    def load_sd_lora(self, state_dict, alpha):
        SDLoRA().add_lora_to_text_encoder(self.model["text_encoder"], state_dict, alpha=alpha, device=self.device)
        SDLoRA().add_lora_to_unet(self.model["unet"], state_dict, alpha=alpha, device=self.device)

    def load_translator(self, state_dict, file_path=""):
        # This model is lightweight, we do not place it on GPU.
        component = "translator"
        from transformers import AutoModelForSeq2SeqLM
        model_folder = os.path.dirname(file_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_folder).eval()
        self.model[component] = model
        self.model_path[component] = file_path

    def load_ipadapter(self, state_dict, file_path=""):
        component = "ipadapter"
        model = SDIpAdapter()
        model.load_state_dict(model.state_dict_converter().from_civitai(state_dict))
        model.to(self.torch_dtype).to(self.device)
        self.model[component] = model
        self.model_path[component] = file_path

    def load_ipadapter_image_encoder(self, state_dict, file_path=""):
        component = "ipadapter_image_encoder"
        model = IpAdapterCLIPImageEmbedder()
        model.load_state_dict(model.state_dict_converter().from_diffusers(state_dict))
        model.to(self.torch_dtype).to(self.device)
        self.model[component] = model
        self.model_path[component] = file_path

    def load_ipadapter_xl(self, state_dict, file_path=""):
        component = "ipadapter_xl"
        model = SDXLIpAdapter()
        model.load_state_dict(model.state_dict_converter().from_civitai(state_dict))
        model.to(self.torch_dtype).to(self.device)
        self.model[component] = model
        self.model_path[component] = file_path

    def load_ipadapter_xl_image_encoder(self, state_dict, file_path=""):
        component = "ipadapter_xl_image_encoder"
        model = IpAdapterXLCLIPImageEmbedder()
        model.load_state_dict(model.state_dict_converter().from_diffusers(state_dict))
        model.to(self.torch_dtype).to(self.device)
        self.model[component] = model
        self.model_path[component] = file_path

    def load_hunyuan_dit_clip_text_encoder(self, state_dict, file_path=""):
        component = "hunyuan_dit_clip_text_encoder"
        model = HunyuanDiTCLIPTextEncoder()
        model.load_state_dict(model.state_dict_converter().from_civitai(state_dict))
        model.to(self.torch_dtype).to(self.device)
        self.model[component] = model
        self.model_path[component] = file_path

    def load_hunyuan_dit_t5_text_encoder(self, state_dict, file_path=""):
        component = "hunyuan_dit_t5_text_encoder"
        model = HunyuanDiTT5TextEncoder()
        model.load_state_dict(model.state_dict_converter().from_civitai(state_dict))
        model.to(self.torch_dtype).to(self.device)
        self.model[component] = model
        self.model_path[component] = file_path

    def load_hunyuan_dit(self, state_dict, file_path=""):
        component = "hunyuan_dit"
        model = HunyuanDiT()
        model.load_state_dict(model.state_dict_converter().from_civitai(state_dict))
        model.to(self.torch_dtype).to(self.device)
        self.model[component] = model
        self.model_path[component] = file_path

    def load_diffusers_vae(self, state_dict, file_path=""):
        # TODO: detect SD and SDXL
        component = "vae_encoder"
        model = SDXLVAEEncoder()
        model.load_state_dict(model.state_dict_converter().from_diffusers(state_dict))
        model.to(self.torch_dtype).to(self.device)
        self.model[component] = model
        self.model_path[component] = file_path
        component = "vae_decoder"
        model = SDXLVAEDecoder()
        model.load_state_dict(model.state_dict_converter().from_diffusers(state_dict))
        model.to(self.torch_dtype).to(self.device)
        self.model[component] = model
        self.model_path[component] = file_path

    def load_ExVideo_StableVideoDiffusion(self, state_dict, file_path=""):
        unet_state_dict = self.model["unet"].state_dict()
        self.model["unet"].to("cpu")
        del self.model["unet"]
        add_positional_conv = state_dict["blocks.185.positional_embedding.embeddings"].shape[0]
        self.model["unet"] = SVDUNet(add_positional_conv=add_positional_conv)
        self.model["unet"].load_state_dict(unet_state_dict, strict=False)
        self.model["unet"].load_state_dict(state_dict, strict=False)
        self.model["unet"].to(self.torch_dtype).to(self.device)

    def search_for_embeddings(self, state_dict):
        embeddings = []
        for k in state_dict:
            if isinstance(state_dict[k], torch.Tensor):
                embeddings.append(state_dict[k])
            elif isinstance(state_dict[k], dict):
                embeddings += self.search_for_embeddings(state_dict[k])
        return embeddings

    def load_textual_inversions(self, folder):
        # Store additional tokens here
        self.textual_inversion_dict = {}

        # Load every textual inversion file
        for file_name in os.listdir(folder):
            if file_name.endswith(".txt"):
                continue
            keyword = os.path.splitext(file_name)[0]
            state_dict = load_state_dict(os.path.join(folder, file_name))

            # Search for embeddings
            for embeddings in self.search_for_embeddings(state_dict):
                if len(embeddings.shape) == 2 and embeddings.shape[1] == 768:
                    tokens = [f"{keyword}_{i}" for i in range(embeddings.shape[0])]
                    self.textual_inversion_dict[keyword] = (tokens, embeddings)
                    break
        
    def load_model(self, file_path, components=None, lora_alphas=[]):
        state_dict = load_state_dict(file_path, torch_dtype=self.torch_dtype)
        if self.is_stable_video_diffusion(state_dict):
            self.load_stable_video_diffusion(state_dict, file_path=file_path)
        elif self.is_animatediff(state_dict):
            self.load_animatediff(state_dict, file_path=file_path)
        elif self.is_animatediff_xl(state_dict):
            self.load_animatediff_xl(state_dict, file_path=file_path)
        elif self.is_controlnet(state_dict):
            self.load_controlnet(state_dict, file_path=file_path)
        elif self.is_stabe_diffusion_xl(state_dict):
            self.load_stable_diffusion_xl(state_dict, components=components, file_path=file_path)
        elif self.is_stable_diffusion(state_dict):
            self.load_stable_diffusion(state_dict, components=components, file_path=file_path)
        elif self.is_sd_lora(state_dict):
            self.load_sd_lora(state_dict, alpha=lora_alphas.pop(0))
        elif self.is_beautiful_prompt(state_dict):
            self.load_beautiful_prompt(state_dict, file_path=file_path)
        elif self.is_RIFE(state_dict):
            self.load_RIFE(state_dict, file_path=file_path)
        elif self.is_translator(state_dict):
            self.load_translator(state_dict, file_path=file_path)
        elif self.is_ipadapter(state_dict):
            self.load_ipadapter(state_dict, file_path=file_path)
        elif self.is_ipadapter_image_encoder(state_dict):
            self.load_ipadapter_image_encoder(state_dict, file_path=file_path)
        elif self.is_ipadapter_xl(state_dict):
            self.load_ipadapter_xl(state_dict, file_path=file_path)
        elif self.is_ipadapter_xl_image_encoder(state_dict):
            self.load_ipadapter_xl_image_encoder(state_dict, file_path=file_path)
        elif self.is_hunyuan_dit_clip_text_encoder(state_dict):
            self.load_hunyuan_dit_clip_text_encoder(state_dict, file_path=file_path)
        elif self.is_hunyuan_dit_t5_text_encoder(state_dict):
            self.load_hunyuan_dit_t5_text_encoder(state_dict, file_path=file_path)
        elif self.is_hunyuan_dit(state_dict):
            self.load_hunyuan_dit(state_dict, file_path=file_path)
        elif self.is_diffusers_vae(state_dict):
            self.load_diffusers_vae(state_dict, file_path=file_path)
        elif self.is_ExVideo_StableVideoDiffusion(state_dict):
            self.load_ExVideo_StableVideoDiffusion(state_dict, file_path=file_path)

    def load_models(self, file_path_list, lora_alphas=[]):
        for file_path in file_path_list:
            self.load_model(file_path, lora_alphas=lora_alphas)
        
    def to(self, device):
        for component in self.model:
            if isinstance(self.model[component], list):
                for model in self.model[component]:
                    model.to(device)
            else:
                self.model[component].to(device)
        torch.cuda.empty_cache()

    def get_model_with_model_path(self, model_path):
        for component in self.model_path:
            if isinstance(self.model_path[component], str):
                if os.path.samefile(self.model_path[component], model_path):
                    return self.model[component]
            elif isinstance(self.model_path[component], list):
                for i, model_path_ in enumerate(self.model_path[component]):
                    if os.path.samefile(model_path_, model_path):
                        return self.model[component][i]
        raise ValueError(f"Please load model {model_path} before you use it.")
    
    def __getattr__(self, __name):
        if __name in self.model:
            return self.model[__name]
        else:
            return super.__getattribute__(__name)


def load_state_dict(file_path, torch_dtype=None):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype)


def load_state_dict_from_safetensors(file_path, torch_dtype=None):
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None):
    state_dict = torch.load(file_path, map_location="cpu")
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict


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
