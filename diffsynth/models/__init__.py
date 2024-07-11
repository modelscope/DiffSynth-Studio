import torch, os, json
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

from .sd3_text_encoder import SD3TextEncoder1, SD3TextEncoder2, SD3TextEncoder3
from .sd3_dit import SD3DiT
from .sd3_vae_decoder import SD3VAEDecoder
from .sd3_vae_encoder import SD3VAEEncoder

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
from .kolors_text_encoder import ChatGLMModel


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
    # Hunyuan DiT
    "HunyuanDiT": [
        ("modelscope/HunyuanDiT", "t2i/clip_text_encoder/pytorch_model.bin", "models/HunyuanDiT/t2i/clip_text_encoder"),
        ("modelscope/HunyuanDiT", "t2i/mt5/pytorch_model.bin", "models/HunyuanDiT/t2i/mt5"),
        ("modelscope/HunyuanDiT", "t2i/model/pytorch_model_ema.pt", "models/HunyuanDiT/t2i/model"),
        ("modelscope/HunyuanDiT", "t2i/sdxl-vae-fp16-fix/diffusion_pytorch_model.bin", "models/HunyuanDiT/t2i/sdxl-vae-fp16-fix"),
    ],
    # Stable Video Diffusion
    "stable-video-diffusion-img2vid-xt": [
        ("AI-ModelScope/stable-video-diffusion-img2vid-xt", "svd_xt.safetensors", "models/stable_video_diffusion"),
    ],
    # ExVideo
    "ExVideo-SVD-128f-v1": [
        ("ECNU-CILab/ExVideo-SVD-128f-v1", "model.fp16.safetensors", "models/stable_video_diffusion"),
    ],
    # Stable Diffusion
    "StableDiffusion_v15": [
        ("AI-ModelScope/stable-diffusion-v1-5", "v1-5-pruned-emaonly.safetensors", "models/stable_diffusion"),
    ],
    "DreamShaper_8": [
        ("sd_lora/dreamshaper_8", "dreamshaper_8.safetensors", "models/stable_diffusion"),
    ],
    "AingDiffusion_v12": [
        ("sd_lora/aingdiffusion_v12", "aingdiffusion_v12.safetensors", "models/stable_diffusion"),
    ],
    "Flat2DAnimerge_v45Sharp": [
        ("sd_lora/Flat-2D-Animerge", "flat2DAnimerge_v45Sharp.safetensors", "models/stable_diffusion"),
    ],
    # Textual Inversion
    "TextualInversion_VeryBadImageNegative_v1.3": [
        ("sd_lora/verybadimagenegative_v1.3", "verybadimagenegative_v1.3.pt", "models/textual_inversion"),
    ],
    # Stable Diffusion XL
    "StableDiffusionXL_v1": [
        ("AI-ModelScope/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors", "models/stable_diffusion_xl"),
    ],
    "BluePencilXL_v200": [
        ("sd_lora/bluePencilXL_v200", "bluePencilXL_v200.safetensors", "models/stable_diffusion_xl"),
    ],
    "StableDiffusionXL_Turbo": [
        ("AI-ModelScope/sdxl-turbo", "sd_xl_turbo_1.0_fp16.safetensors", "models/stable_diffusion_xl_turbo"),
    ],
    # Stable Diffusion 3
    "StableDiffusion3": [
        ("AI-ModelScope/stable-diffusion-3-medium", "sd3_medium_incl_clips_t5xxlfp16.safetensors", "models/stable_diffusion_3"),
    ],
    "StableDiffusion3_without_T5": [
        ("AI-ModelScope/stable-diffusion-3-medium", "sd3_medium_incl_clips.safetensors", "models/stable_diffusion_3"),
    ],
    # ControlNet
    "ControlNet_v11f1p_sd15_depth": [
        ("AI-ModelScope/ControlNet-v1-1", "control_v11f1p_sd15_depth.pth", "models/ControlNet"),
        ("sd_lora/Annotators", "dpt_hybrid-midas-501f0c75.pt", "models/Annotators")
    ],
    "ControlNet_v11p_sd15_softedge": [
        ("AI-ModelScope/ControlNet-v1-1", "control_v11p_sd15_softedge.pth", "models/ControlNet"),
        ("sd_lora/Annotators", "ControlNetHED.pth", "models/Annotators")
    ],
    "ControlNet_v11f1e_sd15_tile": [
        ("AI-ModelScope/ControlNet-v1-1", "control_v11f1e_sd15_tile.pth", "models/ControlNet")
    ],
    "ControlNet_v11p_sd15_lineart": [
        ("AI-ModelScope/ControlNet-v1-1", "control_v11p_sd15_lineart.pth", "models/ControlNet"),
        ("sd_lora/Annotators", "sk_model.pth", "models/Annotators"),
        ("sd_lora/Annotators", "sk_model2.pth", "models/Annotators")
    ],
    # AnimateDiff
    "AnimateDiff_v2": [
        ("Shanghai_AI_Laboratory/animatediff", "mm_sd_v15_v2.ckpt", "models/AnimateDiff"),
    ],
    "AnimateDiff_xl_beta": [
        ("Shanghai_AI_Laboratory/animatediff", "mm_sdxl_v10_beta.ckpt", "models/AnimateDiff"),
    ],
    # RIFE
    "RIFE": [
        ("Damo_XR_Lab/cv_rife_video-frame-interpolation", "flownet.pkl", "models/RIFE"),
    ],
    # Beautiful Prompt
    "BeautifulPrompt": [
        ("AI-ModelScope/pai-bloom-1b1-text2prompt-sd", "config.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
        ("AI-ModelScope/pai-bloom-1b1-text2prompt-sd", "generation_config.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
        ("AI-ModelScope/pai-bloom-1b1-text2prompt-sd", "model.safetensors", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
        ("AI-ModelScope/pai-bloom-1b1-text2prompt-sd", "special_tokens_map.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
        ("AI-ModelScope/pai-bloom-1b1-text2prompt-sd", "tokenizer.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
        ("AI-ModelScope/pai-bloom-1b1-text2prompt-sd", "tokenizer_config.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
    ],
    # Translator
    "opus-mt-zh-en": [
        ("moxying/opus-mt-zh-en", "config.json", "models/translator/opus-mt-zh-en"),
        ("moxying/opus-mt-zh-en", "generation_config.json", "models/translator/opus-mt-zh-en"),
        ("moxying/opus-mt-zh-en", "metadata.json", "models/translator/opus-mt-zh-en"),
        ("moxying/opus-mt-zh-en", "pytorch_model.bin", "models/translator/opus-mt-zh-en"),
        ("moxying/opus-mt-zh-en", "source.spm", "models/translator/opus-mt-zh-en"),
        ("moxying/opus-mt-zh-en", "target.spm", "models/translator/opus-mt-zh-en"),
        ("moxying/opus-mt-zh-en", "tokenizer_config.json", "models/translator/opus-mt-zh-en"),
        ("moxying/opus-mt-zh-en", "vocab.json", "models/translator/opus-mt-zh-en"),
    ],
    # IP-Adapter
    "IP-Adapter-SD": [
        ("AI-ModelScope/IP-Adapter", "models/image_encoder/model.safetensors", "models/IpAdapter/stable_diffusion/image_encoder"),
        ("AI-ModelScope/IP-Adapter", "models/ip-adapter_sd15.bin", "models/IpAdapter/stable_diffusion"),
    ],
    "IP-Adapter-SDXL": [
        ("AI-ModelScope/IP-Adapter", "sdxl_models/image_encoder/model.safetensors", "models/IpAdapter/stable_diffusion_xl/image_encoder"),
        ("AI-ModelScope/IP-Adapter", "sdxl_models/ip-adapter_sdxl.bin", "models/IpAdapter/stable_diffusion_xl"),
    ],
    # Kolors
    "Kolors": [
        ("Kwai-Kolors/Kolors", "text_encoder/config.json", "models/kolors/Kolors/text_encoder"),
        ("Kwai-Kolors/Kolors", "text_encoder/pytorch_model.bin.index.json", "models/kolors/Kolors/text_encoder"),
        ("Kwai-Kolors/Kolors", "text_encoder/pytorch_model-00001-of-00007.bin", "models/kolors/Kolors/text_encoder"),
        ("Kwai-Kolors/Kolors", "text_encoder/pytorch_model-00002-of-00007.bin", "models/kolors/Kolors/text_encoder"),
        ("Kwai-Kolors/Kolors", "text_encoder/pytorch_model-00003-of-00007.bin", "models/kolors/Kolors/text_encoder"),
        ("Kwai-Kolors/Kolors", "text_encoder/pytorch_model-00004-of-00007.bin", "models/kolors/Kolors/text_encoder"),
        ("Kwai-Kolors/Kolors", "text_encoder/pytorch_model-00005-of-00007.bin", "models/kolors/Kolors/text_encoder"),
        ("Kwai-Kolors/Kolors", "text_encoder/pytorch_model-00006-of-00007.bin", "models/kolors/Kolors/text_encoder"),
        ("Kwai-Kolors/Kolors", "text_encoder/pytorch_model-00007-of-00007.bin", "models/kolors/Kolors/text_encoder"),
        ("Kwai-Kolors/Kolors", "unet/diffusion_pytorch_model.safetensors", "models/kolors/Kolors/unet"),
        ("Kwai-Kolors/Kolors", "vae/diffusion_pytorch_model.safetensors", "models/kolors/Kolors/vae"),
    ],
}
Preset_model_id: TypeAlias = Literal[
    "HunyuanDiT",
    "stable-video-diffusion-img2vid-xt",
    "ExVideo-SVD-128f-v1",
    "StableDiffusion_v15",
    "DreamShaper_8",
    "AingDiffusion_v12",
    "Flat2DAnimerge_v45Sharp",
    "TextualInversion_VeryBadImageNegative_v1.3",
    "StableDiffusionXL_v1",
    "BluePencilXL_v200",
    "StableDiffusionXL_Turbo",
    "ControlNet_v11f1p_sd15_depth",
    "ControlNet_v11p_sd15_softedge",
    "ControlNet_v11f1e_sd15_tile",
    "ControlNet_v11p_sd15_lineart",
    "AnimateDiff_v2",
    "AnimateDiff_xl_beta",
    "RIFE",
    "BeautifulPrompt",
    "opus-mt-zh-en",
    "IP-Adapter-SD",
    "IP-Adapter-SDXL",
    "StableDiffusion3",
    "StableDiffusion3_without_T5",
    "Kolors",
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


def download_models(
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
        downloaded_files = download_models(model_id_list, downloading_priority)
        self.load_models(downloaded_files + file_path_list)

    def load_model_from_origin(
        self,
        download_from: Preset_model_website = "ModelScope",
        model_id = "",
        origin_file_path = "",
        local_dir = ""
    ):
        website_to_download_fn[download_from](model_id, origin_file_path, local_dir)
        file_to_download = os.path.join(local_dir, os.path.basename(origin_file_path))
        self.load_model(file_to_download)

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
        return param_name in state_dict
    
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
        return param_name in state_dict and len(state_dict) == 258
    
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
        param_name_ = "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
        return param_name in state_dict and param_name_ in state_dict
    
    def is_hunyuan_dit(self, state_dict):
        param_name = "final_layer.adaLN_modulation.1.weight"
        return param_name in state_dict
    
    def is_diffusers_vae(self, state_dict):
        param_name = "quant_conv.weight"
        return param_name in state_dict
    
    def is_ExVideo_StableVideoDiffusion(self, state_dict):
        param_name = "blocks.185.positional_embedding.embeddings"
        return param_name in state_dict
    
    def is_stable_diffusion_3(self, state_dict):
        param_names = [
            "text_encoders.clip_l.transformer.text_model.encoder.layers.9.self_attn.v_proj.weight",
            "text_encoders.clip_g.transformer.text_model.encoder.layers.9.self_attn.v_proj.weight",
            "model.diffusion_model.joint_blocks.9.x_block.mlp.fc2.weight",
            "first_stage_model.encoder.mid.block_2.norm2.weight",
            "first_stage_model.decoder.mid.block_2.norm2.weight",
        ]
        for param_name in param_names:
            if param_name not in state_dict:
                return False
        return True
    
    def is_stable_diffusion_3_t5(self, state_dict):
        param_name = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
        return param_name in state_dict
    
    def is_kolors_text_encoder(self, file_path):
        file_list = os.listdir(file_path)
        if "config.json" in file_list:
            try:
                with open(os.path.join(file_path, "config.json"), "r") as f:
                    config = json.load(f)
                    if config.get("model_type") == "chatglm":
                        return True
            except:
                pass
        return False
    
    def is_kolors_unet(self, state_dict):
        return "up_blocks.2.resnets.2.time_emb_proj.weight" in state_dict and "encoder_hid_proj.weight" in state_dict
    
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
        model.to(torch.float32).to(self.device)
        self.model[component] = model
        self.model_path[component] = file_path
        component = "vae_decoder"
        model = SDXLVAEDecoder()
        model.load_state_dict(model.state_dict_converter().from_diffusers(state_dict))
        model.to(torch.float32).to(self.device)
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

    def load_stable_diffusion_3(self, state_dict, components=None, file_path=""):
        component_dict = {
            "sd3_text_encoder_1": SD3TextEncoder1,
            "sd3_text_encoder_2": SD3TextEncoder2,
            "sd3_text_encoder_3": SD3TextEncoder3,
            "sd3_dit": SD3DiT,
            "sd3_vae_decoder": SD3VAEDecoder,
            "sd3_vae_encoder": SD3VAEEncoder,
        }
        if components is None:
            components = ["sd3_text_encoder_1", "sd3_text_encoder_2", "sd3_text_encoder_3", "sd3_dit", "sd3_vae_decoder", "sd3_vae_encoder"]
        for component in components:
            if component == "sd3_text_encoder_3":
                if "text_encoders.t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight" not in state_dict:
                    continue
            if component == "sd3_text_encoder_1":
                # Add additional token embeddings to text encoder
                token_embeddings = [state_dict["text_encoders.clip_l.transformer.text_model.embeddings.token_embedding.weight"]]
                for keyword in self.textual_inversion_dict:
                    _, embeddings = self.textual_inversion_dict[keyword]
                    token_embeddings.append(embeddings.to(dtype=token_embeddings[0].dtype))
                token_embeddings = torch.concat(token_embeddings, dim=0)
                state_dict["text_encoders.clip_l.transformer.text_model.embeddings.token_embedding.weight"] = token_embeddings
                self.model[component] = component_dict[component](vocab_size=token_embeddings.shape[0])
                self.model[component].load_state_dict(self.model[component].state_dict_converter().from_civitai(state_dict))
                self.model[component].to(self.torch_dtype).to(self.device)
            else:
                self.model[component] = component_dict[component]()
                self.model[component].load_state_dict(self.model[component].state_dict_converter().from_civitai(state_dict))
                self.model[component].to(self.torch_dtype).to(self.device)
                self.model_path[component] = file_path

    def load_stable_diffusion_3_t5(self, state_dict, file_path=""):
        component = "sd3_text_encoder_3"
        model = SD3TextEncoder3()
        model.load_state_dict(model.state_dict_converter().from_civitai(state_dict))
        model.to(self.torch_dtype).to(self.device)
        self.model[component] = model
        self.model_path[component] = file_path

    def load_kolors_text_encoder(self, state_dict=None, file_path=""):
        component = "kolors_text_encoder"
        model = ChatGLMModel.from_pretrained(file_path, torch_dtype=self.torch_dtype)
        model = model.to(dtype=self.torch_dtype, device=self.device)
        self.model[component] = model
        self.model_path[component] = file_path

    def load_kolors_unet(self, state_dict, file_path=""):
        component = "kolors_unet"
        model = SDXLUNet(is_kolors=True)
        model.load_state_dict(model.state_dict_converter().from_diffusers(state_dict))
        model.to(self.torch_dtype).to(self.device)
        self.model[component] = model
        self.model_path[component] = file_path

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
            if os.path.isdir(os.path.join(folder, file_name)) or \
                not (file_name.endswith(".bin") or \
                     file_name.endswith(".safetensors") or \
                     file_name.endswith(".pth") or \
                     file_name.endswith(".pt")):
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
        if os.path.isdir(file_path):
            if self.is_kolors_text_encoder(file_path):
                self.load_kolors_text_encoder(file_path=file_path)
            return
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
        elif self.is_stable_diffusion_3(state_dict):
            self.load_stable_diffusion_3(state_dict, components=components, file_path=file_path)
        elif self.is_stable_diffusion_3_t5(state_dict):
            self.load_stable_diffusion_3_t5(state_dict, file_path=file_path)
        elif self.is_kolors_unet(state_dict):
            self.load_kolors_unet(state_dict, file_path=file_path)

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
