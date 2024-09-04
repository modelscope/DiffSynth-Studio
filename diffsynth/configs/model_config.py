from typing_extensions import Literal, TypeAlias

from ..models.sd_text_encoder import SDTextEncoder
from ..models.sd_unet import SDUNet
from ..models.sd_vae_encoder import SDVAEEncoder
from ..models.sd_vae_decoder import SDVAEDecoder

from ..models.sdxl_text_encoder import SDXLTextEncoder, SDXLTextEncoder2
from ..models.sdxl_unet import SDXLUNet
from ..models.sdxl_vae_decoder import SDXLVAEDecoder
from ..models.sdxl_vae_encoder import SDXLVAEEncoder

from ..models.sd3_text_encoder import SD3TextEncoder1, SD3TextEncoder2, SD3TextEncoder3
from ..models.sd3_dit import SD3DiT
from ..models.sd3_vae_decoder import SD3VAEDecoder
from ..models.sd3_vae_encoder import SD3VAEEncoder

from ..models.sd_controlnet import SDControlNet
from ..models.sdxl_controlnet import SDXLControlNetUnion

from ..models.sd_motion import SDMotionModel
from ..models.sdxl_motion import SDXLMotionModel

from ..models.svd_image_encoder import SVDImageEncoder
from ..models.svd_unet import SVDUNet
from ..models.svd_vae_decoder import SVDVAEDecoder
from ..models.svd_vae_encoder import SVDVAEEncoder

from ..models.sd_ipadapter import SDIpAdapter, IpAdapterCLIPImageEmbedder
from ..models.sdxl_ipadapter import SDXLIpAdapter, IpAdapterXLCLIPImageEmbedder

from ..models.hunyuan_dit_text_encoder import HunyuanDiTCLIPTextEncoder, HunyuanDiTT5TextEncoder
from ..models.hunyuan_dit import HunyuanDiT


from ..models.flux_dit import FluxDiT
from ..models.flux_text_encoder import FluxTextEncoder1, FluxTextEncoder2
from ..models.flux_vae import FluxVAEEncoder, FluxVAEDecoder



model_loader_configs = [
    # These configs are provided for detecting model type automatically.
    # The format is (state_dict_keys_hash, state_dict_keys_hash_with_shape, model_names, model_classes, model_resource)
    (None, "091b0e30e77c76626b3ba62acdf95343", ["sd_controlnet"], [SDControlNet], "civitai"),
    (None, "4a6c8306a27d916dea81263c8c88f450", ["hunyuan_dit_clip_text_encoder"], [HunyuanDiTCLIPTextEncoder], "civitai"),
    (None, "f4aec400fe394297961218c768004521", ["hunyuan_dit"], [HunyuanDiT], "civitai"),
    (None, "9e6e58043a5a2e332803ed42f6ee7181", ["hunyuan_dit_t5_text_encoder"], [HunyuanDiTT5TextEncoder], "civitai"),
    (None, "13115dd45a6e1c39860f91ab073b8a78", ["sdxl_vae_encoder", "sdxl_vae_decoder"], [SDXLVAEEncoder, SDXLVAEDecoder], "diffusers"),
    (None, "d78aa6797382a6d455362358a3295ea9", ["sd_ipadapter_clip_image_encoder"], [IpAdapterCLIPImageEmbedder], "diffusers"),
    (None, "e291636cc15e803186b47404262ef812", ["sd_ipadapter"], [SDIpAdapter], "civitai"),
    (None, "399c81f2f8de8d1843d0127a00f3c224", ["sdxl_ipadapter_clip_image_encoder"], [IpAdapterXLCLIPImageEmbedder], "diffusers"),
    (None, "a64eac9aa0db4b9602213bc0131281c7", ["sdxl_ipadapter"], [SDXLIpAdapter], "civitai"),
    (None, "52817e4fdd89df154f02749ca6f692ac", ["sdxl_unet"], [SDXLUNet], "diffusers"),
    (None, "03343c606f16d834d6411d0902b53636", ["sd_text_encoder", "sd_unet", "sd_vae_decoder", "sd_vae_encoder"], [SDTextEncoder, SDUNet, SDVAEDecoder, SDVAEEncoder], "civitai"),
    (None, "d4ba77a7ece070679b4a987f58f201e9", ["sd_text_encoder"], [SDTextEncoder], "civitai"),
    (None, "d0c89e55c5a57cf3981def0cb1c9e65a", ["sd_vae_decoder", "sd_vae_encoder"], [SDVAEDecoder, SDVAEEncoder], "civitai"),
    (None, "3926bf373b39a67eeafd7901478a47a7", ["sd_unet"], [SDUNet], "civitai"),
    (None, "1e0c39ec176b9007c05f76d52b554a4d", ["sd3_text_encoder_1", "sd3_text_encoder_2", "sd3_dit", "sd3_vae_encoder", "sd3_vae_decoder"], [SD3TextEncoder1, SD3TextEncoder2, SD3DiT, SD3VAEEncoder, SD3VAEDecoder], "civitai"),
    (None, "d9e0290829ba8d98e28e1a2b1407db4a", ["sd3_text_encoder_1", "sd3_text_encoder_2", "sd3_text_encoder_3", "sd3_dit", "sd3_vae_encoder", "sd3_vae_decoder"], [SD3TextEncoder1, SD3TextEncoder2, SD3TextEncoder3, SD3DiT, SD3VAEEncoder, SD3VAEDecoder], "civitai"),
    (None, "5072d0b24e406b49507abe861cf97691", ["sd3_text_encoder_3"], [SD3TextEncoder3], "civitai"),
    (None, "4cf64a799d04260df438c6f33c9a047e", ["sdxl_text_encoder", "sdxl_text_encoder_2", "sdxl_unet", "sdxl_vae_decoder", "sdxl_vae_encoder"], [SDXLTextEncoder, SDXLTextEncoder2, SDXLUNet, SDXLVAEDecoder, SDXLVAEEncoder], "civitai"),
    (None, "d9b008a867c498ab12ad24042eff8e3f", ["sdxl_text_encoder", "sdxl_text_encoder_2", "sdxl_unet", "sdxl_vae_decoder", "sdxl_vae_encoder"], [SDXLTextEncoder, SDXLTextEncoder2, SDXLUNet, SDXLVAEDecoder, SDXLVAEEncoder], "civitai"), # SDXL-Turbo
    (None, "025bb7452e531a3853d951d77c63f032", ["sdxl_text_encoder", "sdxl_text_encoder_2"], [SDXLTextEncoder, SDXLTextEncoder2], "civitai"),
    (None, "298997b403a4245c04102c9f36aac348", ["sdxl_unet"], [SDXLUNet], "civitai"),
    (None, "2a07abce74b4bdc696b76254ab474da6", ["svd_image_encoder", "svd_unet", "svd_vae_decoder", "svd_vae_encoder"], [SVDImageEncoder, SVDUNet, SVDVAEDecoder, SVDVAEEncoder], "civitai"),
    (None, "c96a285a6888465f87de22a984d049fb", ["sd_motion_modules"], [SDMotionModel], "civitai"),
    (None, "72907b92caed19bdb2adb89aa4063fe2", ["sdxl_motion_modules"], [SDXLMotionModel], "civitai"),
    (None, "31d2d9614fba60511fc9bf2604aa01f7", ["sdxl_controlnet"], [SDXLControlNetUnion], "diffusers"),
    (None, "94eefa3dac9cec93cb1ebaf1747d7b78", ["flux_text_encoder_1"], [FluxTextEncoder1], "diffusers"),
    (None, "1aafa3cc91716fb6b300cc1cd51b85a3", ["flux_vae_encoder", "flux_vae_decoder"], [FluxVAEEncoder, FluxVAEDecoder], "diffusers"),
    (None, "21ea55f476dfc4fd135587abb59dfe5d", ["flux_vae_encoder", "flux_vae_decoder"], [FluxVAEEncoder, FluxVAEDecoder], "civitai"),
    (None, "a29710fea6dddb0314663ee823598e50", ["flux_dit"], [FluxDiT], "civitai")
]
huggingface_model_loader_configs = [
    # These configs are provided for detecting model type automatically.
    # The format is (architecture_in_huggingface_config, huggingface_lib, model_name, redirected_architecture)
    ("ChatGLMModel", "diffsynth.models.kolors_text_encoder", "kolors_text_encoder", None),
    ("MarianMTModel", "transformers.models.marian.modeling_marian", "translator", None),
    ("BloomForCausalLM", "transformers.models.bloom.modeling_bloom", "beautiful_prompt", None),
    ("LlamaForCausalLM", "transformers.models.llama.modeling_llama", "omost_prompt", None),
    ("T5EncoderModel", "diffsynth.models.flux_text_encoder", "flux_text_encoder_2", "FluxTextEncoder2"),
]
patch_model_loader_configs = [
    # These configs are provided for detecting model type automatically.
    # The format is (state_dict_keys_hash_with_shape, model_name, model_class, extra_kwargs)
    ("9a4ab6869ac9b7d6e31f9854e397c867", ["svd_unet"], [SVDUNet], {"add_positional_conv": 128}),
]

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
    # FLUX
    "FLUX.1-dev": [
        ("black-forest-labs/FLUX.1-dev", "text_encoder/model.safetensors", "models/FLUX/FLUX.1-dev/text_encoder"),
        ("black-forest-labs/FLUX.1-dev", "text_encoder_2/config.json", "models/FLUX/FLUX.1-dev/text_encoder_2"),
        ("black-forest-labs/FLUX.1-dev", "text_encoder_2/model-00001-of-00002.safetensors", "models/FLUX/FLUX.1-dev/text_encoder_2"),
        ("black-forest-labs/FLUX.1-dev", "text_encoder_2/model-00002-of-00002.safetensors", "models/FLUX/FLUX.1-dev/text_encoder_2"),
        ("black-forest-labs/FLUX.1-dev", "text_encoder_2/model.safetensors.index.json", "models/FLUX/FLUX.1-dev/text_encoder_2"),
        ("black-forest-labs/FLUX.1-dev", "ae.safetensors", "models/FLUX/FLUX.1-dev"),
        ("black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors", "models/FLUX/FLUX.1-dev"),
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
    "SDXL_lora_zyd232_ChineseInkStyle_SDXL_v1_0": [
        ("sd_lora/zyd232_ChineseInkStyle_SDXL_v1_0", "zyd232_ChineseInkStyle_SDXL_v1_0.safetensors", "models/lora"),
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
    "ControlNet_union_sdxl_promax": [
        ("AI-ModelScope/controlnet-union-sdxl-1.0", "diffusion_pytorch_model_promax.safetensors", "models/ControlNet/controlnet_union"),
        ("sd_lora/Annotators", "dpt_hybrid-midas-501f0c75.pt", "models/Annotators")
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
    # Omost prompt
    "OmostPrompt":[
        ("Omost/omost-llama-3-8b-4bits", "model-00001-of-00002.safetensors", "models/OmostPrompt/omost-llama-3-8b-4bits"),
        ("Omost/omost-llama-3-8b-4bits", "model-00002-of-00002.safetensors", "models/OmostPrompt/omost-llama-3-8b-4bits"),
        ("Omost/omost-llama-3-8b-4bits", "tokenizer.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
        ("Omost/omost-llama-3-8b-4bits", "tokenizer_config.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),  
        ("Omost/omost-llama-3-8b-4bits", "config.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
        ("Omost/omost-llama-3-8b-4bits", "generation_config.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
        ("Omost/omost-llama-3-8b-4bits", "model.safetensors.index.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
        ("Omost/omost-llama-3-8b-4bits", "special_tokens_map.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
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
    "SDXL-vae-fp16-fix": [
        ("AI-ModelScope/sdxl-vae-fp16-fix", "diffusion_pytorch_model.safetensors", "models/sdxl-vae-fp16-fix")
    ],
    # FLUX
    "FLUX.1-dev": [
        ("AI-ModelScope/FLUX.1-dev", "text_encoder/model.safetensors", "models/FLUX/FLUX.1-dev/text_encoder"),
        ("AI-ModelScope/FLUX.1-dev", "text_encoder_2/config.json", "models/FLUX/FLUX.1-dev/text_encoder_2"),
        ("AI-ModelScope/FLUX.1-dev", "text_encoder_2/model-00001-of-00002.safetensors", "models/FLUX/FLUX.1-dev/text_encoder_2"),
        ("AI-ModelScope/FLUX.1-dev", "text_encoder_2/model-00002-of-00002.safetensors", "models/FLUX/FLUX.1-dev/text_encoder_2"),
        ("AI-ModelScope/FLUX.1-dev", "text_encoder_2/model.safetensors.index.json", "models/FLUX/FLUX.1-dev/text_encoder_2"),
        ("AI-ModelScope/FLUX.1-dev", "ae.safetensors", "models/FLUX/FLUX.1-dev"),
        ("AI-ModelScope/FLUX.1-dev", "flux1-dev.safetensors", "models/FLUX/FLUX.1-dev"),
    ]
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
    "SDXL-vae-fp16-fix",
    "ControlNet_union_sdxl_promax",
    "FLUX.1-dev",
    "SDXL_lora_zyd232_ChineseInkStyle_SDXL_v1_0",
    "OmostPrompt",
]