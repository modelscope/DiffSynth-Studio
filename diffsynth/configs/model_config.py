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
from ..models.flux_text_encoder import FluxTextEncoder2
from ..models.flux_vae import FluxVAEEncoder, FluxVAEDecoder
from ..models.flux_controlnet import FluxControlNet
from ..models.flux_ipadapter import FluxIpAdapter
from ..models.flux_infiniteyou import InfiniteYouImageProjector

from ..models.cog_vae import CogVAEEncoder, CogVAEDecoder
from ..models.cog_dit import CogDiT

from ..models.omnigen import OmniGenTransformer

from ..models.hunyuan_video_vae_decoder import HunyuanVideoVAEDecoder
from ..models.hunyuan_video_vae_encoder import HunyuanVideoVAEEncoder

from ..extensions.RIFE import IFNet
from ..extensions.ESRGAN import RRDBNet

from ..models.hunyuan_video_dit import HunyuanVideoDiT

from ..models.stepvideo_vae import StepVideoVAE
from ..models.stepvideo_dit import StepVideoModel

from ..models.wan_video_dit import WanModel
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..models.wan_video_vace import VaceWanModel

from ..models.step1x_connector import Qwen2Connector


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
    (None, "94eefa3dac9cec93cb1ebaf1747d7b78", ["sd3_text_encoder_1"], [SD3TextEncoder1], "diffusers"),
    (None, "1aafa3cc91716fb6b300cc1cd51b85a3", ["flux_vae_encoder", "flux_vae_decoder"], [FluxVAEEncoder, FluxVAEDecoder], "diffusers"),
    (None, "21ea55f476dfc4fd135587abb59dfe5d", ["flux_vae_encoder", "flux_vae_decoder"], [FluxVAEEncoder, FluxVAEDecoder], "civitai"),
    (None, "a29710fea6dddb0314663ee823598e50", ["flux_dit"], [FluxDiT], "civitai"),
    (None, "57b02550baab820169365b3ee3afa2c9", ["flux_dit"], [FluxDiT], "civitai"),
    (None, "3394f306c4cbf04334b712bf5aaed95f", ["flux_dit"], [FluxDiT], "civitai"),
    (None, "023f054d918a84ccf503481fd1e3379e", ["flux_dit"], [FluxDiT], "civitai"),
    (None, "d02f41c13549fa5093d3521f62a5570a", ["flux_dit"], [FluxDiT], "civitai"),
    (None, "605c56eab23e9e2af863ad8f0813a25d", ["flux_dit"], [FluxDiT], "diffusers"),
    (None, "280189ee084bca10f70907bf6ce1649d", ["cog_vae_encoder", "cog_vae_decoder"], [CogVAEEncoder, CogVAEDecoder], "diffusers"),
    (None, "9b9313d104ac4df27991352fec013fd4", ["rife"], [IFNet], "civitai"),
    (None, "6b7116078c4170bfbeaedc8fe71f6649", ["esrgan"], [RRDBNet], "civitai"),
    (None, "61cbcbc7ac11f169c5949223efa960d1", ["omnigen_transformer"], [OmniGenTransformer], "diffusers"),
    (None, "78d18b9101345ff695f312e7e62538c0", ["flux_controlnet"], [FluxControlNet], "diffusers"),
    (None, "b001c89139b5f053c715fe772362dd2a", ["flux_controlnet"], [FluxControlNet], "diffusers"),
    (None, "52357cb26250681367488a8954c271e8", ["flux_controlnet"], [FluxControlNet], "diffusers"),
    (None, "0cfd1740758423a2a854d67c136d1e8c", ["flux_controlnet"], [FluxControlNet], "diffusers"),
    (None, "7f9583eb8ba86642abb9a21a4b2c9e16", ["flux_controlnet"], [FluxControlNet], "diffusers"),
    (None, "43ad5aaa27dd4ee01b832ed16773fa52", ["flux_controlnet"], [FluxControlNet], "diffusers"),
    (None, "c07c0f04f5ff55e86b4e937c7a40d481", ["infiniteyou_image_projector"], [InfiniteYouImageProjector], "diffusers"),
    (None, "4daaa66cc656a8fe369908693dad0a35", ["flux_ipadapter"], [FluxIpAdapter], "diffusers"),
    (None, "51aed3d27d482fceb5e0739b03060e8f", ["sd3_dit", "sd3_vae_encoder", "sd3_vae_decoder"], [SD3DiT, SD3VAEEncoder, SD3VAEDecoder], "civitai"),
    (None, "98cc34ccc5b54ae0e56bdea8688dcd5a", ["sd3_text_encoder_2"], [SD3TextEncoder2], "civitai"),
    (None, "77ff18050dbc23f50382e45d51a779fe", ["sd3_dit", "sd3_vae_encoder", "sd3_vae_decoder"], [SD3DiT, SD3VAEEncoder, SD3VAEDecoder], "civitai"),
    (None, "5da81baee73198a7c19e6d2fe8b5148e", ["sd3_text_encoder_1"], [SD3TextEncoder1], "diffusers"),
    (None, "aeb82dce778a03dcb4d726cb03f3c43f", ["hunyuan_video_vae_decoder", "hunyuan_video_vae_encoder"], [HunyuanVideoVAEDecoder, HunyuanVideoVAEEncoder], "diffusers"),
    (None, "b9588f02e78f5ccafc9d7c0294e46308", ["hunyuan_video_dit"], [HunyuanVideoDiT], "civitai"),
    (None, "84ef4bd4757f60e906b54aa6a7815dc6", ["hunyuan_video_dit"], [HunyuanVideoDiT], "civitai"),
    (None, "68beaf8429b7c11aa8ca05b1bd0058bd", ["stepvideo_vae"], [StepVideoVAE], "civitai"),
    (None, "5c0216a2132b082c10cb7a0e0377e681", ["stepvideo_dit"], [StepVideoModel], "civitai"),
    (None, "9269f8db9040a9d860eaca435be61814", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "aafcfd9672c3a2456dc46e1cb6e52c70", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6d6ccde6845b95ad9114ab993d917893", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "349723183fc063b2bfc10bb2835cf677", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "efa44cddf936c70abd0ea28b6cbe946c", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "3ef3b1f8e1dab83d5b71fd7b617f859f", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "70ddad9d3a133785da5ea371aae09504", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "26bde73488a92e64cc20b0a7485b9e5b", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "ac6a5aa74f4a0aab6f64eb9a72f19901", ["wan_video_dit"], [WanModel], "civitai"), 
    (None, "b61c605c2adbd23124d152ed28e049ae", ["wan_video_dit"], [WanModel], "civitai"), 
    (None, "a61453409b67cd3246cf0c3bebad47ba", ["wan_video_dit", "wan_video_vace"], [WanModel, VaceWanModel], "civitai"),
    (None, "7a513e1f257a861512b1afd387a8ecd9", ["wan_video_dit", "wan_video_vace"], [WanModel, VaceWanModel], "civitai"),
    (None, "cb104773c6c2cb6df4f9529ad5c60d0b", ["wan_video_dit"], [WanModel], "diffusers"),
    (None, "9c8818c2cbea55eca56c7b447df170da", ["wan_video_text_encoder"], [WanTextEncoder], "civitai"),
    (None, "5941c53e207d62f20f9025686193c40b", ["wan_video_image_encoder"], [WanImageEncoder], "civitai"),
    (None, "1378ea763357eea97acdef78e65d6d96", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "ccc42284ea13e1ad04693284c7a09be6", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "dbd5ec76bbf977983f972c151d545389", ["wan_video_motion_controller"], [WanMotionControllerModel], "civitai"),
    (None, "d30fb9e02b1dbf4e509142f05cf7dd50", ["flux_dit", "step1x_connector"], [FluxDiT, Qwen2Connector], "civitai"),
]
huggingface_model_loader_configs = [
    # These configs are provided for detecting model type automatically.
    # The format is (architecture_in_huggingface_config, huggingface_lib, model_name, redirected_architecture)
    ("ChatGLMModel", "diffsynth.models.kolors_text_encoder", "kolors_text_encoder", None),
    ("MarianMTModel", "transformers.models.marian.modeling_marian", "translator", None),
    ("BloomForCausalLM", "transformers.models.bloom.modeling_bloom", "beautiful_prompt", None),
    ("Qwen2ForCausalLM", "transformers.models.qwen2.modeling_qwen2", "qwen_prompt", None),
    # ("LlamaForCausalLM", "transformers.models.llama.modeling_llama", "omost_prompt", None),
    ("T5EncoderModel", "diffsynth.models.flux_text_encoder", "flux_text_encoder_2", "FluxTextEncoder2"),
    ("CogVideoXTransformer3DModel", "diffsynth.models.cog_dit", "cog_dit", "CogDiT"),
    ("SiglipModel", "transformers.models.siglip.modeling_siglip", "siglip_vision_model", "SiglipVisionModel"),
    ("LlamaForCausalLM", "diffsynth.models.hunyuan_video_text_encoder", "hunyuan_video_text_encoder_2", "HunyuanVideoLLMEncoder"),
    ("LlavaForConditionalGeneration", "diffsynth.models.hunyuan_video_text_encoder", "hunyuan_video_text_encoder_2", "HunyuanVideoMLLMEncoder"),
    ("Step1Model", "diffsynth.models.stepvideo_text_encoder", "stepvideo_text_encoder_2", "STEP1TextEncoder"),
    ("Qwen2_5_VLForConditionalGeneration", "diffsynth.models.qwenvl", "qwenvl", "Qwen25VL_7b_Embedder"),
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
    # Stable Diffusion
    "StableDiffusion_v15": [
        ("benjamin-paine/stable-diffusion-v1-5", "v1-5-pruned-emaonly.safetensors", "models/stable_diffusion"),
    ],
    "DreamShaper_8": [
        ("Yntec/Dreamshaper8", "dreamshaper_8.safetensors", "models/stable_diffusion"),
    ],
    # Textual Inversion
    "TextualInversion_VeryBadImageNegative_v1.3": [
        ("gemasai/verybadimagenegative_v1.3", "verybadimagenegative_v1.3.pt", "models/textual_inversion"),
    ],
    # Stable Diffusion XL
    "StableDiffusionXL_v1": [
        ("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors", "models/stable_diffusion_xl"),
    ],
    "BluePencilXL_v200": [
        ("frankjoshua/bluePencilXL_v200", "bluePencilXL_v200.safetensors", "models/stable_diffusion_xl"),
    ],
    "StableDiffusionXL_Turbo": [
        ("stabilityai/sdxl-turbo", "sd_xl_turbo_1.0_fp16.safetensors", "models/stable_diffusion_xl_turbo"),
    ],
    # Stable Diffusion 3
    "StableDiffusion3": [
        ("stabilityai/stable-diffusion-3-medium", "sd3_medium_incl_clips_t5xxlfp16.safetensors", "models/stable_diffusion_3"),
    ],
    "StableDiffusion3_without_T5": [
        ("stabilityai/stable-diffusion-3-medium", "sd3_medium_incl_clips.safetensors", "models/stable_diffusion_3"),
    ],
    # ControlNet
    "ControlNet_v11f1p_sd15_depth": [
        ("lllyasviel/ControlNet-v1-1", "control_v11f1p_sd15_depth.pth", "models/ControlNet"),
        ("lllyasviel/Annotators", "dpt_hybrid-midas-501f0c75.pt", "models/Annotators")
    ],
    "ControlNet_v11p_sd15_softedge": [
        ("lllyasviel/ControlNet-v1-1", "control_v11p_sd15_softedge.pth", "models/ControlNet"),
        ("lllyasviel/Annotators", "ControlNetHED.pth", "models/Annotators")
    ],
    "ControlNet_v11f1e_sd15_tile": [
        ("lllyasviel/ControlNet-v1-1", "control_v11f1e_sd15_tile.pth", "models/ControlNet")
    ],
    "ControlNet_v11p_sd15_lineart": [
        ("lllyasviel/ControlNet-v1-1", "control_v11p_sd15_lineart.pth", "models/ControlNet"),
        ("lllyasviel/Annotators", "sk_model.pth", "models/Annotators"),
        ("lllyasviel/Annotators", "sk_model2.pth", "models/Annotators")
    ],
    "ControlNet_union_sdxl_promax": [
        ("xinsir/controlnet-union-sdxl-1.0", "diffusion_pytorch_model_promax.safetensors", "models/ControlNet/controlnet_union"),
        ("lllyasviel/Annotators", "dpt_hybrid-midas-501f0c75.pt", "models/Annotators")
    ],
    # AnimateDiff
    "AnimateDiff_v2": [
        ("guoyww/animatediff", "mm_sd_v15_v2.ckpt", "models/AnimateDiff"),
    ],
    "AnimateDiff_xl_beta": [
        ("guoyww/animatediff", "mm_sdxl_v10_beta.ckpt", "models/AnimateDiff"),
    ],

    # Qwen Prompt
    "QwenPrompt": [
        ("Qwen/Qwen2-1.5B-Instruct", "config.json", "models/QwenPrompt/qwen2-1.5b-instruct"),
        ("Qwen/Qwen2-1.5B-Instruct", "generation_config.json", "models/QwenPrompt/qwen2-1.5b-instruct"),
        ("Qwen/Qwen2-1.5B-Instruct", "model.safetensors", "models/QwenPrompt/qwen2-1.5b-instruct"),
        ("Qwen/Qwen2-1.5B-Instruct", "special_tokens_map.json", "models/QwenPrompt/qwen2-1.5b-instruct"),
        ("Qwen/Qwen2-1.5B-Instruct", "tokenizer.json", "models/QwenPrompt/qwen2-1.5b-instruct"),
        ("Qwen/Qwen2-1.5B-Instruct", "tokenizer_config.json", "models/QwenPrompt/qwen2-1.5b-instruct"),
        ("Qwen/Qwen2-1.5B-Instruct", "merges.txt", "models/QwenPrompt/qwen2-1.5b-instruct"),
        ("Qwen/Qwen2-1.5B-Instruct", "vocab.json", "models/QwenPrompt/qwen2-1.5b-instruct"),
    ],
    # Beautiful Prompt
    "BeautifulPrompt": [
        ("alibaba-pai/pai-bloom-1b1-text2prompt-sd", "config.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
        ("alibaba-pai/pai-bloom-1b1-text2prompt-sd", "generation_config.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
        ("alibaba-pai/pai-bloom-1b1-text2prompt-sd", "model.safetensors", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
        ("alibaba-pai/pai-bloom-1b1-text2prompt-sd", "special_tokens_map.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
        ("alibaba-pai/pai-bloom-1b1-text2prompt-sd", "tokenizer.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
        ("alibaba-pai/pai-bloom-1b1-text2prompt-sd", "tokenizer_config.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
    ],
    # Omost prompt
    "OmostPrompt":[
        ("lllyasviel/omost-llama-3-8b-4bits", "model-00001-of-00002.safetensors", "models/OmostPrompt/omost-llama-3-8b-4bits"),
        ("lllyasviel/omost-llama-3-8b-4bits", "model-00002-of-00002.safetensors", "models/OmostPrompt/omost-llama-3-8b-4bits"),
        ("lllyasviel/omost-llama-3-8b-4bits", "tokenizer.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
        ("lllyasviel/omost-llama-3-8b-4bits", "tokenizer_config.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),  
        ("lllyasviel/omost-llama-3-8b-4bits", "config.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
        ("lllyasviel/omost-llama-3-8b-4bits", "generation_config.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
        ("lllyasviel/omost-llama-3-8b-4bits", "model.safetensors.index.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
        ("lllyasviel/omost-llama-3-8b-4bits", "special_tokens_map.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
    ],
    # Translator
    "opus-mt-zh-en": [
        ("Helsinki-NLP/opus-mt-zh-en", "config.json", "models/translator/opus-mt-zh-en"),
        ("Helsinki-NLP/opus-mt-zh-en", "generation_config.json", "models/translator/opus-mt-zh-en"),
        ("Helsinki-NLP/opus-mt-zh-en", "metadata.json", "models/translator/opus-mt-zh-en"),
        ("Helsinki-NLP/opus-mt-zh-en", "pytorch_model.bin", "models/translator/opus-mt-zh-en"),
        ("Helsinki-NLP/opus-mt-zh-en", "source.spm", "models/translator/opus-mt-zh-en"),
        ("Helsinki-NLP/opus-mt-zh-en", "target.spm", "models/translator/opus-mt-zh-en"),
        ("Helsinki-NLP/opus-mt-zh-en", "tokenizer_config.json", "models/translator/opus-mt-zh-en"),
        ("Helsinki-NLP/opus-mt-zh-en", "vocab.json", "models/translator/opus-mt-zh-en"),
    ],
    # IP-Adapter
    "IP-Adapter-SD": [
        ("h94/IP-Adapter", "models/image_encoder/model.safetensors", "models/IpAdapter/stable_diffusion/image_encoder"),
        ("h94/IP-Adapter", "models/ip-adapter_sd15.bin", "models/IpAdapter/stable_diffusion"),
    ],
    "IP-Adapter-SDXL": [
        ("h94/IP-Adapter", "sdxl_models/image_encoder/model.safetensors", "models/IpAdapter/stable_diffusion_xl/image_encoder"),
        ("h94/IP-Adapter", "sdxl_models/ip-adapter_sdxl.bin", "models/IpAdapter/stable_diffusion_xl"),
    ],
    "SDXL-vae-fp16-fix": [
        ("madebyollin/sdxl-vae-fp16-fix", "diffusion_pytorch_model.safetensors", "models/sdxl-vae-fp16-fix")
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
    "InstantX/FLUX.1-dev-IP-Adapter": {
        "file_list": [
            ("InstantX/FLUX.1-dev-IP-Adapter", "ip-adapter.bin", "models/IpAdapter/InstantX/FLUX.1-dev-IP-Adapter"),
            ("google/siglip-so400m-patch14-384", "model.safetensors", "models/IpAdapter/InstantX/FLUX.1-dev-IP-Adapter/image_encoder"),
            ("google/siglip-so400m-patch14-384", "config.json", "models/IpAdapter/InstantX/FLUX.1-dev-IP-Adapter/image_encoder"),
        ],
        "load_path": [
            "models/IpAdapter/InstantX/FLUX.1-dev-IP-Adapter/ip-adapter.bin",
            "models/IpAdapter/InstantX/FLUX.1-dev-IP-Adapter/image_encoder",
        ],
    },
    # RIFE
    "RIFE": [
        ("AlexWortega/RIFE", "flownet.pkl", "models/RIFE"),
    ],
    # CogVideo
    "CogVideoX-5B": [
        ("THUDM/CogVideoX-5b", "text_encoder/config.json", "models/CogVideo/CogVideoX-5b/text_encoder"),
        ("THUDM/CogVideoX-5b", "text_encoder/model.safetensors.index.json", "models/CogVideo/CogVideoX-5b/text_encoder"),
        ("THUDM/CogVideoX-5b", "text_encoder/model-00001-of-00002.safetensors", "models/CogVideo/CogVideoX-5b/text_encoder"),
        ("THUDM/CogVideoX-5b", "text_encoder/model-00002-of-00002.safetensors", "models/CogVideo/CogVideoX-5b/text_encoder"),
        ("THUDM/CogVideoX-5b", "transformer/config.json", "models/CogVideo/CogVideoX-5b/transformer"),
        ("THUDM/CogVideoX-5b", "transformer/diffusion_pytorch_model.safetensors.index.json", "models/CogVideo/CogVideoX-5b/transformer"),
        ("THUDM/CogVideoX-5b", "transformer/diffusion_pytorch_model-00001-of-00002.safetensors", "models/CogVideo/CogVideoX-5b/transformer"),
        ("THUDM/CogVideoX-5b", "transformer/diffusion_pytorch_model-00002-of-00002.safetensors", "models/CogVideo/CogVideoX-5b/transformer"),
        ("THUDM/CogVideoX-5b", "vae/diffusion_pytorch_model.safetensors", "models/CogVideo/CogVideoX-5b/vae"),
    ],
    # Stable Diffusion 3.5
    "StableDiffusion3.5-large": [
        ("stabilityai/stable-diffusion-3.5-large", "sd3.5_large.safetensors", "models/stable_diffusion_3"),
        ("stabilityai/stable-diffusion-3.5-large", "text_encoders/clip_l.safetensors", "models/stable_diffusion_3/text_encoders"),
        ("stabilityai/stable-diffusion-3.5-large", "text_encoders/clip_g.safetensors", "models/stable_diffusion_3/text_encoders"),
        ("stabilityai/stable-diffusion-3.5-large", "text_encoders/t5xxl_fp16.safetensors", "models/stable_diffusion_3/text_encoders"),
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
    "ExVideo-CogVideoX-LoRA-129f-v1": [
        ("ECNU-CILab/ExVideo-CogVideoX-LoRA-129f-v1", "ExVideo-CogVideoX-LoRA-129f-v1.safetensors", "models/lora"),
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
    "Annotators:Depth": [
        ("sd_lora/Annotators", "dpt_hybrid-midas-501f0c75.pt", "models/Annotators"),
    ],
    "Annotators:Softedge": [
        ("sd_lora/Annotators", "ControlNetHED.pth", "models/Annotators"),
    ],
    "Annotators:Lineart": [
        ("sd_lora/Annotators", "sk_model.pth", "models/Annotators"),
        ("sd_lora/Annotators", "sk_model2.pth", "models/Annotators"),
    ],
    "Annotators:Normal": [
        ("sd_lora/Annotators", "scannet.pt", "models/Annotators"),
    ],
    "Annotators:Openpose": [
        ("sd_lora/Annotators", "body_pose_model.pth", "models/Annotators"),
        ("sd_lora/Annotators", "facenet.pth", "models/Annotators"),
        ("sd_lora/Annotators", "hand_pose_model.pth", "models/Annotators"),
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
    # Qwen Prompt
    "QwenPrompt": {
        "file_list": [
            ("qwen/Qwen2-1.5B-Instruct", "config.json", "models/QwenPrompt/qwen2-1.5b-instruct"),
            ("qwen/Qwen2-1.5B-Instruct", "generation_config.json", "models/QwenPrompt/qwen2-1.5b-instruct"),
            ("qwen/Qwen2-1.5B-Instruct", "model.safetensors", "models/QwenPrompt/qwen2-1.5b-instruct"),
            ("qwen/Qwen2-1.5B-Instruct", "special_tokens_map.json", "models/QwenPrompt/qwen2-1.5b-instruct"),
            ("qwen/Qwen2-1.5B-Instruct", "tokenizer.json", "models/QwenPrompt/qwen2-1.5b-instruct"),
            ("qwen/Qwen2-1.5B-Instruct", "tokenizer_config.json", "models/QwenPrompt/qwen2-1.5b-instruct"),
            ("qwen/Qwen2-1.5B-Instruct", "merges.txt", "models/QwenPrompt/qwen2-1.5b-instruct"),
            ("qwen/Qwen2-1.5B-Instruct", "vocab.json", "models/QwenPrompt/qwen2-1.5b-instruct"),
        ],
        "load_path": [
            "models/QwenPrompt/qwen2-1.5b-instruct",
        ],
    },
    # Beautiful Prompt
    "BeautifulPrompt": {
        "file_list": [
            ("AI-ModelScope/pai-bloom-1b1-text2prompt-sd", "config.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
            ("AI-ModelScope/pai-bloom-1b1-text2prompt-sd", "generation_config.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
            ("AI-ModelScope/pai-bloom-1b1-text2prompt-sd", "model.safetensors", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
            ("AI-ModelScope/pai-bloom-1b1-text2prompt-sd", "special_tokens_map.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
            ("AI-ModelScope/pai-bloom-1b1-text2prompt-sd", "tokenizer.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
            ("AI-ModelScope/pai-bloom-1b1-text2prompt-sd", "tokenizer_config.json", "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd"),
        ],
        "load_path": [
            "models/BeautifulPrompt/pai-bloom-1b1-text2prompt-sd",
        ],
    },
    # Omost prompt
    "OmostPrompt": {
        "file_list": [
            ("Omost/omost-llama-3-8b-4bits", "model-00001-of-00002.safetensors", "models/OmostPrompt/omost-llama-3-8b-4bits"),
            ("Omost/omost-llama-3-8b-4bits", "model-00002-of-00002.safetensors", "models/OmostPrompt/omost-llama-3-8b-4bits"),
            ("Omost/omost-llama-3-8b-4bits", "tokenizer.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
            ("Omost/omost-llama-3-8b-4bits", "tokenizer_config.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),  
            ("Omost/omost-llama-3-8b-4bits", "config.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
            ("Omost/omost-llama-3-8b-4bits", "generation_config.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
            ("Omost/omost-llama-3-8b-4bits", "model.safetensors.index.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
            ("Omost/omost-llama-3-8b-4bits", "special_tokens_map.json", "models/OmostPrompt/omost-llama-3-8b-4bits"),
        ],
        "load_path": [
            "models/OmostPrompt/omost-llama-3-8b-4bits",
        ],
    },
    # Translator
    "opus-mt-zh-en": {
        "file_list": [
            ("moxying/opus-mt-zh-en", "config.json", "models/translator/opus-mt-zh-en"),
            ("moxying/opus-mt-zh-en", "generation_config.json", "models/translator/opus-mt-zh-en"),
            ("moxying/opus-mt-zh-en", "metadata.json", "models/translator/opus-mt-zh-en"),
            ("moxying/opus-mt-zh-en", "pytorch_model.bin", "models/translator/opus-mt-zh-en"),
            ("moxying/opus-mt-zh-en", "source.spm", "models/translator/opus-mt-zh-en"),
            ("moxying/opus-mt-zh-en", "target.spm", "models/translator/opus-mt-zh-en"),
            ("moxying/opus-mt-zh-en", "tokenizer_config.json", "models/translator/opus-mt-zh-en"),
            ("moxying/opus-mt-zh-en", "vocab.json", "models/translator/opus-mt-zh-en"),
        ],
        "load_path": [
            "models/translator/opus-mt-zh-en",
        ],
    },
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
    "Kolors": {
        "file_list": [
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
        "load_path": [
            "models/kolors/Kolors/text_encoder",
            "models/kolors/Kolors/unet/diffusion_pytorch_model.safetensors",
            "models/kolors/Kolors/vae/diffusion_pytorch_model.safetensors",
        ],
    },
    "SDXL-vae-fp16-fix": [
        ("AI-ModelScope/sdxl-vae-fp16-fix", "diffusion_pytorch_model.safetensors", "models/sdxl-vae-fp16-fix")
    ],
    # FLUX
    "FLUX.1-dev": {
        "file_list": [
            ("AI-ModelScope/FLUX.1-dev", "text_encoder/model.safetensors", "models/FLUX/FLUX.1-dev/text_encoder"),
            ("AI-ModelScope/FLUX.1-dev", "text_encoder_2/config.json", "models/FLUX/FLUX.1-dev/text_encoder_2"),
            ("AI-ModelScope/FLUX.1-dev", "text_encoder_2/model-00001-of-00002.safetensors", "models/FLUX/FLUX.1-dev/text_encoder_2"),
            ("AI-ModelScope/FLUX.1-dev", "text_encoder_2/model-00002-of-00002.safetensors", "models/FLUX/FLUX.1-dev/text_encoder_2"),
            ("AI-ModelScope/FLUX.1-dev", "text_encoder_2/model.safetensors.index.json", "models/FLUX/FLUX.1-dev/text_encoder_2"),
            ("AI-ModelScope/FLUX.1-dev", "ae.safetensors", "models/FLUX/FLUX.1-dev"),
            ("AI-ModelScope/FLUX.1-dev", "flux1-dev.safetensors", "models/FLUX/FLUX.1-dev"),
        ],
        "load_path": [
            "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
            "models/FLUX/FLUX.1-dev/text_encoder_2",
            "models/FLUX/FLUX.1-dev/ae.safetensors",
            "models/FLUX/FLUX.1-dev/flux1-dev.safetensors"
        ],
    },
    "FLUX.1-schnell": {
        "file_list": [
            ("AI-ModelScope/FLUX.1-dev", "text_encoder/model.safetensors", "models/FLUX/FLUX.1-dev/text_encoder"),
            ("AI-ModelScope/FLUX.1-dev", "text_encoder_2/config.json", "models/FLUX/FLUX.1-dev/text_encoder_2"),
            ("AI-ModelScope/FLUX.1-dev", "text_encoder_2/model-00001-of-00002.safetensors", "models/FLUX/FLUX.1-dev/text_encoder_2"),
            ("AI-ModelScope/FLUX.1-dev", "text_encoder_2/model-00002-of-00002.safetensors", "models/FLUX/FLUX.1-dev/text_encoder_2"),
            ("AI-ModelScope/FLUX.1-dev", "text_encoder_2/model.safetensors.index.json", "models/FLUX/FLUX.1-dev/text_encoder_2"),
            ("AI-ModelScope/FLUX.1-dev", "ae.safetensors", "models/FLUX/FLUX.1-dev"),
            ("AI-ModelScope/FLUX.1-schnell", "flux1-schnell.safetensors", "models/FLUX/FLUX.1-schnell"),
        ],
        "load_path": [
            "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
            "models/FLUX/FLUX.1-dev/text_encoder_2",
            "models/FLUX/FLUX.1-dev/ae.safetensors",
            "models/FLUX/FLUX.1-schnell/flux1-schnell.safetensors"
        ],
    },
    "InstantX/FLUX.1-dev-Controlnet-Union-alpha": [
        ("InstantX/FLUX.1-dev-Controlnet-Union-alpha", "diffusion_pytorch_model.safetensors", "models/ControlNet/InstantX/FLUX.1-dev-Controlnet-Union-alpha"),
    ],
    "jasperai/Flux.1-dev-Controlnet-Depth": [
        ("jasperai/Flux.1-dev-Controlnet-Depth", "diffusion_pytorch_model.safetensors", "models/ControlNet/jasperai/Flux.1-dev-Controlnet-Depth"),
    ],
    "jasperai/Flux.1-dev-Controlnet-Surface-Normals": [
        ("jasperai/Flux.1-dev-Controlnet-Surface-Normals", "diffusion_pytorch_model.safetensors", "models/ControlNet/jasperai/Flux.1-dev-Controlnet-Surface-Normals"),
    ],
    "jasperai/Flux.1-dev-Controlnet-Upscaler": [
        ("jasperai/Flux.1-dev-Controlnet-Upscaler", "diffusion_pytorch_model.safetensors", "models/ControlNet/jasperai/Flux.1-dev-Controlnet-Upscaler"),
    ],
    "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha": [
        ("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha", "diffusion_pytorch_model.safetensors", "models/ControlNet/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha"),
    ],
    "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta": [
        ("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", "diffusion_pytorch_model.safetensors", "models/ControlNet/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta"),
    ],
    "Shakker-Labs/FLUX.1-dev-ControlNet-Depth": [
        ("Shakker-Labs/FLUX.1-dev-ControlNet-Depth", "diffusion_pytorch_model.safetensors", "models/ControlNet/Shakker-Labs/FLUX.1-dev-ControlNet-Depth"),
    ],
    "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro": [
        ("Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro", "diffusion_pytorch_model.safetensors", "models/ControlNet/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"),
    ],
    "InstantX/FLUX.1-dev-IP-Adapter": {
        "file_list": [
            ("InstantX/FLUX.1-dev-IP-Adapter", "ip-adapter.bin", "models/IpAdapter/InstantX/FLUX.1-dev-IP-Adapter"),
            ("AI-ModelScope/siglip-so400m-patch14-384", "model.safetensors", "models/IpAdapter/InstantX/FLUX.1-dev-IP-Adapter/image_encoder"),
            ("AI-ModelScope/siglip-so400m-patch14-384", "config.json", "models/IpAdapter/InstantX/FLUX.1-dev-IP-Adapter/image_encoder"),
        ],
        "load_path": [
            "models/IpAdapter/InstantX/FLUX.1-dev-IP-Adapter/ip-adapter.bin",
            "models/IpAdapter/InstantX/FLUX.1-dev-IP-Adapter/image_encoder",
        ],
    },
    "InfiniteYou":{
        "file_list":[
            ("ByteDance/InfiniteYou", "infu_flux_v1.0/aes_stage2/InfuseNetModel/diffusion_pytorch_model-00001-of-00002.safetensors", "models/InfiniteYou/InfuseNetModel"),
            ("ByteDance/InfiniteYou", "infu_flux_v1.0/aes_stage2/InfuseNetModel/diffusion_pytorch_model-00002-of-00002.safetensors", "models/InfiniteYou/InfuseNetModel"),
            ("ByteDance/InfiniteYou", "infu_flux_v1.0/aes_stage2/image_proj_model.bin", "models/InfiniteYou"),
            ("ByteDance/InfiniteYou", "supports/insightface/models/antelopev2/1k3d68.onnx", "models/InfiniteYou/insightface/models/antelopev2"),
            ("ByteDance/InfiniteYou", "supports/insightface/models/antelopev2/2d106det.onnx", "models/InfiniteYou/insightface/models/antelopev2"),
            ("ByteDance/InfiniteYou", "supports/insightface/models/antelopev2/genderage.onnx", "models/InfiniteYou/insightface/models/antelopev2"),
            ("ByteDance/InfiniteYou", "supports/insightface/models/antelopev2/glintr100.onnx", "models/InfiniteYou/insightface/models/antelopev2"),
            ("ByteDance/InfiniteYou", "supports/insightface/models/antelopev2/scrfd_10g_bnkps.onnx", "models/InfiniteYou/insightface/models/antelopev2"),            
        ],
        "load_path":[
            [
                "models/InfiniteYou/InfuseNetModel/diffusion_pytorch_model-00001-of-00002.safetensors",
                "models/InfiniteYou/InfuseNetModel/diffusion_pytorch_model-00002-of-00002.safetensors"
            ],
            "models/InfiniteYou/image_proj_model.bin",
            ],
    },
    # ESRGAN
    "ESRGAN_x4": [
        ("AI-ModelScope/Real-ESRGAN", "RealESRGAN_x4.pth", "models/ESRGAN"),
    ],
    # RIFE
    "RIFE": [
        ("AI-ModelScope/RIFE", "flownet.pkl", "models/RIFE"),
    ],
    # Omnigen
    "OmniGen-v1": {
        "file_list": [
            ("BAAI/OmniGen-v1", "vae/diffusion_pytorch_model.safetensors", "models/OmniGen/OmniGen-v1/vae"),
            ("BAAI/OmniGen-v1", "model.safetensors", "models/OmniGen/OmniGen-v1"),
            ("BAAI/OmniGen-v1", "config.json", "models/OmniGen/OmniGen-v1"),
            ("BAAI/OmniGen-v1", "special_tokens_map.json", "models/OmniGen/OmniGen-v1"),
            ("BAAI/OmniGen-v1", "tokenizer_config.json", "models/OmniGen/OmniGen-v1"),
            ("BAAI/OmniGen-v1", "tokenizer.json", "models/OmniGen/OmniGen-v1"),
        ],
        "load_path": [
            "models/OmniGen/OmniGen-v1/vae/diffusion_pytorch_model.safetensors",
            "models/OmniGen/OmniGen-v1/model.safetensors",
        ]
    },
    # CogVideo
    "CogVideoX-5B": {
        "file_list": [
            ("ZhipuAI/CogVideoX-5b", "text_encoder/config.json", "models/CogVideo/CogVideoX-5b/text_encoder"),
            ("ZhipuAI/CogVideoX-5b", "text_encoder/model.safetensors.index.json", "models/CogVideo/CogVideoX-5b/text_encoder"),
            ("ZhipuAI/CogVideoX-5b", "text_encoder/model-00001-of-00002.safetensors", "models/CogVideo/CogVideoX-5b/text_encoder"),
            ("ZhipuAI/CogVideoX-5b", "text_encoder/model-00002-of-00002.safetensors", "models/CogVideo/CogVideoX-5b/text_encoder"),
            ("ZhipuAI/CogVideoX-5b", "transformer/config.json", "models/CogVideo/CogVideoX-5b/transformer"),
            ("ZhipuAI/CogVideoX-5b", "transformer/diffusion_pytorch_model.safetensors.index.json", "models/CogVideo/CogVideoX-5b/transformer"),
            ("ZhipuAI/CogVideoX-5b", "transformer/diffusion_pytorch_model-00001-of-00002.safetensors", "models/CogVideo/CogVideoX-5b/transformer"),
            ("ZhipuAI/CogVideoX-5b", "transformer/diffusion_pytorch_model-00002-of-00002.safetensors", "models/CogVideo/CogVideoX-5b/transformer"),
            ("ZhipuAI/CogVideoX-5b", "vae/diffusion_pytorch_model.safetensors", "models/CogVideo/CogVideoX-5b/vae"),
        ],
        "load_path": [
            "models/CogVideo/CogVideoX-5b/text_encoder",
            "models/CogVideo/CogVideoX-5b/transformer",
            "models/CogVideo/CogVideoX-5b/vae/diffusion_pytorch_model.safetensors",
        ],
    },
    # Stable Diffusion 3.5
    "StableDiffusion3.5-large": [
        ("AI-ModelScope/stable-diffusion-3.5-large", "sd3.5_large.safetensors", "models/stable_diffusion_3"),
        ("AI-ModelScope/stable-diffusion-3.5-large", "text_encoders/clip_l.safetensors", "models/stable_diffusion_3/text_encoders"),
        ("AI-ModelScope/stable-diffusion-3.5-large", "text_encoders/clip_g.safetensors", "models/stable_diffusion_3/text_encoders"),
        ("AI-ModelScope/stable-diffusion-3.5-large", "text_encoders/t5xxl_fp16.safetensors", "models/stable_diffusion_3/text_encoders"),
    ],
    "StableDiffusion3.5-medium": [
        ("AI-ModelScope/stable-diffusion-3.5-medium", "sd3.5_medium.safetensors", "models/stable_diffusion_3"),
        ("AI-ModelScope/stable-diffusion-3.5-large", "text_encoders/clip_l.safetensors", "models/stable_diffusion_3/text_encoders"),
        ("AI-ModelScope/stable-diffusion-3.5-large", "text_encoders/clip_g.safetensors", "models/stable_diffusion_3/text_encoders"),
        ("AI-ModelScope/stable-diffusion-3.5-large", "text_encoders/t5xxl_fp16.safetensors", "models/stable_diffusion_3/text_encoders"),
    ],
    "StableDiffusion3.5-large-turbo": [
        ("AI-ModelScope/stable-diffusion-3.5-large-turbo", "sd3.5_large_turbo.safetensors", "models/stable_diffusion_3"),
        ("AI-ModelScope/stable-diffusion-3.5-large", "text_encoders/clip_l.safetensors", "models/stable_diffusion_3/text_encoders"),
        ("AI-ModelScope/stable-diffusion-3.5-large", "text_encoders/clip_g.safetensors", "models/stable_diffusion_3/text_encoders"),
        ("AI-ModelScope/stable-diffusion-3.5-large", "text_encoders/t5xxl_fp16.safetensors", "models/stable_diffusion_3/text_encoders"),
    ],
    "HunyuanVideo":{
        "file_list": [
            ("AI-ModelScope/clip-vit-large-patch14", "model.safetensors", "models/HunyuanVideo/text_encoder"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model-00001-of-00004.safetensors", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model-00002-of-00004.safetensors", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model-00003-of-00004.safetensors", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model-00004-of-00004.safetensors", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "config.json", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model.safetensors.index.json", "models/HunyuanVideo/text_encoder_2"),
            ("AI-ModelScope/HunyuanVideo", "hunyuan-video-t2v-720p/vae/pytorch_model.pt", "models/HunyuanVideo/vae"),
            ("AI-ModelScope/HunyuanVideo", "hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt", "models/HunyuanVideo/transformers")
        ],
        "load_path": [
            "models/HunyuanVideo/text_encoder/model.safetensors",
            "models/HunyuanVideo/text_encoder_2",
            "models/HunyuanVideo/vae/pytorch_model.pt",
            "models/HunyuanVideo/transformers/mp_rank_00_model_states.pt"
        ],
    },
    "HunyuanVideoI2V":{
        "file_list": [
            ("AI-ModelScope/clip-vit-large-patch14", "model.safetensors", "models/HunyuanVideoI2V/text_encoder"),
            ("AI-ModelScope/llava-llama-3-8b-v1_1-transformers", "model-00001-of-00004.safetensors", "models/HunyuanVideoI2V/text_encoder_2"),
            ("AI-ModelScope/llava-llama-3-8b-v1_1-transformers", "model-00002-of-00004.safetensors", "models/HunyuanVideoI2V/text_encoder_2"),
            ("AI-ModelScope/llava-llama-3-8b-v1_1-transformers", "model-00003-of-00004.safetensors", "models/HunyuanVideoI2V/text_encoder_2"),
            ("AI-ModelScope/llava-llama-3-8b-v1_1-transformers", "model-00004-of-00004.safetensors", "models/HunyuanVideoI2V/text_encoder_2"),
            ("AI-ModelScope/llava-llama-3-8b-v1_1-transformers", "config.json", "models/HunyuanVideoI2V/text_encoder_2"),
            ("AI-ModelScope/llava-llama-3-8b-v1_1-transformers", "model.safetensors.index.json", "models/HunyuanVideoI2V/text_encoder_2"),
            ("AI-ModelScope/HunyuanVideo-I2V", "hunyuan-video-i2v-720p/vae/pytorch_model.pt", "models/HunyuanVideoI2V/vae"),
            ("AI-ModelScope/HunyuanVideo-I2V", "hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt", "models/HunyuanVideoI2V/transformers")
        ],
        "load_path": [
            "models/HunyuanVideoI2V/text_encoder/model.safetensors",
            "models/HunyuanVideoI2V/text_encoder_2",
            "models/HunyuanVideoI2V/vae/pytorch_model.pt",
            "models/HunyuanVideoI2V/transformers/mp_rank_00_model_states.pt"
        ],
    },
    "HunyuanVideo-fp8":{
        "file_list": [
            ("AI-ModelScope/clip-vit-large-patch14", "model.safetensors", "models/HunyuanVideo/text_encoder"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model-00001-of-00004.safetensors", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model-00002-of-00004.safetensors", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model-00003-of-00004.safetensors", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model-00004-of-00004.safetensors", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "config.json", "models/HunyuanVideo/text_encoder_2"),
            ("DiffSynth-Studio/HunyuanVideo_MLLM_text_encoder", "model.safetensors.index.json", "models/HunyuanVideo/text_encoder_2"),
            ("AI-ModelScope/HunyuanVideo", "hunyuan-video-t2v-720p/vae/pytorch_model.pt", "models/HunyuanVideo/vae"),
            ("DiffSynth-Studio/HunyuanVideo-safetensors", "model.fp8.safetensors", "models/HunyuanVideo/transformers")
        ],
        "load_path": [
            "models/HunyuanVideo/text_encoder/model.safetensors",
            "models/HunyuanVideo/text_encoder_2",
            "models/HunyuanVideo/vae/pytorch_model.pt",
            "models/HunyuanVideo/transformers/model.fp8.safetensors"
        ],
    },
}
Preset_model_id: TypeAlias = Literal[
    "HunyuanDiT",
    "stable-video-diffusion-img2vid-xt",
    "ExVideo-SVD-128f-v1",
    "ExVideo-CogVideoX-LoRA-129f-v1",
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
    "FLUX.1-schnell",
    "InstantX/FLUX.1-dev-Controlnet-Union-alpha",
    "jasperai/Flux.1-dev-Controlnet-Depth",
    "jasperai/Flux.1-dev-Controlnet-Surface-Normals",
    "jasperai/Flux.1-dev-Controlnet-Upscaler",
    "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha",
    "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta",
    "Shakker-Labs/FLUX.1-dev-ControlNet-Depth",
    "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
    "InstantX/FLUX.1-dev-IP-Adapter",
    "InfiniteYou",
    "SDXL_lora_zyd232_ChineseInkStyle_SDXL_v1_0",
    "QwenPrompt",
    "OmostPrompt",
    "ESRGAN_x4",
    "RIFE",
    "OmniGen-v1",
    "CogVideoX-5B",
    "Annotators:Depth",
    "Annotators:Softedge",
    "Annotators:Lineart",
    "Annotators:Normal",
    "Annotators:Openpose",
    "StableDiffusion3.5-large",
    "StableDiffusion3.5-medium",
    "HunyuanVideo",
    "HunyuanVideo-fp8",
    "HunyuanVideoI2V",
]
