"""
Qwen-Image-2.1 — EntroPack pre-quantized inference example.

Downloads the pre-quantized packages and their json sidecars from the
`DiffSynth-Studio/EntroPackPreQuants` model repo (each json is a ready-to-use
`MODEL_CONFIGS` entry), then runs inference.

Available packages (paths inside the prequant model repo):

    dit:
        models/Qwen-Image-2.1/dit_4bpp.safetensors             4bpp
        models/Qwen-Image-2.1/dit_5bpp.safetensors             5bpp
        models/Qwen-Image-2.1/dit_6bpp.safetensors             6bpp
        models/Qwen-Image-2.1/dit_7bpp.safetensors             7bpp
        models/Qwen-Image-2.1/dit_8bpp.safetensors             8bpp
        models/Qwen-Image-2.1/dit_extreme_3.0bpp.safetensors   mixed allocation, 3.0bpp
        models/Qwen-Image-2.1/dit_fp8_5bpp.safetensors         fp8 weights, 5bpp
    text_encoder:
        models/Qwen-Image-2.1/text_encoder_4bpp.safetensors    4bpp
        models/Qwen-Image-2.1/text_encoder_5bpp.safetensors    5bpp
        models/Qwen-Image-2.1/text_encoder_6bpp.safetensors    6bpp
        models/Qwen-Image-2.1/text_encoder_7bpp.safetensors    7bpp
        models/Qwen-Image-2.1/text_encoder_8bpp.safetensors    8bpp

`dit_package` and `text_encoder_package` are chosen independently: any combination works
(e.g. `dit_8bpp` with `text_encoder_4bpp`).
"""
import json

import torch
from PIL import Image

from diffsynth.configs import MODEL_CONFIGS
from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline, ModelConfig

prequant_model_id = "DiffSynth-Studio/EntroPackPreQuants"
dit_package = "models/Qwen-Image-2.1/dit_4bpp.safetensors"
text_encoder_package = "models/Qwen-Image-2.1/text_encoder_4bpp.safetensors"


def register_prequantized(package_pattern):
    """Download the pre-quantized package and its json sidecar (a ready-to-use MODEL_CONFIGS entry), return a loadable ModelConfig."""
    package = ModelConfig(model_id=prequant_model_id, origin_file_pattern=package_pattern)
    package.download_if_necessary()
    sidecar = ModelConfig(model_id=prequant_model_id, origin_file_pattern=package_pattern.replace(".safetensors", ".json"))
    sidecar.download_if_necessary()
    with open(sidecar.path) as handle:
        MODEL_CONFIGS.append(json.load(handle))
    return package


pipe = QwenImage21Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        register_prequantized(dit_package),
        register_prequantized(text_encoder_package),
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="vae/diffusion_pytorch_model*.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="processor/"),
)
# Text-to-Image, the output is an RGBA image
prompt = "Flat anime-style illustration, a girl with long black hair, wearing a JK uniform."
image = pipe(prompt, seed=0)
image.save("image1.png")

prompt = "Flat anime-style illustration, a sunny and cheerful high school girl."
image_2 = pipe(prompt=prompt, seed=0)
image_2.save("image2.png")

# Image Editing, the generated RGBA image is fed back as the condition
prompt = "Generate a group photo of these two characters."
edit_image = [Image.open("image1.png"), Image.open("image2.png")]
image_3 = pipe(prompt, edit_image=edit_image, seed=1)
image_3.save("image3.png")
