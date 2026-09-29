"""
Z-Image-Turbo — EntroPack pre-quantized inference example.

Downloads the pre-quantized packages and their json sidecars from the
`DiffSynth-Studio/EntroPackPreQuants` model repo (each json is a ready-to-use
`MODEL_CONFIGS` entry), then runs inference.

Available packages (paths inside the prequant model repo):

    dit:
        models/Z-Image-Turbo/dit_4bpp.safetensors             4bpp
        models/Z-Image-Turbo/dit_5bpp.safetensors             5bpp
        models/Z-Image-Turbo/dit_6bpp.safetensors             6bpp
        models/Z-Image-Turbo/dit_7bpp.safetensors             7bpp
        models/Z-Image-Turbo/dit_8bpp.safetensors             8bpp
        models/Z-Image-Turbo/dit_extreme_2.3bpp.safetensors   mixed allocation, 2.3bpp
        models/Z-Image-Turbo/dit_fp8_5bpp.safetensors         fp8 weights, 5bpp
    text_encoder:
        models/Z-Image-Turbo/text_encoder_4bpp.safetensors    4bpp
        models/Z-Image-Turbo/text_encoder_5bpp.safetensors    5bpp
        models/Z-Image-Turbo/text_encoder_6bpp.safetensors    6bpp
        models/Z-Image-Turbo/text_encoder_7bpp.safetensors    7bpp
        models/Z-Image-Turbo/text_encoder_8bpp.safetensors    8bpp

`dit_package` and `text_encoder_package` are chosen independently: any combination works
(e.g. `dit_8bpp` with `text_encoder_4bpp`).
"""
import json

import torch

from diffsynth.configs import MODEL_CONFIGS
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig

prequant_model_id = "DiffSynth-Studio/EntroPackPreQuants"
dit_package = "models/Z-Image-Turbo/dit_4bpp.safetensors"
text_encoder_package = "models/Z-Image-Turbo/text_encoder_4bpp.safetensors"


def register_prequantized(package_pattern):
    """Download the pre-quantized package and its json sidecar (a ready-to-use MODEL_CONFIGS entry), return a loadable ModelConfig."""
    package = ModelConfig(model_id=prequant_model_id, origin_file_pattern=package_pattern)
    package.download_if_necessary()
    sidecar = ModelConfig(model_id=prequant_model_id, origin_file_pattern=package_pattern.replace(".safetensors", ".json"))
    sidecar.download_if_necessary()
    with open(sidecar.path) as handle:
        MODEL_CONFIGS.append(json.load(handle))
    return package


pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        register_prequantized(dit_package),
        register_prequantized(text_encoder_package),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
)
prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
image = pipe(prompt=prompt, seed=42, rand_device="cuda")
image.save("image_Z-Image-Turbo-EntroPack.jpg")
