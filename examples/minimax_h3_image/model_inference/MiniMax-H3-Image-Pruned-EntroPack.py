"""
MiniMax-H3-Image (AdaLN-pruned) — EntroPack pre-quantized inference example.

Downloads the pre-quantized packages and their json sidecars (each json is a ready-to-use
`MODEL_CONFIGS` entry), then runs the distilled six-step pipeline. The DiT packages come from
`DiffSynth-Studio/MiniMax-H3-Image-EntroPack`, the shared text encoder from
`DiffSynth-Studio/EntroPackPreQuants`.

Available packages:

    dit:
        models/dit_8bit.safetensors           8 bit, sensitive layers held at 8.5 bpp
        models/dit_7bit.safetensors           7 bit, sensitive layers held at 8.5 bpp
        models/dit_6bit.safetensors           6 bit, sensitive layers held at 8.5 bpp
        models/dit_4p5bit.safetensors         4.5 bit, sensitive layers held at 8.5 bpp
        models/dit_extreme_3bpp.safetensors   extreme quantization, mixed allocation (sensitive layers 8.5 / 5.0 bpp)
        models/dit_8bit_int8.safetensors      int8 gemm, raw int8 weights (no lattice re-coding)
        models/dit_7bit_int8.safetensors      int8 gemm, 7 bit
        models/dit_6bit_int8.safetensors      int8 gemm, 6 bit
        models/dit_4p5bit_int8.safetensors    int8 gemm, 4.5 bit
    text encoder (shared by every MiniMax-H3 model):
        models/MiniMax-H3/text_encoder_4bpp.safetensors  4 bit

`dit_package` and `text_encoder_package` are chosen independently: any combination works.
The int8 packages need Triton and a CUDA GPU of SM 8.0+.
"""
import json

import torch

from diffsynth.configs import MODEL_CONFIGS
from diffsynth.pipelines.minimax_h3_image import MiniMaxH3ImagePipeline, ModelConfig
from PIL import Image

dit_model_id = "DiffSynth-Studio/MiniMax-H3-Image-EntroPack"
text_encoder_model_id = "DiffSynth-Studio/EntroPackPreQuants"
dit_package = "models/dit_7bit_int8.safetensors"
text_encoder_package = "models/MiniMax-H3/text_encoder_4bpp.safetensors"


def register_prequantized(model_id, package_pattern):
    """Download the pre-quantized package and its json sidecar (a ready-to-use MODEL_CONFIGS entry), return a loadable ModelConfig."""
    package = ModelConfig(model_id=model_id, origin_file_pattern=package_pattern)
    package.download_if_necessary()
    sidecar = ModelConfig(model_id=model_id, origin_file_pattern=package_pattern.replace(".safetensors", ".json"))
    sidecar.download_if_necessary()
    with open(sidecar.path) as handle:
        entry = json.load(handle)
    if not any(config.get("model_hash") == entry["model_hash"] for config in MODEL_CONFIGS):
        MODEL_CONFIGS.append(entry)
    return package


pipe = MiniMaxH3ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        register_prequantized(dit_model_id, dit_package),
        register_prequantized(text_encoder_model_id, text_encoder_package),
        ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="pdd_fused_heads.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-Image-EntroPack", origin_file_pattern="models/vae_int8.safetensors"),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="processor/"),
)

prompt = (
    "An overhead photograph of a rustic farm-to-table spread on a weathered wooden table: a round loaf of sourdough "
    "on burlap, wedges of cheese on a dark board, a copper bowl of vegetables, vine tomatoes, olives and walnuts, "
    "warm afternoon light raking across the grain of the wood."
)
image = pipe(prompt=prompt, height=4096, width=4096, seed=42)
image.save("MiniMax-H3-Image-Pruned-EntroPack.jpg")

prompt = "Change the color of the tomato in <Picture 1> to green."
image = pipe(
    prompt=prompt, height=4096, width=4096, seed=43,
    edit_image=[Image.open("MiniMax-H3-Image-Pruned-EntroPack.jpg")],
)
image.save("MiniMax-H3-Image-Pruned-EntroPack-Edit.jpg")
