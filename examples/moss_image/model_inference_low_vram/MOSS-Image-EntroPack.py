"""
MOSS-Image — EntroPack pre-quantized inference example (low VRAM).

Downloads the pre-quantized packages and their json sidecars from the
`DiffSynth-Studio/MOSS-Image-EntroPack` model repo (each json is a
ready-to-use `MODEL_CONFIGS` entry), then runs inference with VRAM management.
Defaults to SFT INT8 6bit; switch `dit_package` / `text_encoder_package` (and
`base_model_id` for Base) to use another level.

Available packages (paths inside the prequant model repo):

    dit:
        models/MOSS-Image-SFT/dit_3.5bit.safetensors        mixed allocation, 3.5bit
        models/MOSS-Image-SFT/dit_4.5bit.safetensors        mixed allocation, 4.5bit
        models/MOSS-Image-SFT/dit_6bit.safetensors          6bit
        models/MOSS-Image-SFT/dit_int8_6bit.safetensors     hybrid INT8 + 8.5bit sensitive, 6bit
        models/MOSS-Image-SFT/dit_int8_7bit.safetensors     hybrid INT8 + 8.5bit sensitive, 7bit
        models/MOSS-Image-Base/dit_4.5bit.safetensors       mixed allocation, 4.5bit
        models/MOSS-Image-Base/dit_6bit.safetensors         6bit
        models/MOSS-Image-Base/dit_int8_6bit.safetensors    hybrid INT8 + 8.5bit sensitive, 6bit
        models/MOSS-Image-Base/dit_int8_7bit.safetensors    hybrid INT8 + 8.5bit sensitive, 7bit
    text_encoder:
        models/text_encoder_4.5bit.safetensors              4.5bit
        models/text_encoder_6bit.safetensors                6bit
        models/text_encoder_int8_6bit.safetensors           hybrid INT8 + 8.5bit sensitive, 6bit
        models/text_encoder_int8_7bit.safetensors           hybrid INT8 + 8.5bit sensitive, 7bit

The text encoder packages are shared between Base and SFT.
`dit_package` and `text_encoder_package` are chosen independently: any combination works
(e.g. `dit_int8_6bit` with `text_encoder_4.5bit`).
"""
import json

import torch

from diffsynth.configs import MODEL_CONFIGS
from diffsynth.pipelines.moss_image import MOSSImagePipeline, ModelConfig

prequant_model_id = "DiffSynth-Studio/MOSS-Image-EntroPack"
base_model_id = "openmoss/MOSS-Image-SFT"
dit_package = "models/MOSS-Image-SFT/dit_int8_7bit.safetensors"
text_encoder_package = "models/text_encoder_int8_7bit.safetensors"

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}


def register_prequantized(package_pattern):
    """Download the pre-quantized package and its json sidecar (a ready-to-use MODEL_CONFIGS entry), return a loadable ModelConfig."""
    package = ModelConfig(model_id=prequant_model_id, origin_file_pattern=package_pattern, **vram_config)
    package.download_if_necessary()
    sidecar = ModelConfig(model_id=prequant_model_id, origin_file_pattern=package_pattern.replace(".safetensors", ".json"))
    sidecar.download_if_necessary()
    with open(sidecar.path) as handle:
        entry = json.load(handle)
    if not any(config.get("model_hash") == entry["model_hash"] for config in MODEL_CONFIGS):
        MODEL_CONFIGS.append(entry)
    return package


pipe = MOSSImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        register_prequantized(dit_package),
        register_prequantized(text_encoder_package),
        ModelConfig(model_id=base_model_id, origin_file_pattern="vae/diffusion_pytorch_model.safetensors", computation_dtype=torch.float32),
    ],
    tokenizer_config=ModelConfig(model_id=base_model_id, origin_file_pattern="tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

# Optional: torch.compile acceleration.
pipe.compile_pipeline()

prompt = '''A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small sticky note on the refrigerator door is handwritten in Chinese:
"记得买牛奶和鸡蛋！
牛奶两盒，鸡蛋一打。
顺便带一袋面包。"'''

image = pipe(prompt, height=1024, width=1024, seed=42, num_inference_steps=50, cfg_scale=3.0)
image.save("image_MOSS-Image-EntroPack.jpg")
