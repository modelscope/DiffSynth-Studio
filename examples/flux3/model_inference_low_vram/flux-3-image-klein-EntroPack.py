"""
FLUX.3 Image Klein — EntroPack pre-quantized inference example (low VRAM, disk offload).

Downloads the pre-quantized packages and their json sidecars from the
`DiffSynth-Studio/flux-3-image-klein-entropack` model repo (each json is a
ready-to-use `MODEL_CONFIGS` entry), then runs inference with disk offload: weights
live on disk and are streamed in per module.

Available packages (paths inside the prequant model repo):

    dit:
        models/dit_3bit.safetensors           mixed allocation, 3bit
        models/dit_4bit.safetensors           4bit
        models/dit_6bit.safetensors           6bit
        models/dit_fp8_3bit.safetensors       fp8 gemm, mixed allocation, 3bit
        models/dit_fp8_4bit.safetensors       fp8 gemm, 4bit
        models/dit_fp8_6bit.safetensors       fp8 gemm, 6bit
    text_encoder:
        models/text_encoder_3bit.safetensors  3bit
        models/text_encoder_4bit.safetensors  4bit
        models/text_encoder_6bit.safetensors  6bit

`dit_package` and `text_encoder_package` are chosen independently: any combination works
(e.g. `dit_4bit` with `text_encoder_3bit`).
"""
import json

import torch

from diffsynth.configs import MODEL_CONFIGS
from diffsynth.pipelines.flux3_image import Flux3ImagePipeline, ModelConfig

prequant_model_id = "DiffSynth-Studio/flux-3-image-klein-entropack"
dit_package = "models/dit_4bit.safetensors"
text_encoder_package = "models/text_encoder_4bit.safetensors"

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
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


# This example also loads the VAE and text encoder from `black-forest-labs/flux-3-image-klein`,
# which is an access-request-only model. To download it:
# 1. Request access at https://modelscope.cn/models/black-forest-labs/flux-3-image-klein
# 2. Log in via CLI: `modelscope login --token ms-xxx`. Get your token at https://modelscope.cn/my/settings/token
pipe = Flux3ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        register_prequantized(dit_package),
        register_prequantized(text_encoder_package),
        ModelConfig(model_id="black-forest-labs/flux-3-image-klein", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="black-forest-labs/flux-3-image-klein", origin_file_pattern="text_encoder/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
prompt = "Full-color cinematic outdoor night photograph of a stunning young European woman in her early twenties with fair pale skin, wearing a soft blush pink off-shoulder dress with a fitted bodice, framed from the waist up against softly illuminated historic architecture. She stands near the centre of the frame, facing the camera with a gentle closed-lip smile, her hands resting loosely clasped in front of her waist with relaxed elbows; she has a slim oval face with softly defined cheekbones and a delicate narrow jaw, soft hazel eyes, natural-shaped brows, and subtle natural makeup. Her soft brown hair falls in loose natural waves over her shoulders, with fine strands framing her face. The fabric of the dress falls in soft folds across her shoulders and chest. In the frame-left background, the word \"EntroPack\" glows as a hand-drawn bubbly yellow neon sign mounted on a dark building above the street, thick rounded irregular letters riding a gently bouncing baseline with a bright core and a soft outer glow, kept sharp and unblurred against the night. The background recedes into deep blue-black tones: the Eiffel Tower rises above distant buildings on the frame-right as a softly silhouetted tapering lattice outline in the shallow depth of field, while small amber, muted turquoise and rose-coloured lights dissolve into rounded bokeh across the upper background, leaving a darker area behind her head. Soft white frontal illumination reveals her face with clear pale skin, a faint cooler rim separates her hair from the night, and the skirt of her dress fills the lower edge of the frame."
image = pipe(prompt=prompt, seed=0, num_inference_steps=16, cfg_scale=1.0)
image.save("image.jpg")
