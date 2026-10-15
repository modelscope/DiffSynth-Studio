"""
MiniMax-H3 — EntroPack pre-quantized FL2VA and Ref2VA inference example
(low VRAM, disk offload).

Downloads the pre-quantized packages and their json sidecars from the
`DiffSynth-Studio/EntroPackPreQuants` model repo (each json is a ready-to-use
`MODEL_CONFIGS` entry), and the Ref2VA reference image from the same repo's `assets/`.

Available packages (paths inside the prequant model repo):

    FL2VA dit:
        models/MiniMax-H3/FL2VA/dit_4bpp.safetensors             4bpp
        models/MiniMax-H3/FL2VA/dit_5bpp.safetensors             5bpp
        models/MiniMax-H3/FL2VA/dit_6bpp.safetensors             6bpp
        models/MiniMax-H3/FL2VA/dit_7bpp.safetensors             7bpp
        models/MiniMax-H3/FL2VA/dit_8bpp.safetensors             8bpp
        models/MiniMax-H3/FL2VA/dit_extreme_3.0bpp.safetensors   mixed allocation, 3.0bpp
        models/MiniMax-H3/FL2VA/dit_fp8_5bpp.safetensors         fp8 weights, 5bpp
    Ref2VA dit:
        models/MiniMax-H3/REF2VA/dit_4bpp.safetensors            4bpp
        models/MiniMax-H3/REF2VA/dit_5bpp.safetensors            5bpp
        models/MiniMax-H3/REF2VA/dit_6bpp.safetensors            6bpp
        models/MiniMax-H3/REF2VA/dit_7bpp.safetensors            7bpp
        models/MiniMax-H3/REF2VA/dit_8bpp.safetensors            8bpp
        models/MiniMax-H3/REF2VA/dit_extreme_3.0bpp.safetensors  mixed allocation, 3.0bpp
        models/MiniMax-H3/REF2VA/dit_fp8_5bpp.safetensors        fp8 weights, 5bpp
    text_encoder:
        models/MiniMax-H3/text_encoder_4bpp.safetensors    4bpp
        models/MiniMax-H3/text_encoder_5bpp.safetensors    5bpp
        models/MiniMax-H3/text_encoder_6bpp.safetensors    6bpp
        models/MiniMax-H3/text_encoder_7bpp.safetensors    7bpp
        models/MiniMax-H3/text_encoder_8bpp.safetensors    8bpp
    video_vae:
        models/MiniMax-H3/video_vae_4bpp.safetensors       4bpp
        models/MiniMax-H3/video_vae_5bpp.safetensors       5bpp
        models/MiniMax-H3/video_vae_6bpp.safetensors       6bpp
        models/MiniMax-H3/video_vae_7bpp.safetensors       7bpp
        models/MiniMax-H3/video_vae_8bpp.safetensors       8bpp
    reference image:
        assets/minimax_h3_ref2va_reference_woman.png

The dit / video_vae packages are streamed from disk; the text_encoder keeps CPU offload
because the reference image path runs its vision tower, whose quantized layers cannot be
restored from disk while wrapped as one container module (see
`notes/diffsynth_disk_offload_quantized_container.md` in the EntroPack workspace).
"""
import json

import torch
from modelscope import snapshot_download
from PIL import Image

from diffsynth.configs import MODEL_CONFIGS
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio

prequant_model_id = "DiffSynth-Studio/EntroPackPreQuants"
prequant_root = f"./models/{prequant_model_id}"
text_encoder_package = "models/MiniMax-H3/text_encoder_4bpp.safetensors"
video_vae_package = "models/MiniMax-H3/video_vae_4bpp.safetensors"
fl2va_dit_package = "models/MiniMax-H3/FL2VA/dit_4bpp.safetensors"
ref2va_dit_package = "models/MiniMax-H3/REF2VA/dit_4bpp.safetensors"
reference_package = "assets/minimax_h3_ref2va_reference_woman.png"

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
        MODEL_CONFIGS.append(json.load(handle))
    return package


def build_pipe(dit_package):
    return MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            register_prequantized(text_encoder_package),
            register_prequantized(dit_package),
            register_prequantized(video_vae_package),
            ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
        ],
        processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
    )


prompt = "A girl is very happy, she is speaking in english: “I enjoy working with Diffsynth-Studio, it's a perfect framework.”"

# Text -> Video + Audio
pipe = build_pipe(fl2va_dit_package)
video, audio = pipe(prompt=prompt, height=1344, width=768, num_frames=124, num_inference_steps=20, seed=0)
write_video_audio(video=video, audio=audio, output_path="MiniMax-H3-FL2VA-EntroPack.mp4", fps=24, audio_sample_rate=32000)
del pipe
torch.cuda.empty_cache()

# Text + Reference Image -> Video + Audio
snapshot_download(prequant_model_id, allow_file_pattern=reference_package, local_dir=prequant_root)
reference = Image.open(f"{prequant_root}/{reference_package}").convert("RGB")
pipe = build_pipe(ref2va_dit_package)
video, audio = pipe(
    prompt=prompt,
    height=1344, width=768, num_frames=124, num_inference_steps=20, seed=42,
    references=[{"type": "image", "image": reference}],
)
write_video_audio(video=video, audio=audio, output_path="MiniMax-H3-Ref2VA-EntroPack.mp4", fps=24, audio_sample_rate=32000)
