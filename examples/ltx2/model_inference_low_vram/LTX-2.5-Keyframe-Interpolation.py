import os

import torch
from PIL import Image

from diffsynth.core import ModelConfig
from diffsynth.pipelines.ltx25_audio_video import LTX25AudioVideoPipeline
from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2

MODEL_ROOT = "models/Lightricks/LTX-2.5"
START_IMAGE = "start.png"
MIDDLE_IMAGE = "middle.png"
END_IMAGE = "end.png"
GEMMA = f"{MODEL_ROOT}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
TRANSFORMER = f"{MODEL_ROOT}/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
STAGE2_LORA = f"{MODEL_ROOT}/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
VRAM_LIMIT_GB = float(os.environ.get("LTX25_VRAM_LIMIT_GB", "16"))

vram_config = {
    "offload_dtype": torch.float8_e5m2,
    "offload_device": "cpu",
    "onload_dtype": torch.float8_e5m2,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e5m2,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = LTX25AudioVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=GEMMA, **vram_config),
        ModelConfig(path=[GEMMA, TRANSFORMER], **vram_config),
        ModelConfig(path=TRANSFORMER, **vram_config),
        ModelConfig(path=f"{MODEL_ROOT}/vae/ltx-2.5-video-vae-bf16.safetensors", **vram_config),
        ModelConfig(path=f"{MODEL_ROOT}/vae/ltx-2.5-audio-vae-bf16.safetensors", **vram_config),
        ModelConfig(
            path=f"{MODEL_ROOT}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
            **vram_config,
        ),
    ],
    gemma_path=GEMMA,
    stage2_lora_config=ModelConfig(path=STAGE2_LORA),
    vram_limit=VRAM_LIMIT_GB,
)

video, audio = pipe(
    prompt="A colorful sailboat crosses a calm lake at sunrise. Gentle water sounds and distant birds.",
    seed=42,
    height=576,
    width=960,
    num_frames=121,
    frame_rate=24,
    input_images=[Image.open(path).convert("RGB") for path in (START_IMAGE, MIDDLE_IMAGE, END_IMAGE)],
    input_images_indexes=[0, 60, 120],
    tiled=True,
    use_distilled_pipeline=False,
    use_two_stage_pipeline=True,
    cfg_scale=3.0,
    num_inference_steps=8,
)
write_video_audio_ltx2(
    video=video,
    audio=audio,
    output_path="ltx2.5_keyframe_interpolation.mp4",
    fps=24,
    audio_sample_rate=pipe.audio_vocoder.output_sampling_rate,
)
