import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from modelscope import snapshot_download
from PIL import Image
import numpy as np


snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir="./models")
snapshot_download("stepfun-ai/Step1X-Edit", cache_dir="./models")

pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path="models/Qwen/Qwen2.5-VL-7B-Instruct"),
        ModelConfig(model_id="stepfun-ai/Step1X-Edit", origin_file_pattern="step1x-edit-i1258.safetensors"),
        ModelConfig(model_id="stepfun-ai/Step1X-Edit", origin_file_pattern="vae.safetensors"),
    ],
)


pipe.enable_vram_management()

image = Image.fromarray(np.zeros((1248, 832, 3), dtype=np.uint8) + 255)
image = pipe(
    prompt="draw red flowers in Chinese ink painting style",
    step1x_reference_image=image,
    width=832, height=1248, cfg_scale=6,
    seed=1,
    rand_device='cuda'
)
image.save("image_1.jpg")



image = pipe(
    prompt="add more flowers in Chinese ink painting style",
    step1x_reference_image=image,
    width=832, height=1248, cfg_scale=6,
    seed=2,
    rand_device='cuda'
)
image.save("image_2.jpg")

