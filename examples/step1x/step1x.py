import torch
from diffsynth import FluxImagePipeline, ModelManager
from modelscope import snapshot_download
from PIL import Image
import numpy as np


snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct", cache_dir="./models")
snapshot_download("stepfun-ai/Step1X-Edit", cache_dir="./models")

model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda")
model_manager.load_models([
    "models/Qwen/Qwen2.5-VL-7B-Instruct",
    "models/stepfun-ai/Step1X-Edit/step1x-edit-i1258.safetensors",
    "models/stepfun-ai/Step1X-Edit/vae.safetensors",
])
pipe = FluxImagePipeline.from_model_manager(model_manager)
pipe.enable_vram_management()

image = Image.fromarray(np.zeros((1248, 832, 3), dtype=np.uint8) + 255)
image = pipe(
    prompt="draw red flowers in Chinese ink painting style",
    step1x_reference_image=image,
    width=832, height=1248, cfg_scale=6,
    seed=1,
)
image.save("image_1.jpg")

image = pipe(
    prompt="add more flowers in Chinese ink painting style",
    step1x_reference_image=image,
    width=832, height=1248, cfg_scale=6,
    seed=2,
)
image.save("image_2.jpg")
