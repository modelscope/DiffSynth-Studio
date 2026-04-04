import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from PIL import Image
import numpy as np


vram_config = {
    "offload_dtype": torch.float8_e4m3fn,
    "offload_device": "cpu",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen2.5-VL-7B-Instruct", origin_file_pattern="model-*.safetensors", **vram_config),
        ModelConfig(model_id="stepfun-ai/Step1X-Edit", origin_file_pattern="step1x-edit-i1258.safetensors", **vram_config),
        ModelConfig(model_id="stepfun-ai/Step1X-Edit", origin_file_pattern="vae.safetensors", **vram_config),
    ],
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

image = Image.fromarray(np.zeros((1248, 832, 3), dtype=np.uint8) + 255)
image = pipe(
    prompt="draw red flowers in Chinese ink painting style",
    step1x_reference_image=image,
    width=832, height=1248, cfg_scale=6,
    seed=1, rand_device='cuda'
)
image.save("image_1.jpg")

image = pipe(
    prompt="add more flowers in Chinese ink painting style",
    step1x_reference_image=image,
    width=832, height=1248, cfg_scale=6,
    seed=2, rand_device='cuda'
)
image.save("image_2.jpg")
