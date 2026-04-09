import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig, ControlNetInput
import numpy as np
from PIL import Image


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
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors", **vram_config),
        ModelConfig(model_id="alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", origin_file_pattern="diffusion_pytorch_model.safetensors", **vram_config),
    ],
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

image_1 = pipe(
    prompt="a cat sitting on a chair",
    height=1024, width=1024,
    seed=8, rand_device="cuda",
)
image_1.save("image_1.jpg")

mask = np.zeros((1024, 1024, 3), dtype=np.uint8)
mask[100:350, 350: -300] = 255
mask = Image.fromarray(mask)
mask.save("mask.jpg")

image_2 = pipe(
    prompt="a cat sitting on a chair, wearing sunglasses",
    controlnet_inputs=[ControlNetInput(image=image_1, inpaint_mask=mask, scale=0.9)],
    height=1024, width=1024,
    seed=9, rand_device="cuda",
)
image_2.save("image_2.jpg")