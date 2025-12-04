import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig


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
        ModelConfig(model_id="InstantX/FLUX.1-dev-IP-Adapter", origin_file_pattern="ip-adapter.bin", **vram_config),
        ModelConfig(model_id="google/siglip-so400m-patch14-384", origin_file_pattern="model.safetensors", **vram_config),
    ],
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

origin_prompt = "a rabbit in a garden, colorful flowers"
image = pipe(prompt=origin_prompt, height=1280, width=960, seed=42)
image.save("style image.jpg")

image = pipe(prompt="A piggy", height=1280, width=960, seed=42,
    ipadapter_images=[image], ipadapter_scale=0.7)
image.save("A piggy.jpg")
