import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
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
        ModelConfig(model_id="black-forest-labs/FLUX.1-Kontext-dev", origin_file_pattern="flux1-kontext-dev.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors", **vram_config),
    ],
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

image_1 = pipe(
    prompt="a beautiful Asian long-haired female college student.",
    embedded_guidance=2.5,
    seed=1,
)
image_1.save("image_1.jpg")

image_2 = pipe(
    prompt="transform the style to anime style.",
    kontext_images=image_1,
    embedded_guidance=2.5,
    seed=2,
)
image_2.save("image_2.jpg")

image_3 = pipe(
    prompt="let her smile.",
    kontext_images=image_1,
    embedded_guidance=2.5,
    seed=3,
)
image_3.save("image_3.jpg")

image_4 = pipe(
    prompt="let the girl play basketball.",
    kontext_images=image_1,
    embedded_guidance=2.5,
    seed=4,
)
image_4.save("image_4.jpg")

image_5 = pipe(
    prompt="move the girl to a park, let her sit on a chair.",
    kontext_images=image_1,
    embedded_guidance=2.5,
    seed=5,
)
image_5.save("image_5.jpg")