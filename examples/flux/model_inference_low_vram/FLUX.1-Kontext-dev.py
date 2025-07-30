import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from PIL import Image


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-Kontext-dev", origin_file_pattern="flux1-kontext-dev.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
    ],
)
pipe.enable_vram_management()

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