import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig, ControlNetInput


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ModelConfig(model_id="jasperai/Flux.1-dev-Controlnet-Upscaler", origin_file_pattern="diffusion_pytorch_model.safetensors"),
    ],
)

image_1 = pipe(
    prompt="a photo of a cat, highly detailed",
    height=768, width=768,
    seed=0, rand_device="cuda",
)
image_1.save("image_1.jpg")

image_1 = image_1.resize((2048, 2048))
image_2 = pipe(
    prompt="a photo of a cat, highly detailed",
    controlnet_inputs=[ControlNetInput(image=image_1, scale=0.7)],
    input_image=image_1,
    denoising_strength=0.99,
    height=2048, width=2048, tiled=True,
    seed=1, rand_device="cuda",
)
image_2.save("image_2.jpg")