import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="ostris/Flex.2-preview", origin_file_pattern="Flex.2-preview.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
    ],
)
pipe.load_lora(pipe.dit, "models/train/FLEX.2-preview_lora/epoch-4.safetensors", alpha=1)

image = pipe(prompt="dog,white and brown dog, sitting on wall, under pink flowers", seed=0)
image.save("image_FLEX.2-preview_lora.jpg")
