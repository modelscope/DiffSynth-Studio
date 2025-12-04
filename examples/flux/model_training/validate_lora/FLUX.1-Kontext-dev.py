import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from PIL import Image


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-Kontext-dev", origin_file_pattern="flux1-kontext-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
    ],
)
pipe.load_lora(pipe.dit, "models/train/FLUX.1-Kontext-dev_lora/epoch-4.safetensors", alpha=1)

image = pipe(
    prompt="Make the dog turn its head around.",
    kontext_images=Image.open("data/example_image_dataset/2.jpg").resize((768, 768)),
    height=768, width=768,
    seed=0
)
image.save("image_FLUX.1-Kontext-dev_lora.jpg")
