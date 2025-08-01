import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-Krea-dev", origin_file_pattern="flux1-krea-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
    ],
)
pipe.load_lora(pipe.dit, "models/train/FLUX.1-Kera-dev_lora/epoch-4.safetensors", alpha=1)

image = pipe(prompt="a dog", seed=0)
image.save("image_FLUX.1-Kera-dev_lora.jpg")
