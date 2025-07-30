import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from diffsynth import load_state_dict


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="ostris/Flex.2-preview", origin_file_pattern="Flex.2-preview.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
    ],
)
state_dict = load_state_dict("models/train/FLEX.2-preview_full/epoch-0.safetensors")
pipe.dit.load_state_dict(state_dict)

image = pipe(prompt="dog,white and brown dog, sitting on wall, under pink flowers", seed=0)
image.save("image_FLEX.2-preview_full.jpg")
