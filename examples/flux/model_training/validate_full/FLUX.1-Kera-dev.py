import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from diffsynth import load_state_dict


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
state_dict = load_state_dict("models/train/FLUX.1-Kera-dev_full/epoch-0.safetensors")
pipe.dit.load_state_dict(state_dict)

image = pipe(prompt="a dog", seed=0)
image.save("image_FLUX.1-Kera-dev_full.jpg")
