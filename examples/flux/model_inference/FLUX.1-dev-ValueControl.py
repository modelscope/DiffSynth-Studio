import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
    ],
)

from diffsynth.models.flux_value_control import SingleValueEncoder, MultiValueEncoder
pipe.value_controller = MultiValueEncoder(encoders=[SingleValueEncoder(), SingleValueEncoder(), SingleValueEncoder(), SingleValueEncoder()]).to(dtype=torch.bfloat16, device="cuda")

image = pipe(prompt="a cat", seed=0, value_controller_inputs=[0.5, 0.5, 1, 0])
image.save("flux.jpg")
