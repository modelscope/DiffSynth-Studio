import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from diffsynth import load_state_dict
from PIL import Image


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-Redux-dev", origin_file_pattern="image_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-Redux-dev", origin_file_pattern="image_embedder/diffusion_pytorch_model.safetensors"),
    ],
)
state_dict = load_state_dict("models/train/FLUX.1-Redux-dev_full/epoch-0.safetensors")
pipe.dit.load_state_dict(state_dict)

image = pipe(
    flux_redux_image=Image.open("data/diffsynth_example_dataset/flux/FLUX.1-Redux-dev/robot.png").convert("RGB"),
    embedded_guidance=2.5, num_inference_steps=50, seed=0,
)
image.save("image_FLUX.1-Redux-dev_full.jpg")
