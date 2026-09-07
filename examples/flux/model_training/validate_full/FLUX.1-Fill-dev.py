import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from diffsynth import load_state_dict
from PIL import Image


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-Fill-dev", origin_file_pattern="flux1-fill-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-Fill-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-Fill-dev", origin_file_pattern="text_encoder_2/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-Fill-dev", origin_file_pattern="ae.safetensors"),
    ],
)
state_dict = load_state_dict("models/train/FLUX.1-Fill-dev_full/epoch-0.safetensors")
pipe.dit.load_state_dict(state_dict)

image = pipe(
    prompt="a white paper cup",
    flux_fill_image=Image.open("data/diffsynth_example_dataset/flux/FLUX.1-Fill-dev/cup.png").convert("RGB"),
    flux_fill_mask=Image.open("data/diffsynth_example_dataset/flux/FLUX.1-Fill-dev/cup_mask.png").convert("L"),
    height=1632, width=1232,
    seed=0, embedded_guidance=30.0, num_inference_steps=50,
)
image.save("image_FLUX.1-Fill-dev_full.jpg")
