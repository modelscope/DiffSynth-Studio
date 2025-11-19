import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig, ControlNetInput
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
        ModelConfig(model_id="alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", origin_file_pattern="diffusion_pytorch_model.safetensors"),
    ],
)
state_dict = load_state_dict("models/train/FLUX.1-dev-Controlnet-Inpainting-Beta_full/epoch-0.safetensors")
pipe.controlnet.models[0].load_state_dict(state_dict)

image = pipe(
    prompt="a cat sitting on a chair, wearing sunglasses",
    controlnet_inputs=[ControlNetInput(
        image=Image.open("data/example_image_dataset/inpaint/image_1.jpg"),
        inpaint_mask=Image.open("data/example_image_dataset/inpaint/mask.jpg"),
        scale=0.9
    )],
    height=1024, width=1024,
    seed=0, rand_device="cuda",
)
image.save("image_FLUX.1-dev-Controlnet-Inpainting-Beta_full.jpg")
