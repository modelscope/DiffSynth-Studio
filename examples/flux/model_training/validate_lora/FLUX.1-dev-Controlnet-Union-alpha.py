import torch
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig, ControlNetInput
from PIL import Image


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ModelConfig(model_id="InstantX/FLUX.1-dev-Controlnet-Union-alpha", origin_file_pattern="diffusion_pytorch_model.safetensors"),
    ],
)
pipe.load_lora(pipe.dit, "models/train/FLUX.1-dev-Controlnet-Union-alpha_lora/epoch-4.safetensors", alpha=1)

image = pipe(
    prompt="a dog",
    controlnet_inputs=[ControlNetInput(
        image=Image.open("data/example_image_dataset/canny/image_1.jpg"),
        scale=0.9,
        processor_id="canny",
    )],
    height=768, width=768,
    seed=0, rand_device="cuda",
)
image.save("image_FLUX.1-dev-Controlnet-Union-alpha_lora.jpg")
