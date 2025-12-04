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
        ModelConfig(model_id="ByteDance/InfiniteYou", origin_file_pattern="infu_flux_v1.0/aes_stage2/image_proj_model.bin"),
        ModelConfig(model_id="ByteDance/InfiniteYou", origin_file_pattern="infu_flux_v1.0/aes_stage2/InfuseNetModel/*.safetensors"),
    ],
)
pipe.load_lora(pipe.dit, "models/train/FLUX.1-dev-InfiniteYou_lora/epoch-4.safetensors", alpha=1)

image = pipe(
    prompt="a man with a red hat",
    controlnet_inputs=[ControlNetInput(
        image=Image.open("data/example_image_dataset/infiniteyou/image_1.jpg"),
    )],
    height=1024, width=1024,
    seed=0, rand_device="cuda",
)
image.save("image_FLUX.1-dev-InfiniteYou_lora.jpg")
