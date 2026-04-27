from diffsynth.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline, ModelConfig
from diffsynth.core import load_state_dict
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    torch_dtype=torch.float32,
    model_configs=[
        ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="text_encoder_2/model.safetensors"),
        ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="unet/diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="tokenizer/"),
    tokenizer_2_config=ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="tokenizer_2/"),
)
state_dict = load_state_dict("./models/train/stable-diffusion-xl-base-1.0_full/epoch-1.safetensors", torch_dtype=torch.float32)
pipe.unet.load_state_dict(state_dict)

image = pipe(
    prompt="a dog",
    negative_prompt="",
    cfg_scale=7.0,
    height=1024,
    width=1024,
    seed=42,
    num_inference_steps=50,
)
image.save("image_stable-diffusion-xl-base-1.0_full.jpg")
