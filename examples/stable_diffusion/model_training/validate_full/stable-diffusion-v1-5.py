from diffsynth.pipelines.stable_diffusion import StableDiffusionPipeline, ModelConfig
from diffsynth.core import load_state_dict
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    torch_dtype=torch.float32,
    model_configs=[
        ModelConfig(model_id="AI-ModelScope/stable-diffusion-v1-5", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="AI-ModelScope/stable-diffusion-v1-5", origin_file_pattern="unet/diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id="AI-ModelScope/stable-diffusion-v1-5", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="AI-ModelScope/stable-diffusion-v1-5", origin_file_pattern="tokenizer/"),
)
state_dict = load_state_dict("./models/train/stable-diffusion-v1-5_full/epoch-1.safetensors", torch_dtype=torch.float32)
pipe.unet.load_state_dict(state_dict)

image = pipe(
    prompt="a dog",
    negative_prompt="blurry, low quality, deformed",
    cfg_scale=7.5,
    height=512,
    width=512,
    seed=42,
    rand_device="cuda",
    num_inference_steps=50,
)
image.save("image_stable-diffusion-v1-5_full.jpg")
