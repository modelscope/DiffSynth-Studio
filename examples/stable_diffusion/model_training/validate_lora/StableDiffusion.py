from diffsynth.pipelines.stable_diffusion import StableDiffusionPipeline, ModelConfig
import torch


pipe = StableDiffusionPipeline.from_pretrained(
    torch_dtype=torch.float32,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="AI-ModelScope/stable-diffusion-v1-5", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="AI-ModelScope/stable-diffusion-v1-5", origin_file_pattern="unet/diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id="AI-ModelScope/stable-diffusion-v1-5", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="AI-ModelScope/stable-diffusion-v1-5", origin_file_pattern="tokenizer/"),
)
pipe.load_lora(pipe.unet, "./models/train/StableDiffusion_lora/epoch-4.safetensors")
prompt = "dog, white and brown dog, sitting on wall, under pink flowers"
image = pipe(prompt=prompt, seed=42, rand_device="cuda", num_inference_steps=50, cfg_scale=7.5)
image.save("image.jpg")
