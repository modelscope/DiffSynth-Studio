from diffsynth.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline, ModelConfig
import torch


pipe = StableDiffusionXLPipeline.from_pretrained(
    torch_dtype=torch.float32,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="AI-ModelScope/stable-diffusion-xl-base-1.0", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="AI-ModelScope/stable-diffusion-xl-base-1.0", origin_file_pattern="text_encoder_2/model.safetensors"),
        ModelConfig(model_id="AI-ModelScope/stable-diffusion-xl-base-1.0", origin_file_pattern="unet/diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id="AI-ModelScope/stable-diffusion-xl-base-1.0", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="AI-ModelScope/stable-diffusion-xl-base-1.0", origin_file_pattern="tokenizer/"),
    tokenizer_2_config=ModelConfig(model_id="AI-ModelScope/stable-diffusion-xl-base-1.0", origin_file_pattern="tokenizer_2/"),
)
prompt = "dog, white and brown dog, sitting on wall, under pink flowers"
image = pipe(prompt=prompt, seed=42, rand_device="cuda", num_inference_steps=50, cfg_scale=5.0)
image.save("image.jpg")
