import torch
from diffsynth.core import ModelConfig
from diffsynth.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = StableDiffusionXLPipeline.from_pretrained(
    torch_dtype=torch.float32,
    model_configs=[
        ModelConfig(model_id="AI-ModelScope/stable-diffusion-xl-base-1.0", origin_file_pattern="text_encoder/model.safetensors", **vram_config),
        ModelConfig(model_id="AI-ModelScope/stable-diffusion-xl-base-1.0", origin_file_pattern="text_encoder_2/model.safetensors", **vram_config),
        ModelConfig(model_id="AI-ModelScope/stable-diffusion-xl-base-1.0", origin_file_pattern="unet/diffusion_pytorch_model.safetensors", **vram_config),
        ModelConfig(model_id="AI-ModelScope/stable-diffusion-xl-base-1.0", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="AI-ModelScope/stable-diffusion-xl-base-1.0", origin_file_pattern="tokenizer/"),
    tokenizer_2_config=ModelConfig(model_id="AI-ModelScope/stable-diffusion-xl-base-1.0", origin_file_pattern="tokenizer_2/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

image = pipe(
    prompt="a photo of an astronaut riding a horse on mars",
    negative_prompt="",
    cfg_scale=5.0,
    height=1024,
    width=1024,
    seed=42,
    num_inference_steps=50,
)
image.save("output_stable_diffusion_xl_t2i_low_vram.png")
print("Image saved to output_stable_diffusion_xl_t2i_low_vram.png")
