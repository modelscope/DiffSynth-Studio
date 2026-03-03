from diffsynth.pipelines.anima_image import AnimaImagePipeline, ModelConfig
import torch


pipe = AnimaImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="circlestone-labs/Anima", origin_file_pattern="split_files/diffusion_models/anima-preview.safetensors"),
        ModelConfig(model_id="circlestone-labs/Anima", origin_file_pattern="split_files/text_encoders/qwen_3_06b_base.safetensors"),
        ModelConfig(model_id="circlestone-labs/Anima", origin_file_pattern="split_files/vae/qwen_image_vae.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen3-0.6B", origin_file_pattern="./"),
    tokenizer_t5xxl_config=ModelConfig(model_id="stabilityai/stable-diffusion-3.5-large", origin_file_pattern="tokenizer_3/")
)
prompt = "Masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait."
negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"
image = pipe(prompt, seed=0, num_inference_steps=50)
image.save("image.jpg")
