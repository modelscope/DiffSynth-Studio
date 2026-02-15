from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
import torch


pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Tongyi-MAI/Z-Image", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
)
prompt = "Chinese woman in red Hanfu holding a fan, with a bright yellow neon lightning bolt floating above her palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."

image = pipe(
    prompt=prompt, 
    seed=42, 
    num_inference_steps=50,
    cfg_scale=4,
    rand_device="cuda",
    enable_ses=True,
    ses_reward_model="pick",
    ses_eval_budget=20,
    ses_inference_steps=10
)
image.save("image_Z-Image_ses.jpg")


