from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
import torch
from diffsynth.utils.device import get_device_type


pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=get_device_type(),
    model_configs=[
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
)
pipe.load_lora(pipe.dit, "./models/train/Z-Image-Turbo_lora_differential/epoch-4.safetensors")
prompt = "a dog"
image = pipe(prompt=prompt, seed=42, rand_device=get_device_type())
image.save("image.jpg")
