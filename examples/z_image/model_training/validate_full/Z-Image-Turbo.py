from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
from diffsynth.core import load_state_dict
import torch


pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
)
state_dict = load_state_dict("./models/train/Z-Image-Turbo_full/epoch-1.safetensors", torch_dtype=torch.bfloat16)
pipe.dit.load_state_dict(state_dict)
prompt = "a dog"
image = pipe(prompt=prompt, seed=42, rand_device="cuda")
image.save("image.jpg")
