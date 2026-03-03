from diffsynth.pipelines.anima_image import AnimaImagePipeline, ModelConfig
from diffsynth.core import load_state_dict
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
state_dict = load_state_dict("./models/train/anima-preview_full/epoch-1.safetensors", torch_dtype=torch.bfloat16)
pipe.dit.load_state_dict(state_dict)
prompt = "a dog"
image = pipe(prompt=prompt, seed=0)
image.save("image.jpg")