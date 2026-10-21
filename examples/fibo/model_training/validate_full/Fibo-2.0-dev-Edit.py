from PIL import Image
from diffsynth.pipelines.fibo import FiboPipeline, ModelConfig
from diffsynth.core import load_state_dict
import torch


pipe = FiboPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="processor/"),
    font_config=ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="fonts/"),
)
state_dict = load_state_dict("./models/train/Fibo-2.0-dev-Edit_full/epoch-1.safetensors", torch_dtype=torch.bfloat16)
pipe.dit.load_state_dict(state_dict)
source = Image.open("data/diffsynth_example_dataset/fibo/Fibo-2.0-dev-Edit/image_1.png").convert("RGB")
mask = Image.open("data/diffsynth_example_dataset/fibo/Fibo-2.0-dev-Edit/mask.png").convert("L")
prompt = "Transform the video into Japanese anime style"
pipe(prompt=prompt, edit_image=source, seed=0, num_inference_steps=50, cfg_scale=5.0).save("Fibo-2.0-dev-Edit_full.jpg")
pipe(prompt=prompt, edit_image=source, mask=mask, seed=0, num_inference_steps=50, cfg_scale=5.0).save("Fibo-2.0-dev-Edit_full_mask.jpg")
