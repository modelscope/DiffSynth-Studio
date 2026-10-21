from PIL import Image
from diffsynth.pipelines.fibo import FiboPipeline, ModelConfig
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
pipe.load_lora(pipe.dit, "./models/train/Fibo-2.0-dev-Edit_lora/epoch-4.safetensors")
source = Image.open("data/diffsynth_example_dataset/fibo/Fibo-2.0-dev-Edit/image_1.png").convert("RGB")
mask = Image.open("data/diffsynth_example_dataset/fibo/Fibo-2.0-dev-Edit/mask.png").convert("L")
prompt = "Transform the video into Japanese anime style"
pipe(prompt=prompt, edit_image=source, seed=0, num_inference_steps=50, cfg_scale=5.0).save("Fibo-2.0-dev-Edit_lora.jpg")
pipe(prompt=prompt, edit_image=source, mask=mask, seed=0, num_inference_steps=50, cfg_scale=5.0).save("Fibo-2.0-dev-Edit_lora_mask.jpg")
