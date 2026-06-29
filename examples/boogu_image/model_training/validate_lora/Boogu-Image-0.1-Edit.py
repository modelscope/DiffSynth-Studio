import torch
from PIL import Image
from diffsynth.pipelines.boogu_image import BooguImagePipeline, ModelConfig

pipe = BooguImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Boogu/Boogu-Image-0.1-Edit", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="Boogu/Boogu-Image-0.1-Edit", origin_file_pattern="mllm/*.safetensors"),
        ModelConfig(model_id="Boogu/Boogu-Image-0.1-Edit", origin_file_pattern="vae/*.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Boogu/Boogu-Image-0.1-Edit", origin_file_pattern="mllm/"),
)

pipe.load_lora(pipe.dit, "models/train/Boogu-Image-0.1-Edit_lora/epoch-4.safetensors")

prompt = "将裙子改为粉色"
edit_image = Image.open("data/diffsynth_example_dataset/boogu_image/Boogu-Image-0.1-Edit/edit/image1.jpg").convert("RGB")

output = pipe(
    prompt=prompt,
    negative_prompt="",
    edit_image=edit_image,
    height=1024,
    width=1024,
    seed=42,
    rand_device="cuda",
    num_inference_steps=50,
    cfg_scale=1.0,
)
output.save("image_Boogu-Image-0.1-Edit_lora.jpg")
