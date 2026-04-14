import torch
from PIL import Image
from diffsynth.pipelines.joyai_image import JoyAIImagePipeline, ModelConfig

pipe = JoyAIImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="jd-opensource/JoyAI-Image-Edit", origin_file_pattern="transformer/transformer.pth"),
        ModelConfig(model_id="jd-opensource/JoyAI-Image-Edit", origin_file_pattern="JoyAI-Image-Und/model*.safetensors"),
        ModelConfig(model_id="jd-opensource/JoyAI-Image-Edit", origin_file_pattern="vae/Wan2.1_VAE.pth"),
    ],
    processor_config=ModelConfig(model_id="jd-opensource/JoyAI-Image-Edit", origin_file_pattern="JoyAI-Image-Und/"),
)

# Load LoRA weights from dual-stage training output
pipe.load_lora(pipe.dit, "models/train/JoyAI-Image-Edit-lora/epoch-4.safetensors")

# Use training dataset prompt and edit_images
prompt = "将裙子改为粉色"
edit_images = Image.open("data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit/edit/image1.jpg").convert("RGB")

image = pipe(
    prompt=prompt,
    edit_images=[edit_images],
    height=1024,
    width=1024,
    seed=0,
    num_inference_steps=30,
    cfg_scale=5.0,
)
image.save("image_lora.jpg")
