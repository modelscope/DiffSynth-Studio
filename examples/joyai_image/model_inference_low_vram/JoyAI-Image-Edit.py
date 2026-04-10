from diffsynth.pipelines.joyai_image import JoyAIImagePipeline, ModelConfig
import torch
from PIL import Image

pipe = JoyAIImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="jd-opensource/JoyAI-Image-Edit",
            origin_file_pattern="transformer/transformer.pth",
        ),
        ModelConfig(
            model_id="jd-opensource/JoyAI-Image-Edit",
            origin_file_pattern="JoyAI-Image-Und/model*.safetensors",
        ),
        ModelConfig(
            model_id="jd-opensource/JoyAI-Image-Edit",
            origin_file_pattern="vae/Wan2.1_VAE.pth",
        ),
    ],
    processor_config=ModelConfig(
        model_id="jd-opensource/JoyAI-Image-Edit",
        origin_file_pattern="JoyAI-Image-Und/",
    ),
    vram_limit=0.8,
)

prompt = "Turn the plate blue"
input_image = None  # Image.open("input.jpg").convert("RGB")

if input_image is not None:
    output = pipe(
        prompt=prompt,
        input_image=input_image,
        denoising_strength=1.0,
        seed=42,
        num_inference_steps=50,
        cfg_scale=5.0,
    )
else:
    output = pipe(
        prompt=prompt,
        seed=42,
        num_inference_steps=50,
        cfg_scale=5.0,
        height=1024,
        width=1024,
    )

output.save("output_joyai_edit_low_vram.png")
print("Saved output_joyai_edit_low_vram.png")
