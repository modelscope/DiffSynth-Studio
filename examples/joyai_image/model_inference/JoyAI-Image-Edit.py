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
)
pipe.eval()
# Image editing
prompt = "Turn the plate blue"
# Replace with your input image path
input_image = Image.open("/mnt/nas1/zhanghong/project26/main_project/opencode/packages/joyai-image/JoyAI-Image/test_images/test_1.jpg").convert("RGB")

output = pipe(
    prompt=prompt,
    edit_images=[input_image],
    edit_image_basesize=1024,
    height=1024,
    width=1024,
    seed=1,
    num_inference_steps=30,
    cfg_scale=5.0,
)

output.save("output_joyai_edit.png")
