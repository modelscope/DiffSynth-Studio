import torch
from PIL import Image
from diffsynth.pipelines.hidream_o1_image import HiDreamO1ImagePipeline
from diffsynth.core.loader.config import ModelConfig


pipe = HiDreamO1ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="HiDream-ai/HiDream-O1-Image-Dev",
            origin_file_pattern="model-*.safetensors",
        ),
    ],
    processor_config=ModelConfig(
        model_id="HiDream-ai/HiDream-O1-Image-Dev",
        origin_file_pattern="./",
    ),
)

# Load two reference images
ref_image_1 = Image.open("image.jpg").convert("RGB")

image = pipe(
    prompt="change her hair to black",
    cfg_scale=1.0,
    height=2048,
    width=2048,
    seed=42,
    num_inference_steps=28,
    model_type="dev",
    noise_scale=7.5,
    ref_images=[ref_image_1],
)
image.save("image_edit.jpg")
