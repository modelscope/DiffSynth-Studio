import torch
from PIL import Image
from diffsynth.pipelines.hidream_o1_image import HiDreamO1ImagePipeline
from diffsynth.core.loader.config import ModelConfig


vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}


pipe = HiDreamO1ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="HiDream-ai/HiDream-O1-Image",
            origin_file_pattern="model-*.safetensors",
            **vram_config,
        ),
    ],
    processor_config=ModelConfig(
        model_id="HiDream-ai/HiDream-O1-Image",
        origin_file_pattern="./",
    ),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

# Load two reference images
ref_image_1 = Image.open("assets/example_ref_1.jpg").convert("RGB")
ref_image_2 = Image.open("assets/example_ref_2.jpg").convert("RGB")

image = pipe(
    prompt="change the background to a snowy mountain landscape",
    negative_prompt=" ",
    cfg_scale=4.0,
    height=2048,
    width=2048,
    seed=42,
    num_inference_steps=50,
    ref_images=[ref_image_1, ref_image_2],
)
image.save("image_edit.jpg")
