import torch
from diffsynth.pipelines.hidream_o1_image import HiDreamO1ImagePipeline, ModelConfig


pipe = HiDreamO1ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="HiDream-ai/HiDream-O1-Image-Dev", origin_file_pattern="model-*.safetensors"),
    ],
    processor_config=ModelConfig(model_id="HiDream-ai/HiDream-O1-Image-Dev", origin_file_pattern="./"),
)
pipe.load_lora(pipe.dit, "./models/train/HiDream-O1-Image-Dev_lora/epoch-4.safetensors")
image = pipe(
    prompt="dog,white and brown dog, sitting on wall, under pink flowers",
    cfg_scale=1.0,
    height=2048,
    width=2048,
    seed=42,
    num_inference_steps=28,
    model_type="dev",
    noise_scale=7.5,
)
image.save("image.jpg")
