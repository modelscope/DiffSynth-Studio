import torch
from diffsynth.pipelines.hidream_o1_image import HiDreamO1ImagePipeline, ModelConfig


pipe = HiDreamO1ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="HiDream-ai/HiDream-O1-Image", origin_file_pattern="model-*.safetensors"),
    ],
    processor_config=ModelConfig(model_id="HiDream-ai/HiDream-O1-Image", origin_file_pattern="./"),
)
state_dict = torch.load("./models/train/HiDream-O1-Image_full/epoch-1.safetensors", map_location="cuda", weights_only=False)
pipe.dit.load_state_dict(state_dict, strict=False)
image = pipe(
    prompt="dog,white and brown dog, sitting on wall, under pink flowers",
    negative_prompt=" ",
    cfg_scale=4.0,
    height=2048,
    width=2048,
    seed=42,
    num_inference_steps=50,
)
image.save("image.jpg")
