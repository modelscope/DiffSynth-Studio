from pathlib import Path


def latest_checkpoint(directory):
    checkpoints = list(Path(directory).glob("epoch-*.safetensors"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {directory}")
    return max(checkpoints, key=lambda path: int(path.stem.rsplit("-", 1)[-1]))


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
pipe.load_lora(pipe.dit, str(latest_checkpoint('./models/train/HiDream-O1-Image_split')))
image = pipe(
    prompt="dog,white and brown dog, sitting on wall, under pink flowers",
    negative_prompt=" ",
    cfg_scale=4.0,
    height=2048,
    width=2048,
    seed=42,
    num_inference_steps=50,
)
image.save('split_training_HiDream-O1-Image.jpg')
