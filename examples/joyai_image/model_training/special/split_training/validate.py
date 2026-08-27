from pathlib import Path


def latest_checkpoint(directory):
    checkpoints = list(Path(directory).glob("epoch-*.safetensors"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {directory}")
    return max(checkpoints, key=lambda path: int(path.stem.rsplit("-", 1)[-1]))


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

pipe.load_lora(pipe.dit, str(latest_checkpoint('./models/train/JoyAI-Image-Edit-split')))

prompt = "将裙子改为粉色"
edit_image = Image.open("data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit/edit/image1.jpg").convert("RGB")

image = pipe(
    prompt=prompt,
    edit_image=edit_image,
    height=1024,
    width=1024,
    seed=0,
    num_inference_steps=30,
    cfg_scale=5.0,
)
image.save('split_training_JoyAI-Image-Edit.jpg')
