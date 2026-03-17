from diffsynth.diffusion.skills import SkillsPipeline
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
import torch
from PIL import Image


pipe = Flux2ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="tokenizer/"),
)
skills = SkillsPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/F2KB4B-Skills-ControlNet"),
        ModelConfig(model_id="DiffSynth-Studio/F2KB4B-Skills-Brightness"),
    ],
)
skill_cache = skills(
    positive_inputs = [
        {
            "model_id": 0,
            "image": Image.open("xxx.jpg"),
            "prompt": "一位长发少女，四周环绕着魔法粒子",
        },
        {
            "model_id": 1,
            "scale": 0.6,
        },
    ],
    negative_inputs = [
        {
            "model_id": 0,
            "image": Image.open("xxx.jpg"),
            "prompt": "一位长发少女，四周环绕着魔法粒子",
        },
        {
            "model_id": 1,
            "scale": 0.5,
        },
    ],
    pipe=pipe,
)
image = pipe(
    prompt="一位长发少女，四周环绕着魔法粒子",
    seed=0, rand_device="cuda", num_inference_steps=50, cfg_scale=4,
    height=1024, width=1024,
    **skill_cache,
)
image.save("image.jpg")
