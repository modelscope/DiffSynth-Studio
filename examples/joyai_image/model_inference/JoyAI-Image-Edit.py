from diffsynth.pipelines.joyai_image import JoyAIImagePipeline, ModelConfig
import torch
from PIL import Image
from modelscope import dataset_snapshot_download

# Download dataset
dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/diffsynth_example_dataset",
    local_dir="data/diffsynth_example_dataset",
    allow_file_pattern="joyai_image/JoyAI-Image-Edit/*"
)

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

# Use first sample from dataset
dataset_base_path = "data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit"
prompt = "将裙子改为粉色"
edit_images = Image.open(f"{dataset_base_path}/edit/image1.jpg").convert("RGB")

output = pipe(
    prompt=prompt,
    edit_images=[edit_images],
    height=1024,
    width=1024,
    seed=1,
    num_inference_steps=30,
    cfg_scale=5.0,
)

output.save("output_joyai_edit.png")
