from diffsynth.pipelines.sensenova_u1_image import SenseNovaU1ImagePipeline, ModelConfig
from diffsynth.models.sensenova_u1_common import PATCH_SIZE, smart_resize
from modelscope import dataset_snapshot_download
from PIL import Image
import torch

pipe = SenseNovaU1ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT", origin_file_pattern="model*.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT", origin_file_pattern="./"),
)

dataset_snapshot_download(
    "DiffSynth-Studio/diffsynth_example_dataset",
    allow_file_pattern="sensenova_u1/SenseNova-U1.5-8B-MoT-Edit/*",
    local_dir="data/diffsynth_example_dataset",
)
edit_image = Image.open("data/diffsynth_example_dataset/sensenova_u1/SenseNova-U1.5-8B-MoT-Edit/edit/image1.jpg").convert("RGB")

# Keep the input aspect ratio and normalize the output to about 2048*2048 pixels.
height, width = smart_resize(
    edit_image.height, edit_image.width,
    factor=PATCH_SIZE, min_pixels=2048 * 2048, max_pixels=2048 * 2048,
)

image = pipe(
    prompt="Change the dress to pink.",
    edit_image=edit_image,
    seed=42, height=height, width=width,
    num_inference_steps=50, cfg_scale=4.0, shift=3.0,
)
image.save("image_SenseNova-U1.5-8B-MoT-Edit.jpg")
