from diffsynth.pipelines.sensenova_u1_image import SenseNovaU1ImagePipeline, ModelConfig
from modelscope import dataset_snapshot_download
from PIL import Image
import torch

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = SenseNovaU1ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT", origin_file_pattern="model*.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT", origin_file_pattern="./"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

dataset_snapshot_download(
    "DiffSynth-Studio/diffsynth_example_dataset",
    allow_file_pattern="sensenova_u1/SenseNova-U1.5-8B-MoT-Edit/*",
    local_dir="data/diffsynth_example_dataset",
)
dataset_path = "data/diffsynth_example_dataset/sensenova_u1/SenseNova-U1.5-8B-MoT-Edit"
edit_image = Image.open(f"{dataset_path}/edit/image1.jpg").convert("RGB")
color_image = Image.open(f"{dataset_path}/edit/image_color.jpg").convert("RGB")

image = pipe(
    prompt="Change the dress to pink.",
    edit_image=edit_image,
    seed=42, height=2048, width=2048,
    num_inference_steps=50, cfg_scale=4.0, shift=3.0,
)
image.save("image_SenseNova-U1.5-8B-MoT-Edit.jpg")

# Multiple inputs are passed as a list and are numbered in the order given, so the prompt can
# refer to them as Figure 1, Figure 2, and so on.
image = pipe(
    prompt="Change the color of the dress in Figure 1 to the color shown in Figure 2.",
    edit_image=[edit_image, color_image],
    seed=42, height=2048, width=2048,
    num_inference_steps=50, cfg_scale=4.0, shift=3.0,
)
image.save("image_SenseNova-U1.5-8B-MoT-Edit-MultiImage.jpg")
