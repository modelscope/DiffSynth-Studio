from diffsynth.pipelines.boogu_image import BooguImagePipeline, ModelConfig
from PIL import Image
import torch
from modelscope import dataset_snapshot_download


vram_config = {
    "offload_dtype": torch.float8_e4m3fn,
    "offload_device": "cpu",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = BooguImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Boogu/Boogu-Image-0.1-Edit", origin_file_pattern="transformer/*.safetensors", **vram_config),
        ModelConfig(model_id="Boogu/Boogu-Image-0.1-Edit", origin_file_pattern="mllm/*.safetensors", **vram_config),
        ModelConfig(model_id="Boogu/Boogu-Image-0.1-Edit", origin_file_pattern="vae/*.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="Boogu/Boogu-Image-0.1-Edit", origin_file_pattern="mllm/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
dataset_snapshot_download("DiffSynth-Studio/example_image_dataset", allow_file_pattern="edit/image1.jpg", local_dir="data/example_image_dataset")
edit_image = Image.open("data/example_image_dataset/edit/image1.jpg").resize((1024, 1024))

output = pipe(
    prompt="Change the color of the dress to red.",
    negative_prompt="",
    edit_image=edit_image,
    height=1024,
    width=1024,
    seed=42,
    rand_device="cuda",
    num_inference_steps=50,
    cfg_scale=1.0,
)
output.save("image_Boogu-Image-0.1-Edit.jpg")
