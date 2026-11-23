from diffsynth.pipelines.hiuni import HiUniPipeline, ModelConfig
from modelscope import dataset_snapshot_download
from PIL import Image
import torch

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

pipe = HiUniPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="ernie-research/HiUni", origin_file_pattern="model-*.safetensors", download_source="huggingface", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="ernie-research/HiUni", origin_file_pattern="tokenizer*.json", download_source="huggingface"),
    processor_config=ModelConfig(model_id="ernie-research/HiUni", origin_file_pattern="preprocessor_config.json", download_source="huggingface"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

# Text-to-image
prompt = "A cozy reading nook by a rain-streaked window, a ginger cat asleep on a knitted blanket, warm lamp light, film photography."
image = pipe(prompt, height=2048, width=2048, seed=42, num_inference_steps=25, cfg_scale=4.0, shift=4.0)
image.save("image_HiUni.jpg")

# Instruction-based image editing
dataset_snapshot_download("DiffSynth-Studio/example_image_dataset", allow_file_pattern="edit/image1.jpg", local_dir="data/example_image_dataset")
edit_image = Image.open("data/example_image_dataset/edit/image1.jpg").convert("RGB")
image = pipe("Change the background to a snowy mountain.", edit_image=edit_image, seed=42, num_inference_steps=25, cfg_scale=4.0)
image.save("image_HiUni_edit.jpg")

# Image understanding and text chat
print(pipe.chat("Describe this image in detail.", images=edit_image))
print(pipe.chat("Write a short poem about autumn.", think_mode="empty"))
