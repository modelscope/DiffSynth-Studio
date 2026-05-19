import csv
from diffsynth.metrics import HPSv2Metric, ModelConfig
from modelscope import dataset_snapshot_download
from PIL import Image

dataset_snapshot_download(
    "DiffSynth-Studio/diffsynth_example_dataset",
    allow_file_pattern="flux/FLUX.1-dev/*",
    local_dir="./data/diffsynth_example_dataset",
)

image = Image.open("data/diffsynth_example_dataset/flux/FLUX.1-dev/1.jpg").convert("RGB")
prompt = "dog,white and brown dog, sitting on wall, under pink flowers"
device = "cuda"

metric = HPSv2Metric.from_pretrained(
    model_config=ModelConfig(model_id="AI-ModelScope/HPSv2"),
    processor_config=ModelConfig(model_id="AI-ModelScope/CLIP-ViT-H-14-laion2B-s32B-b79K"),
    version="v2.0",    # choice: v2.0, v2.1
    device=device,
)

print("HPSv2 score:", metric.compute(prompt, image)[0])