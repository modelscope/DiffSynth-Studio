import csv
from diffsynth.metrics import ImageRewardMetric, ModelConfig
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

metric = ImageRewardMetric.from_pretrained(
    model_config=ModelConfig(model_id="ZhipuAI/ImageReward"),
    device=device,
)

print("ImageReward score:", metric.compute(prompt, image)[0])