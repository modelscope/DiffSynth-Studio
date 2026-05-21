from diffsynth.metrics import HPSv2Metric, ModelConfig
from modelscope import dataset_snapshot_download
from PIL import Image

dataset_snapshot_download(
    "DiffSynth-Studio/diffsynth_example_dataset",
    allow_file_pattern="flux/FLUX.1-dev/*",
    local_dir="./data/diffsynth_example_dataset",
)
image = Image.open("data/diffsynth_example_dataset/flux/FLUX.1-dev/1.jpg").convert("RGB")
prompt = "a dog"
metric = HPSv2Metric.from_pretrained(
    model_config=ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="HPSv2/model.safetensors"),
    device="cuda"
)
score = metric.compute(prompt, image)[0]
print(f"HPSv2 score: {score:.3f}")