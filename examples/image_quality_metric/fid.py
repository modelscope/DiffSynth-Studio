from diffsynth.metrics import FIDMetric, ModelConfig
from modelscope import dataset_snapshot_download

dataset_snapshot_download(
    "DiffSynth-Studio/diffsynth_example_dataset",
    allow_file_pattern=["flux/FLUX.1-dev/*", "flux2/FLUX.2-dev/*"],
    local_dir="./data/diffsynth_example_dataset",
)
metric = FIDMetric.from_pretrained(
    model_config=ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="FID/model.safetensors"),
    device="cuda",
)
score = metric.compute(
    "./data/diffsynth_example_dataset/flux/FLUX.1-dev",
    "./data/diffsynth_example_dataset/flux2/FLUX.2-dev",
)
print(f"FID score: {score:.3f}")