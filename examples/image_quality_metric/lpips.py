from diffsynth.metrics import LPIPSMetric, ModelConfig
from modelscope import dataset_snapshot_download

dataset_snapshot_download(
    "DiffSynth-Studio/diffsynth_example_dataset",
    allow_file_pattern=["flux/FLUX.1-dev/*", "flux2/FLUX.2-dev/*"],
    local_dir="./data/diffsynth_example_dataset",
)

# net="alex" with LPIPS/alexnet.safetensors (default)
# For VGG: net="vgg", model_config=ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="LPIPS/vgg.safetensors")
# For SqueezeNet: net="squeeze", model_config=ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="LPIPS/squeezenet.safetensors")
metric = LPIPSMetric.from_pretrained(
    net="alex",
    model_config=ModelConfig(
        model_id="DiffSynth-Studio/ImageMetrics",
        origin_file_pattern="LPIPS/alexnet.safetensors",
    ),
    device="cuda",
    target_size=512,
)

score = metric.compute(
    "./data/diffsynth_example_dataset/flux/FLUX.1-dev/1.jpg",
    "./data/diffsynth_example_dataset/flux/FLUX.1-dev/2.jpg",
)
print(f"LPIPS score (image vs image): {score:.4f}")

score = metric.compute(
    "./data/diffsynth_example_dataset/flux/FLUX.1-dev",
    "./data/diffsynth_example_dataset/flux2/FLUX.2-dev",
)
print(f"LPIPS score (dir vs dir): {score:.4f}")
