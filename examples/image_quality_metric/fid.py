from diffsynth.metrics import FIDMetric
from modelscope import dataset_snapshot_download

dataset_snapshot_download(
    "DiffSynth-Studio/diffsynth_example_dataset",
    allow_file_pattern="flux/FLUX.1-dev/*",
    local_dir="./data/diffsynth_example_dataset",
)

generated_dir = "data/diffsynth_example_dataset/flux/FLUX.1-dev"
device = "cuda"

reference_dir = FIDMetric.default_reference_dir(
    local_dir="data/examples/ImageQualityMetric/reference/coco_2014_caption_validation",
    max_images=10000,  # use None for the full validation split
)

metric = FIDMetric.from_pretrained(
    device=device,
    batch_size=16,
)

print("FID score:", metric.compute(reference_dir, generated_dir))
