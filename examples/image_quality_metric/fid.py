from diffsynth.metrics import FIDMetric

generated_dir = ""
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
