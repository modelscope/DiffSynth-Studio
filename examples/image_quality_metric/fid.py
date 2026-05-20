from diffsynth.metrics import FIDMetric, ModelConfig

reference_dir = ""
generated_dir = ""
device = "cuda"

metric = FIDMetric.from_pretrained(
    model_config=ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="FID/model.safetensors"),
    device=device,
)

score = metric.compute(reference_dir, generated_dir)
print(f"FID score: {score:.3f}")