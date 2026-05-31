import os
from diffsynth.metrics import ModelConfig, QwenImageBenchMetric
from modelscope import dataset_snapshot_download
from PIL import Image


dataset_snapshot_download(
    "DiffSynth-Studio/diffsynth_example_dataset",
    allow_file_pattern="flux/FLUX.1-dev/*",
    local_dir="./data/diffsynth_example_dataset",
)

image = Image.open("data/diffsynth_example_dataset/flux/FLUX.1-dev/1.jpg").convert("RGB")
prompt = "dog, white and brown dog, sitting on wall, under pink flowers"
device = "cuda"

metric = QwenImageBenchMetric.from_pretrained(
    model_config=ModelConfig(
        model_id="Qwen/Qwen-Image-Bench",
        origin_file_pattern="model-*.safetensors",
    ),
    processor_config=ModelConfig(
        model_id="Qwen/Qwen-Image-Bench",
        origin_file_pattern="",
    ),
    device=device,
)

details = metric.evaluate(prompt, image)[0]
score = details["total_score"] if details["total_score"] is not None else 0.0
print(f"Total Score: {score:.3f}")
print(details["level1_scores"])
print(details["level2_scores"])
