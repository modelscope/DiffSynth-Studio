from diffsynth.metrics import ModelConfig, UnifiedReward2Metric
from modelscope import dataset_snapshot_download
from PIL import Image
import json

dataset_snapshot_download(
    "DiffSynth-Studio/diffsynth_example_dataset",
    allow_file_pattern="flux/FLUX.1-dev/*",
    local_dir="./data/diffsynth_example_dataset",
)
image = Image.open("data/diffsynth_example_dataset/flux/FLUX.1-dev/1.jpg").convert("RGB")
prompt = "a dog"
metric = UnifiedReward2Metric.from_pretrained(
    model_config=ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="UnifiedReward-2.0-qwen35-9b/model-*.safetensors"),
    processor_config=ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="UnifiedReward-2.0-qwen35-9b/"),
    device="cuda",
)
details = metric.evaluate(prompt, image)[0]
print(json.dumps(details, indent=4, ensure_ascii=False))
