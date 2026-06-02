from diffsynth.metrics import ModelConfig, QwenImageBenchMetric
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
metric = QwenImageBenchMetric.from_pretrained(
    model_config=ModelConfig(model_id="Qwen/Qwen-Image-Bench", origin_file_pattern="model-*.safetensors"),
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Bench"),
    device="cuda",
)
details = metric.evaluate(prompt, image)[0]
print(json.dumps(details["level1_scores"], indent=4, ensure_ascii=False))
print(json.dumps(details["level2_scores"], indent=4, ensure_ascii=False))
print(json.dumps(details["level3_scores"], indent=4, ensure_ascii=False))
