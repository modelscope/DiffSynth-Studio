from diffsynth.metrics import ModelConfig, UnifiedRewardEditMetric
from modelscope import dataset_snapshot_download
from PIL import Image
import json

dataset_snapshot_download(
    "DiffSynth-Studio/diffsynth_example_dataset",
    allow_file_pattern="qwen_image/Qwen-Image-Edit/*",
    local_dir="./data/diffsynth_example_dataset",
)
source_image = Image.open("data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit/edit/image1.jpg").convert("RGB")
edited_image_1 = Image.open("data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit/edit/image2.jpg").convert("RGB")
edited_image_2 = source_image
instruction = "Change the dress to pink."
metric = UnifiedRewardEditMetric.from_pretrained(
    model_config=ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="UnifiedReward-Edit-qwen3vl-8b/model-*.safetensors"),  
    processor_config=ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="UnifiedReward-Edit-qwen3vl-8b/"),
    device="cuda",
)
details = metric.evaluate(instruction, [source_image, edited_image_1], task="edit_pointwise_score")[0]
print(json.dumps(details, indent=4, ensure_ascii=False))
details = metric.evaluate(instruction, [source_image, edited_image_1, edited_image_2], task="edit_pairwise_rank")[0]
print(json.dumps(details, indent=4, ensure_ascii=False))
details = metric.evaluate(instruction, [source_image, edited_image_1, edited_image_2], task="edit_pairwise_score")[0]
print(json.dumps(details, indent=4, ensure_ascii=False))
