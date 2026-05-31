from diffsynth.metrics import ModelConfig, UnifiedRewardEditMetric
from modelscope import dataset_snapshot_download
from PIL import Image


dataset_snapshot_download(
    "DiffSynth-Studio/diffsynth_example_dataset",
    allow_file_pattern="qwen_image/Qwen-Image-Edit/*",
    local_dir="./data/diffsynth_example_dataset",
)

source_image = Image.open("data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit/edit/image1.jpg").convert("RGB")
edited_image_1 = Image.open("data/diffsynth_example_dataset/qwen_image/Qwen-Image-Edit/edit/image2.jpg").convert("RGB")
edited_image_2 = source_image
instruction = "将裙子改为粉色"
device = "cuda"

metric = UnifiedRewardEditMetric.from_pretrained(
    model_config=ModelConfig(
        model_id="DiffSynth-Studio/ImageMetrics",
        origin_file_pattern="UnifiedReward-Edit-qwen3vl-8b/model-*.safetensors",
    ),  
    processor_config=ModelConfig(
        model_id="DiffSynth-Studio/ImageMetrics",
        origin_file_pattern="UnifiedReward-Edit-qwen3vl-8b/",
    ),
    device=device,
)

pointwise_details = metric.evaluate(
    instruction, 
    [source_image, edited_image_1], 
    task="edit_pointwise_score"
)[0]
print("---UnifiedReward edit pointwise score---")
print(f"Score: {pointwise_details['score']:.3f}")
print(
    f"Editing Success: {pointwise_details['editing_success']:.3f}\n"
    f"Overediting: {pointwise_details['overediting']:.3f}"
)
print(pointwise_details, "\n")

pairwise_rank_details = metric.evaluate(
    instruction,
    [source_image, edited_image_1, edited_image_2],
    task="edit_pairwise_rank",
)[0]
print("---UnifiedReward edit pairwise rank score---")
print(f"Score: {pairwise_rank_details['score']}")
print(f"Winner: {pairwise_rank_details['winner']}")
print(pairwise_rank_details, "\n")

pairwise_score_details = metric.evaluate(
    instruction,
    [source_image, edited_image_1, edited_image_2],
    task="edit_pairwise_score",
)[0]
print("--UnifiedReward edit pairwise score---")
print(
    f"Image 1 Score: {pairwise_score_details['image_1_score']:.3f}\n"
    f"Image 2 Score: {pairwise_score_details['image_2_score']:.3f}"
)
print(pairwise_score_details)
