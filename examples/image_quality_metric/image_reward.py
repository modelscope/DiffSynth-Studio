from PIL import Image
from diffsynth.metrics import ImageRewardMetric, ModelConfig

prompt = ""
path_to_image = ""
image = Image.open(path_to_image).convert("RGB")
device = "cuda"

metric = ImageRewardMetric.from_pretrained(
    model_config=ModelConfig(model_id="ZhipuAI/ImageReward"),
    device=device,
)

print("ImageReward score:", metric.calc_scores(prompt, image)[0])