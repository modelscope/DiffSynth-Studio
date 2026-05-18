from PIL import Image
from diffsynth.metrics import PickScoreMetric, ModelConfig

prompt = ""
path_to_image = ""
image = Image.open(path_to_image).convert("RGB")
device = "cuda"

metric = PickScoreMetric.from_pretrained(
    model_config=ModelConfig(model_id="AI-ModelScope/PickScore_v1"),
    processor_config=ModelConfig(model_id="AI-ModelScope/CLIP-ViT-H-14-laion2B-s32B-b79K"),
    device=device,
)

print("PickScore score:", metric.calc_scores(prompt, image)[0])