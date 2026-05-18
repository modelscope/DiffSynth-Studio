from PIL import Image
from diffsynth.metrics import HPSv2Metric, ModelConfig

prompt = ""
path_to_image = ""
image = Image.open(path_to_image).convert("RGB")
device = "cuda"

metric = HPSv2Metric.from_pretrained(
    model_config=ModelConfig(model_id="AI-ModelScope/HPSv2"),
    processor_config=ModelConfig(model_id="AI-ModelScope/CLIP-ViT-H-14-laion2B-s32B-b79K"),
    version="v2.0",
    device=device,
)

print("HPSv2 score:", metric.calc_scores(prompt, image)[0])