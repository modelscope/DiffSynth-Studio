from PIL import Image
from diffsynth.metrics import AestheticMetric, ModelConfig

path_to_image = ""
image = Image.open(path_to_image).convert("RGB")
device = "cuda"

metric = AestheticMetric.from_pretrained(
    model_config=ModelConfig(model_id="AI-ModelScope/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"),
    device=device,
)

print("Aesthetic score:", metric.calc_scores(image)[0])
