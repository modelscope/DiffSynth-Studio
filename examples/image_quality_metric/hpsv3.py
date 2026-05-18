from PIL import Image
from diffsynth.metrics import HPSv3Metric, ModelConfig

prompt = ""
path_to_image = ""
image = Image.open(path_to_image).convert("RGB")
device = "cuda"

metric = HPSv3Metric.from_pretrained(
    model_config=ModelConfig(model_id="MizzenAI/HPSv3"),
    base_model_config=ModelConfig(model_id="Qwen/Qwen2-VL-7B-Instruct"),
    device=device,
)

print("HPSv3 score:", metric.calc_scores(prompt, image)[0])