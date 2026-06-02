from diffsynth.metrics import BioCLIPMetric, ModelConfig
from modelscope import dataset_snapshot_download
from PIL import Image

dataset_snapshot_download(
    "DiffSynth-Studio/diffsynth_example_dataset",
    allow_file_pattern="flux/FLUX.1-dev/*",
    local_dir="./data/diffsynth_example_dataset",
)
image = Image.open("data/diffsynth_example_dataset/flux/FLUX.1-dev/1.jpg").convert("RGB")
prompt = "a photo of Animalia Chordata Mammalia Carnivora Canidae Canis Canis lupus familiaris with common name domestic dog."
metric = BioCLIPMetric.from_pretrained(
    model_config=ModelConfig(model_id="DiffSynth-Studio/ImageMetrics", origin_file_pattern="BioCLIPv2/open_clip_model.safetensors"),
    device="cuda",
)
score = metric.compute(prompt, image)[0]
print(f"BioCLIP score: {score:.3f}")
