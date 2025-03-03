from diffsynth.extensions.ImageQualityMetric import download_preference_model, load_preference_model
from modelscope import dataset_snapshot_download
from PIL import Image


# Download example image
dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    allow_file_pattern="data/examples/ImageQualityMetric/image.jpg",
    local_dir="./"
)

# Parameters
prompt = "an orange cat"
image = Image.open("data/examples/ImageQualityMetric/image.jpg")
device = "cuda"
cache_dir = "./models"

# Run preference models
for model_name in ["ImageReward", "Aesthetic", "PickScore", "CLIP", "HPSv2", "HPSv2.1", "MPS"]:
    path = download_preference_model(model_name, cache_dir=cache_dir)
    preference_model = load_preference_model(model_name, device=device, path=path)
    print(model_name, preference_model.score(image, prompt))
