import torch
from diffsynth import ModelManager, FluxImagePipeline


model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda")
model_manager.load_models([
    "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
    "models/FLUX/FLUX.1-dev/text_encoder_2",
    "models/FLUX/FLUX.1-dev/ae.safetensors",
    "models/FLUX/FLUX.1-dev/flux1-dev.safetensors"
])

pipe = FluxImagePipeline.from_model_manager(model_manager)
torch.manual_seed(6)
image = pipe(
    "Anime style. A girl with long silver hair is under water, wearing a blue dress. Her eyes are blue. Her hair is waving in the water."
)
image.save("image.jpg")
