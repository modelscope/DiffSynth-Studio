import torch
from diffsynth import ModelManager, FluxImagePipeline, download_models


download_models(["FLUX.1-dev"])
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
    "A captivating fantasy magic woman portrait set in the deep sea. The woman, with blue spaghetti strap silk dress, swims in the sea. Her flowing silver hair shimmers with every color of the rainbow and cascades down, merging with the floating flora around her. Smooth, delicate and fair skin.",
    num_inference_steps=30
)
image.save("image_1024.jpg")
