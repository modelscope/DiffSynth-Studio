import torch
from diffsynth import ModelManager, FluxImagePipeline, download_models


download_models(["FLUX.1-dev"])
model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda")
model_manager.load_models([
    "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
    "models/FLUX/FLUX.1-dev/text_encoder_2",
    "models/FLUX/FLUX.1-dev/ae.safetensors",
    "models/ostris/Flex.2-preview/Flex.2-preview.safetensors"
])
pipe = FluxImagePipeline.from_model_manager(model_manager)

prompt = "CG, masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait. The girl's flowing silver hair shimmers with every color of the rainbow and cascades down, merging with the floating flora around her."

torch.manual_seed(9)
image = pipe(
    prompt=prompt,
    num_inference_steps=50, embedded_guidance=3.5
)
image.save("image_1024.jpg")
