import torch
from diffsynth import download_models, ModelManager, OmostPromter, FluxImagePipeline


download_models(["OmostPrompt"])
download_models(["FLUX.1-dev"])

model_manager = ModelManager(torch_dtype=torch.bfloat16)
model_manager.load_models([
    "models/OmostPrompt/omost-llama-3-8b-4bits",
    "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
    "models/FLUX/FLUX.1-dev/text_encoder_2",
    "models/FLUX/FLUX.1-dev/ae.safetensors",
    "models/FLUX/FLUX.1-dev/flux1-dev.safetensors"
])

pipe = FluxImagePipeline.from_model_manager(model_manager, prompt_extender_classes=[OmostPromter])

torch.manual_seed(0)
image = pipe(
    prompt="an image of a witch who is releasing ice and fire magic",
    num_inference_steps=30, embedded_guidance=3.5
)
image.save("image_omost.jpg")
