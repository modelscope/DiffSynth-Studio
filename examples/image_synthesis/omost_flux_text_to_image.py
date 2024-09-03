
import torch
from diffsynth import download_models,FluxImagePipeline
from diffsynth.models.model_manager import ModelManager
from diffsynth.prompters.omost import OmostPromter
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

pipe = FluxImagePipeline.from_model_manager(model_manager,prompt_extender_classes=[OmostPromter])

negative_prompt = "dark, worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, dim, fuzzy, depth of Field, nsfw,"
image = pipe("generate an image of a witch who is releasing ice and fire magic",
             num_inference_steps=30, embedded_guidance=3.5,
             negative_prompt=negative_prompt)
image.save("image_omost.jpg")
