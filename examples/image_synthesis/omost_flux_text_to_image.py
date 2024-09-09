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

pipe_omost = FluxImagePipeline.from_model_manager(model_manager, prompt_extender_classes=[OmostPromter])
pipe = FluxImagePipeline.from_model_manager(model_manager)

prompt = "A witch uses ice magic to fight against wild beasts"
seed = 7

torch.manual_seed(seed)
image = pipe_omost(
    prompt=prompt,
    num_inference_steps=30, embedded_guidance=3.5
)
image.save(f"image_omost.jpg")

torch.manual_seed(seed)
image2= pipe(
    prompt=prompt,
    num_inference_steps=30, embedded_guidance=3.5
)
image2.save(f"image.jpg")