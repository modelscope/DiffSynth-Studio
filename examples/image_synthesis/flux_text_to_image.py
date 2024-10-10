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

prompt = "CG, masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait. The girl's flowing silver hair shimmers with every color of the rainbow and cascades down, merging with the floating flora around her."
negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"

# Disable classifier-free guidance (consistent with the original implementation of FLUX.1)
torch.manual_seed(9)
image = pipe(
    prompt=prompt,
    num_inference_steps=50, embedded_guidance=3.5
)
image.save("image_1024.jpg")

# Enable classifier-free guidance
torch.manual_seed(9)
image = pipe(
    prompt=prompt, negative_prompt=negative_prompt,
    num_inference_steps=50, cfg_scale=2.0, embedded_guidance=3.5
)
image.save("image_1024_cfg.jpg")

# Highres-fix
torch.manual_seed(10)
image = pipe(
    prompt=prompt,
    num_inference_steps=50, embedded_guidance=3.5,
    input_image=image.resize((2048, 2048)), height=2048, width=2048, denoising_strength=0.6, tiled=True
)
image.save("image_2048_highres.jpg")
