import torch
from diffsynth import ModelManager, FluxImagePipeline, download_models


download_models(["FLUX.1-dev"])
model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
model_manager.load_models([
    "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
    "models/FLUX/FLUX.1-dev/text_encoder_2",
    "models/FLUX/FLUX.1-dev/ae.safetensors",
    "models/FLUX/FLUX.1-dev/flux1-dev.safetensors"
])
pipe = FluxImagePipeline.from_model_manager(model_manager, device='cuda')
pipe.enable_cpu_offload()

prompt = "CG. Full body. A captivating fantasy magic woman portrait in the deep sea. The woman, with blue spaghetti strap silk dress, swims in the sea. Her flowing silver hair shimmers with every color of the rainbow and cascades down, merging with the floating flora around her. Smooth, delicate and fair skin."
negative_prompt = "dark, worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, dim, fuzzy, depth of Field, nsfw,"

# Disable classifier-free guidance (consistent with the original implementation of FLUX.1)
torch.manual_seed(6)
image = pipe(
    prompt=prompt,
    num_inference_steps=30, embedded_guidance=3.5
)
image.save("image_1024.jpg")

# Enable classifier-free guidance
torch.manual_seed(6)
image = pipe(
    prompt=prompt, negative_prompt=negative_prompt,
    num_inference_steps=30, cfg_scale=2.0, embedded_guidance=3.5
)
image.save("image_1024_cfg.jpg")

# Highres-fix
torch.manual_seed(7)
image = pipe(
    prompt=prompt,
    num_inference_steps=30, embedded_guidance=3.5,
    input_image=image.resize((2048, 2048)), height=2048, width=2048, denoising_strength=0.6, tiled=True
)
image.save("image_2048_highres.jpg")
