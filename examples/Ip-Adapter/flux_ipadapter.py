from diffsynth import ModelManager, download_models, FluxImagePipeline
import torch

# Download models (automatically)
# `models/IpAdapter/InstantX/FLUX.1-dev-IP-Adapter/ip-adapter.bin`: [link](https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter/blob/main/ip-adapter.bin)
# `models/IpAdapter/InstantX/FLUX.1-dev-IP-Adapter/image_encoder`: [link](https://huggingface.co/google/siglip-so400m-patch14-384)
download_models(["InstantX/FLUX.1-dev-IP-Adapter", "FLUX.1-dev"])

# Load models
model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda")
model_manager.load_models([
    "models/IpAdapter/InstantX/FLUX.1-dev-IP-Adapter/ip-adapter.bin",
    "models/IpAdapter/InstantX/FLUX.1-dev-IP-Adapter/image_encoder",
    "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
    "models/FLUX/FLUX.1-dev/text_encoder_2",
    "models/FLUX/FLUX.1-dev/ae.safetensors",
    "models/FLUX/FLUX.1-dev/flux1-dev.safetensors",
])
seed = 42
pipe = FluxImagePipeline.from_model_manager(model_manager)
torch.manual_seed(seed)
origin_prompt = "a rabbit in a garden, colorful flowers"
image = pipe(
    prompt=origin_prompt,
    cfg_scale=1.0, embedded_guidance=3.5,
    height=1280, width=960, num_inference_steps=30
)
image.save("style image.jpg")

torch.manual_seed(seed)
image = pipe(
    prompt="A piggy",
    cfg_scale=1.0, embedded_guidance=3.5,
    height=1280, width=960, num_inference_steps=30,
    ipadapter_images=[image], ipadapter_scale=0.7
)
image.save("A piggy.jpg")

