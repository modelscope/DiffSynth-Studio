from diffsynth.models.flux_controlnet import FluxControlNet
from diffsynth import load_state_dict, ModelManager, FluxImagePipeline, hash_state_dict_keys, ControlNetConfigUnit
import torch
from PIL import Image
import numpy as np


model_manager = ModelManager(torch_dtype=torch.bfloat16, model_id_list=["FLUX.1-dev"])
model_manager.load_models([
    "models/ControlNet/InstantX/FLUX___1-dev-Controlnet-Union-alpha/diffusion_pytorch_model.safetensors",
    "models/ControlNet/jasperai/Flux___1-dev-Controlnet-Depth/diffusion_pytorch_model.safetensors",
    "models/ControlNet/jasperai/Flux___1-dev-Controlnet-Surface-Normals/diffusion_pytorch_model.safetensors",
    "models/ControlNet/jasperai/Flux___1-dev-Controlnet-Upscaler/diffusion_pytorch_model.safetensors",
    "models/ControlNet/alimama-creative/FLUX___1-dev-Controlnet-Inpainting-Alpha/diffusion_pytorch_model.safetensors",
    "models/ControlNet/alimama-creative/FLUX___1-dev-Controlnet-Inpainting-Beta/diffusion_pytorch_model.safetensors",
    "models/ControlNet/Shakker-Labs/FLUX___1-dev-ControlNet-Depth/diffusion_pytorch_model.safetensors",
    "models/ControlNet/Shakker-Labs/FLUX___1-dev-ControlNet-Union-Pro/diffusion_pytorch_model.safetensors"
])
pipe = FluxImagePipeline.from_model_manager(model_manager, controlnet_config_units=[
    ControlNetConfigUnit(processor_id="canny", model_path="models/ControlNet/InstantX/FLUX___1-dev-Controlnet-Union-alpha/diffusion_pytorch_model.safetensors", scale=0.3),
    ControlNetConfigUnit(processor_id="depth", model_path="models/ControlNet/jasperai/Flux___1-dev-Controlnet-Depth/diffusion_pytorch_model.safetensors", scale=0.1),
    ControlNetConfigUnit(processor_id="normal", model_path="models/ControlNet/jasperai/Flux___1-dev-Controlnet-Surface-Normals/diffusion_pytorch_model.safetensors", scale=0.1),
    ControlNetConfigUnit(processor_id="tile", model_path="models/ControlNet/jasperai/Flux___1-dev-Controlnet-Upscaler/diffusion_pytorch_model.safetensors", scale=0.05),
    ControlNetConfigUnit(processor_id="inpaint", model_path="models/ControlNet/alimama-creative/FLUX___1-dev-Controlnet-Inpainting-Alpha/diffusion_pytorch_model.safetensors", scale=0.01),
    ControlNetConfigUnit(processor_id="inpaint", model_path="models/ControlNet/alimama-creative/FLUX___1-dev-Controlnet-Inpainting-Beta/diffusion_pytorch_model.safetensors", scale=0.01),
    ControlNetConfigUnit(processor_id="depth", model_path="models/ControlNet/Shakker-Labs/FLUX___1-dev-ControlNet-Depth/diffusion_pytorch_model.safetensors", scale=0.05),
    ControlNetConfigUnit(processor_id="canny", model_path="models/ControlNet/Shakker-Labs/FLUX___1-dev-ControlNet-Union-Pro/diffusion_pytorch_model.safetensors", scale=0.3),
])

torch.manual_seed(0)

control_image = Image.open("controlnet_input.jpeg").resize((768, 1024))
control_mask = Image.open("controlnet_mask.jpg").resize((768, 1024))

prompt = "masterpiece, best quality, a beautiful girl, CG, blue sky, long red hair, black clothes"
negative_prompt = "oil painting, worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"

image = pipe(
    prompt=prompt, negative_prompt=negative_prompt,
    embedded_guidance=3.5, num_inference_steps=50,
    height=1024, width=768,
    controlnet_image=control_image, controlnet_inpaint_mask=control_mask,
)
image.save("image.jpg")
