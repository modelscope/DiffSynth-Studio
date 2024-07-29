from diffsynth import ModelManager, SDImagePipeline, download_models
import torch


# Download models (automatically)
# `models/stable_diffusion/aingdiffusion_v12.safetensors`: [link](https://civitai.com/api/download/models/229575?type=Model&format=SafeTensor&size=full&fp=fp16)
# `models/IpAdapter/stable_diffusion/image_encoder/model.safetensors`: [link](https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors)
# `models/IpAdapter/stable_diffusion/ip-adapter_sd15.bin`: [link](https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin)
# `models/textual_inversion/verybadimagenegative_v1.3.pt`: [link](https://civitai.com/api/download/models/25820?type=Model&format=PickleTensor&size=full&fp=fp16)
download_models(["AingDiffusion_v12", "IP-Adapter-SD", "TextualInversion_VeryBadImageNegative_v1.3"])

# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models([
    "models/stable_diffusion/aingdiffusion_v12.safetensors",
    "models/IpAdapter/stable_diffusion/image_encoder/model.safetensors",
    "models/IpAdapter/stable_diffusion/ip-adapter_sd15.bin"
])
pipe = SDImagePipeline.from_model_manager(model_manager)
pipe.prompter.load_textual_inversions(["models/textual_inversion/verybadimagenegative_v1.3.pt"])

torch.manual_seed(1)
style_image = pipe(
    prompt="masterpiece, best quality, a car",
    negative_prompt="verybadimagenegative_v1.3",
    cfg_scale=7, clip_skip=2,
    height=512, width=512, num_inference_steps=50,
)
style_image.save("car.jpg")

image = pipe(
    prompt="masterpiece, best quality, a car running on the road",
    negative_prompt="verybadimagenegative_v1.3",
    cfg_scale=7, clip_skip=2,
    height=512, width=512, num_inference_steps=50,
    ipadapter_images=[style_image], ipadapter_scale=1.0
)
image.save("car_on_the_road.jpg")
