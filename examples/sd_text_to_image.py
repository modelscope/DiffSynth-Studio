from diffsynth import ModelManager, SDImagePipeline, ControlNetConfigUnit
import torch


# Download models
# `models/stable_diffusion/aingdiffusion_v12.safetensors`: [link](https://civitai.com/api/download/models/229575?type=Model&format=SafeTensor&size=full&fp=fp16)
# `models/ControlNet/control_v11p_sd15_lineart.pth`: [link](https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth)
# `models/ControlNet/control_v11f1e_sd15_tile.pth`: [link](https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth)
# `models/Annotators/sk_model.pth`: [link](https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth)
# `models/Annotators/sk_model2.pth`: [link](https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model2.pth)


# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_textual_inversions("models/textual_inversion")
model_manager.load_models([
    "models/stable_diffusion/aingdiffusion_v12.safetensors",
    "models/ControlNet/control_v11f1e_sd15_tile.pth",
    "models/ControlNet/control_v11p_sd15_lineart.pth"
])
pipe = SDImagePipeline.from_model_manager(
    model_manager,
    [
        ControlNetConfigUnit(
            processor_id="tile",
            model_path=rf"models/ControlNet/control_v11f1e_sd15_tile.pth",
            scale=0.5
        ),
        ControlNetConfigUnit(
            processor_id="lineart",
            model_path=rf"models/ControlNet/control_v11p_sd15_lineart.pth",
            scale=0.7
        ),
    ]
)

prompt = "masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait,"
negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"

torch.manual_seed(0)
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    cfg_scale=7.5, clip_skip=1,
    height=512, width=512, num_inference_steps=80,
)
image.save("512.jpg")

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    cfg_scale=7.5, clip_skip=1,
    input_image=image.resize((1024, 1024)), controlnet_image=image.resize((1024, 1024)),
    height=1024, width=1024, num_inference_steps=40, denoising_strength=0.7,
)
image.save("1024.jpg")

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    cfg_scale=7.5, clip_skip=1,
    input_image=image.resize((2048, 2048)), controlnet_image=image.resize((2048, 2048)),
    height=2048, width=2048, num_inference_steps=20, denoising_strength=0.7,
)
image.save("2048.jpg")

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    cfg_scale=7.5, clip_skip=1,
    input_image=image.resize((4096, 4096)), controlnet_image=image.resize((4096, 4096)),
    height=4096, width=4096, num_inference_steps=10, denoising_strength=0.5,
    tiled=True, tile_size=128, tile_stride=64
)
image.save("4096.jpg")
