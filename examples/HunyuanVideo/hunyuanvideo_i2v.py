import torch
from diffsynth import ModelManager, HunyuanVideoPipeline, download_models, save_video
from diffsynth.prompters.hunyuan_video_prompter import HunyuanVideoPrompter
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


def generate_crop_size_list(base_size=256, patch_size=32, max_ratio=4.0):
    num_patches = round((base_size / patch_size)**2)
    assert max_ratio >= 1.0
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list


def get_closest_ratio(height: float, width: float, ratios: list, buckets: list):
    aspect_ratio = float(height) / float(width)
    closest_ratio_id = np.abs(ratios - aspect_ratio).argmin()
    closest_ratio = min(ratios, key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return buckets[closest_ratio_id], float(closest_ratio)


def prepare_vae_inputs(semantic_images, i2v_resolution="720p"):
    if i2v_resolution == "720p":
        bucket_hw_base_size = 960
    elif i2v_resolution == "540p":
        bucket_hw_base_size = 720
    elif i2v_resolution == "360p":
        bucket_hw_base_size = 480
    else:
        raise ValueError(f"i2v_resolution: {i2v_resolution} must be in [360p, 540p, 720p]")
    origin_size = semantic_images[0].size

    crop_size_list = generate_crop_size_list(bucket_hw_base_size, 32)
    aspect_ratios = np.array([round(float(h) / float(w), 5) for h, w in crop_size_list])
    closest_size, closest_ratio = get_closest_ratio(origin_size[1], origin_size[0], aspect_ratios, crop_size_list)
    ref_image_transform = transforms.Compose([
        transforms.Resize(closest_size),
        transforms.CenterCrop(closest_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    semantic_image_pixel_values = [ref_image_transform(semantic_image) for semantic_image in semantic_images]
    semantic_image_pixel_values = torch.cat(semantic_image_pixel_values).unsqueeze(0).unsqueeze(2)
    return semantic_image_pixel_values


model_manager = ModelManager()

# The other modules are loaded in float16.

model_manager.load_models(
    [
        "models/HunyuanVideoI2V/transformers/mp_rank_00_model_states.pt"
    ],
    torch_dtype=torch.bfloat16, # you can use torch_dtype=torch.float8_e4m3fn to enable quantization.
    device="cuda"
)

model_manager.load_models(
    [
        "models/HunyuanVideo/text_encoder/model.safetensors",
        "models/HunyuanVideoI2V/text_encoder_2",
        'models/HunyuanVideoI2V/vae/pytorch_model.pt'
        
    ],
    torch_dtype=torch.float16,
    device="cuda"
)
# The computation device is "cuda".
pipe = HunyuanVideoPipeline.from_model_manager(
    model_manager,
    torch_dtype=torch.bfloat16,
    device="cuda",
    enable_vram_management=False
)
# Although you have enough VRAM, we still recommend you to enable offload.
pipe.enable_cpu_offload()
print()