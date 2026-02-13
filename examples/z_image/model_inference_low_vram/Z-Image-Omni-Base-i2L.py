from diffsynth.pipelines.z_image import (
    ZImagePipeline, ModelConfig,
    ZImageUnit_Image2LoRAEncode, ZImageUnit_Image2LoRADecode
)
from modelscope import snapshot_download
from safetensors.torch import save_file
import torch
from PIL import Image

# Use `vram_config` to enable LoRA hot-loading
vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

# Load models
pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Omni-Base", origin_file_pattern="transformer/*.safetensors", **vram_config),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Omni-Base", origin_file_pattern="siglip/model.safetensors", **vram_config),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors", **vram_config),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/General-Image-Encoders", origin_file_pattern="SigLIP2-G384/model.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/General-Image-Encoders", origin_file_pattern="DINOv3-7B/model.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/Z-Image-Omni-Base-i2L", origin_file_pattern="model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

# Load images
snapshot_download(
    model_id="DiffSynth-Studio/Z-Image-Omni-Base-i2L",
    allow_file_pattern="assets/style/*",
    local_dir="data/style_input"
)
images = [Image.open(f"data/style_input/assets/style/1/{i}.jpg") for i in range(6)]

# Image to LoRA
with torch.no_grad():
    embs = ZImageUnit_Image2LoRAEncode().process(pipe, image2lora_images=images)
    lora = ZImageUnit_Image2LoRADecode().process(pipe, **embs)["lora"]
save_file(lora, "lora.safetensors")

# Generate images
prompt = "a cat"
negative_prompt = "泛黄，发绿，模糊，低分辨率，低质量图像，扭曲的肢体，诡异的外观，丑陋，AI感，噪点，网格感，JPEG压缩条纹，异常的肢体，水印，乱码，意义不明的字符"
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    seed=0, cfg_scale=7, num_inference_steps=50,
    positive_only_lora=lora,
    sigma_shift=8
)
image.save("image.jpg")
