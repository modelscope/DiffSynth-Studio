from diffsynth.pipelines.z_image import (
    ZImagePipeline, ModelConfig,
    ZImageUnit_Image2LoRAEncode, ZImageUnit_Image2LoRADecode
)
from modelscope import snapshot_download
from safetensors.torch import save_file
import torch
from PIL import Image

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cuda",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cuda",
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
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Base-1211_Temp", origin_file_pattern="transformer/*.safetensors", **vram_config),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors", **vram_config),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/General-Image-Encoders", origin_file_pattern="SigLIP2-G384/model.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/General-Image-Encoders", origin_file_pattern="DINOv3-7B/model.safetensors", **vram_config),
        ModelConfig("models/train/Z-Image-i2L_v13/step-58000.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
    vram_limit=80,
)

# Load images
snapshot_download(
    model_id="DiffSynth-Studio/Qwen-Image-i2L",
    allow_file_pattern="assets/style/*",
    local_dir="data/examples"
)
for style_id in range(1, 5):
    images = [Image.open(f"data/examples/assets/style/{style_id}/{i}.jpg") for i in range(4)]

    with torch.no_grad():
        embs = ZImageUnit_Image2LoRAEncode().process(pipe, image2lora_images=images)
        lora = ZImageUnit_Image2LoRADecode().process(pipe, **embs)["lora"]

    prompt = "a cat"
    pipe.clear_lora()
    pipe.load_lora(pipe.dit, state_dict=lora, alpha=1)
    image = pipe(prompt=prompt, seed=123, cfg_scale=4, num_inference_steps=50)
    image.save(f"image_lora_{style_id}.jpg")

pipe.clear_lora()
image = pipe(prompt=prompt, seed=123, cfg_scale=4, num_inference_steps=50)
image.save("image_base.jpg")
