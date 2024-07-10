from diffsynth import ModelManager, HunyuanDiTImagePipeline, download_models
import torch


# Download models (automatically)
# `models/HunyuanDiT/t2i/clip_text_encoder/pytorch_model.bin`: [link](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/resolve/main/t2i/clip_text_encoder/pytorch_model.bin)
# `models/HunyuanDiT/t2i/mt5/pytorch_model.bin`: [link](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/resolve/main/t2i/mt5/pytorch_model.bin)
# `models/HunyuanDiT/t2i/model/pytorch_model_ema.pt`: [link](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/resolve/main/t2i/model/pytorch_model_ema.pt)
# `models/HunyuanDiT/t2i/sdxl-vae-fp16-fix/diffusion_pytorch_model.bin`: [link](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT/resolve/main/t2i/sdxl-vae-fp16-fix/diffusion_pytorch_model.bin)
download_models(["HunyuanDiT"])

# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models([
    "models/HunyuanDiT/t2i/clip_text_encoder/pytorch_model.bin",
    "models/HunyuanDiT/t2i/mt5/pytorch_model.bin",
    "models/HunyuanDiT/t2i/model/pytorch_model_ema.pt",
    "models/HunyuanDiT/t2i/sdxl-vae-fp16-fix/diffusion_pytorch_model.bin"
])
pipe = HunyuanDiTImagePipeline.from_model_manager(model_manager)

prompt = "一幅充满诗意美感的全身肖像画，画中一位银发、蓝色眼睛、身穿蓝色连衣裙的少女漂浮在水下，周围是光彩的气泡，和煦的阳光透过水面折射进水下"
negative_prompt = "错误的眼睛，糟糕的人脸，毁容，糟糕的艺术，变形，多余的肢体，模糊的颜色，模糊，重复，病态，残缺，"

# Enjoy!
torch.manual_seed(0)
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50, height=1024, width=1024,
)
image.save("image_1024.png")

# Highres fix
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    input_image=image.resize((2048, 2048)),
    num_inference_steps=50, height=2048, width=2048,
    denoising_strength=0.4, tiled=True,
)
image.save("image_2048.png")