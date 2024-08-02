from diffsynth import ModelManager, SDXLImagePipeline, download_models
import torch

# Download models
# https://huggingface.co/Kwai-Kolors/Kolors
download_models(["Kolors"])
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda",
                             file_path_list=[
                                 "models/kolors/Kolors/text_encoder",
                                 "models/kolors/Kolors/unet/diffusion_pytorch_model.safetensors",
                                 "models/kolors/Kolors/vae/diffusion_pytorch_model.safetensors"
                             ])
pipe = SDXLImagePipeline.from_model_manager(model_manager)

# Optional (Int4 quantize) pip install cpm_kernels
# pipe.text_encoder_kolors = pipe.text_encoder_kolors.quantize(4)
# torch.cuda.empty_cache()

prompt = "一幅充满诗意美感的全身画，泛红的肤色，画中一位银色长发、蓝色眼睛、肤色红润、身穿蓝色吊带连衣裙的少女漂浮在水下，面向镜头，周围是光彩的气泡，和煦的阳光透过水面折射进水下"
negative_prompt = "半身，苍白的肤色，蜡黄的肤色，尸体，错误的眼睛，糟糕的人脸，毁容，糟糕的艺术，变形，多余的肢体，模糊的颜色，模糊，重复，病态，残缺，错误的手指，口红，腮红"

torch.manual_seed(7)
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,
    cfg_scale=4,
)
image.save(f"image_1024.jpg")

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    input_image=image.resize((2048, 2048)), denoising_strength=0.4, height=2048, width=2048,
    num_inference_steps=50,
    cfg_scale=4,
)
image.save("image_2048.jpg")
