from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline, ModelConfig
import torch
from PIL import Image

pipe = QwenImage21Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="vae/diffusion_pytorch_model*.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="processor/"),
)

# Text-to-Image, the output is an RGBA image
prompt = "完全透明背景，无背景，alpha 通道抠图，PNG 透明贴纸风格，边缘干净利落：水下少女精致肖像，蓝裙在水中飘逸，发丝轻扬，面容恬静，人物周围环绕少量气泡，光影只作用于人物本身，除人物与气泡外没有任何背景元素，细节精致，梦幻唯美。"
image = pipe(prompt, seed=0, num_inference_steps=50)
image.save("image1.png")

prompt = "精致肖像，阳光帅哥，剑眉星目，轮廓分明，发型利落，浅笑温柔，光影柔和，气质出众，细节精致，梦幻唯美。"
image_2 = pipe(prompt=prompt, seed=0, num_inference_steps=50)
image_2.save("image2.png")

# Image Editing, the generated RGBA image is fed back as the condition
prompt = "生成这两个人的合影"
edit_image = [Image.open("image1.png"), Image.open("image2.png")]
image_3 = pipe(prompt, edit_image=edit_image, seed=1, num_inference_steps=50)
image_3.save("image3.png")
