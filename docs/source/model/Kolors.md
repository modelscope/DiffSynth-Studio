# Kolors

## 相关链接

* 论文：[Kolors: Effective Training of Diffusion Model for Photorealistic Text-to-Image Synthesis](https://github.com/Kwai-Kolors/Kolors/blob/master/imgs/Kolors_paper.pdf)
* 模型
    * Kolors
        * [HuggingFace](https://huggingface.co/Kwai-Kolors/Kolors)
        * [ModelScope](https://modelscope.cn/models/Kwai-Kolors/Kolors)
    
* 项目页面: https://kwai-kolors.github.io/

## 模型介绍

Kolors是一种用于文本生成图像的潜在扩散模型, 使用了General Language Model（GLM）作为文本编码器, 增强了它的中英文理解能力. Kolors有两个训练阶段, 包括概念学习阶段（使用广泛的知识）和质量提升阶段（使用精心整理的高美学数据）, 并且在质量提升阶段使用1100步的调度器添加噪声, 以达到更低的信噪比. 这些改动使得即使Kolors以U-Net作为骨干模型, 也能达到好的效果.
![image](https://github.com/user-attachments/assets/d6b91d41-3d88-4d26-a399-03ca180640cf)

kolors的生成效果:
![kolors](https://github.com/user-attachments/assets/f6926507-52e2-471d-87ab-a9351338e4ca)


## 代码样例

```python
from diffsynth import ModelManager, SDXLImagePipeline, download_models
import torch

download_models(["Kolors"])
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda",
                             file_path_list=[
                                 "models/kolors/Kolors/text_encoder",
                                 "models/kolors/Kolors/unet/diffusion_pytorch_model.safetensors",
                                 "models/kolors/Kolors/vae/diffusion_pytorch_model.safetensors"
                             ])
pipe = SDXLImagePipeline.from_model_manager(model_manager)

prompt = '一张瓢虫的照片，微距，变焦，高质量，电影，拿着一个牌子，写着"Kolors"'

torch.manual_seed(7)
image = pipe(
    prompt=prompt,
    num_inference_steps=50,
    cfg_scale=4,
)
image.save(f"image_1024.jpg")

```