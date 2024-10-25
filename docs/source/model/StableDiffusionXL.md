# Stable Diffusion XL

## 相关链接

* 论文：[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2307.01952)
* 模型
    * stable-diffusion-xl-base-1.0
        * [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
        * [ModelScope](https://modelscope.cn/models/AI-ModelScope/stable-diffusion-xl-base-1.0)


## 模型介绍

Stable Diffusion XL 与之前版本的 Stable Diffusion 相比，将 UNet 主干网络增大了三倍，SDXL 使用了两个文本编码器：([OpenCLIP-ViT/G](https://github.com/mlfoundations/open_clip) 和 [CLIP-ViT/L](https://github.com/openai/CLIP/tree/main))，因此在 UNet 中增加了更多的注意力模块和更大的交叉注意力上下文。我们设计了多种新颖的条件方案，并在多种宽高比上训练SDXL。同时 SDXL 引入了一个精细化模型 ，在后处理阶段来提高SDXL生成样本的逼真度。

SXDL的模型结构如下：

![](https://github.com/user-attachments/assets/1f94bbe3-a2f4-410b-9f68-d500bf91b0f0)


## 代码样例

```python
from diffsynth import ModelManager, SDXLImagePipeline, download_models
import torch


# Download models (automatically)
# `models/stable_diffusion_xl/bluePencilXL_v200.safetensors`: [link](https://civitai.com/api/download/models/245614?type=Model&format=SafeTensor&size=pruned&fp=fp16)
download_models(["BluePencilXL_v200"])

# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models(["models/stable_diffusion_xl/bluePencilXL_v200.safetensors"])
pipe = SDXLImagePipeline.from_model_manager(model_manager)

prompt = "masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait,"
negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"

torch.manual_seed(0)
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    cfg_scale=6,
    height=1024, width=1024, num_inference_steps=60,
)
image.save("1024.jpg")

```
