# Stable Diffusion

## 相关链接

* 论文：[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
* 模型
    * stable-diffusion-v1-1
        * [HuggingFace](https://huggingface.co/CompVis/stable-diffusion-v1-1)
    * stable-diffusion-v1-2
        * [HuggingFace](https://huggingface.co/CompVis/stable-diffusion-v1-2)
    * stable-diffusion-v1-3
        * [HuggingFace](https://huggingface.co/CompVis/stable-diffusion-v1-3)
    * stable-diffusion-v1-4
        * [HuggingFace](https://huggingface.co/CompVis/stable-diffusion-v1-4)
        * [ModelScope](https://modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-4)
    * stable-diffusion-v1-5
        * [HuggingFace](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
        * [ModelScope](https://modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5)

这里仅提供 Stable Diffusion 官方开源的模型。由于Stable Diffusion 是一个完全免费开源的模型，且能让用户使用消费级显卡实现快速文生图，因此有非常多基于 Stable Diffusion 训练的优秀模型涌现出来，DiffSynth 支持主流开源社区的  Stable Diffusion 模型的训练与推理。

## 模型介绍

Stable Diffusion 是一种基于扩散模型的文本到图像生成技术，它最初由 [Stability AI](https://stability.ai/) 和 [LAION](https://laion.ai/) 基于 [LAION-5B](https://laion.ai/blog/laion-5b/) 的子集，对 512*512 的图像训练了一个 latent diffusion model，使用 CLIP ViT-L/14 文本编码器编码文本作为模型的提示。

扩散模型（DMs）在图像数据及其他领域达到了最先进的合成效果，但是由于直接在像素空间进行加噪和去噪过程，训练和推理时需要大量计算资源，为了在有限的计算资源下训练扩散模型，同时保留其质量和灵活性，Stable Diffusion 在预训练自动编码器的潜在空间 (Latent Space) 中训练扩散模型。

与之前的工作相比，在这种在潜空间表示上训练扩散模型达到了低复杂性和空间下采样之间的近乎最佳平衡，大大提升了视觉保真度。通过将交叉注意力层引入模型架构，扩散模型被转变为功能强大的灵活生成器，可以用于文本或边界框等一般条件输入，并通过卷积方式实现高分辨率合成。

Stable Diffusion 在各种任务上表现出极具竞争力的性能，包括无条件图像生成、图像修复和超分辨率，同时相较于基于像素的扩散模型显著降低了计算需求。

Stable Diffusion 的模型结构如下图所示，通过交叉注意力来实现条件控制。

![](https://github.com/user-attachments/assets/9d383abe-2889-4ceb-bc0a-136228b809c8)


## 代码样例

```python
from diffsynth import ModelManager, SDXLImagePipeline, download_models
import torch


# Download models (automatically)

# `models/stable_diffusion/aingdiffusion_v12.safetensors`: [link](https://civitai.com/api/download/models/229575?type=Model&format=SafeTensor&size=full&fp=fp16)

download_models(["AingDiffusion_v12"])

# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models(["models/stable_diffusion/aingdiffusion_v12.safetensors"])
pipe = SDImagePipeline.from_model_manager(model_manager)

prompt = "masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait,"
negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"

torch.manual_seed(0)
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    cfg_scale=6,
    height=512, width=512, num_inference_steps=60,
)
image.save("1024.jpg")
```
