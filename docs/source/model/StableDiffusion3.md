# Stable Diffusion 3

## 相关链接

* 论文：[Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/pdf/2403.03206)
* 模型
    * stable-diffusion-3-medium
        * [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-3-medium)
        * [ModelScope](https://modelscope.cn/models/AI-ModelScope/stable-diffusion-3-medium)
    
* 项目页面: https://stability.ai/news/stable-diffusion-3-medium

## 模型介绍

Stable Diffusion 3（SD3）是 Stability AI 的文生图开源模型，在基于人类偏好的评估中，Stable Diffusion 3 在文字生成图像的性能上超过了目前最先进的系统，包括 DALL·E 3、Midjourney v6 和 Ideogram v1，并在文字内容生成，复杂提示理解和指令遵循方面的性能有显著提升。SD3 采用了全新的多模态扩散变压器（MMDiT）架构，使用不同的权重集来处理图像和语言表示，提高了模型的文本理解和拼写能力。

最大的 SD3 模型拥有 80 亿参数，可以装入拥有 24GB VRAM 的 RTX 4090 中，使用 50 次采样步骤生成一张 1024x1024 分辨率的图像仅需 34 秒。此外，还发布了多种版本的，参数范围从8亿到80亿的 Stable Diffusion 3。

![image](https://github.com/user-attachments/assets/e6d95a9e-cd0a-4438-a564-0754eb4c10e1)

MMDiT 架构使用三种不同的文本嵌入器（两个 CLIP 模型和 T5）来编码文本表示，并使用改进的自动编码模型来编码图像，然后将结合两种模态的序列拼接起来尽进行注意力操作。相比传统的文本生成图像网络，这种架构在视觉保真度和文本对齐度的训练过程中表现更佳。通过该方法，信息可以在图像和文本之间流动，进而提高生成内容的整体理解能力和视觉设计，同时其设计也容易扩展到视频等多种模态的应用。

此外，SD3 引入了改进的校正流（RF）公式，使得在训练过程中，数据和噪声可以沿着更直的线性轨迹连接，从而减少了采样步骤。通过对采样计划的重加权，尤其是在中间部分，提升了模型的预测任务性能。与其他 60 种扩散轨迹（例如LDM 、 EDM 和 ADM ））相比，重加权的RF变体在性能上具有更优越的表现。

在文本编码方面，尽管在推理过程中将拥有 4.7B 参数的 T5 文本编码器排除在外减少了内存需求并略微影响性能，但这对视觉美学无大影响，只是稍微降低了提示文本的遵循性。为了充分发挥文本生成能力，尤其是在处理复杂提示文本的场景中，建议保留 T5 文本编码器。

Stable Diffusion 3 的生成效果：

![image](https://github.com/user-attachments/assets/1b5b0260-6421-47fb-abe7-de6758f4721f)


## 代码样例

```python
from diffsynth import ModelManager, SD3ImagePipeline, download_models
import torch

download_models(["StableDiffusion3_without_T5"])
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda",
                             file_path_list=["models/stable_diffusion_3/sd3_medium_incl_clips.safetensors"])
pipe = SD3ImagePipeline.from_model_manager(model_manager)


prompt = "masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait,"
negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"

torch.manual_seed(7)
image = pipe(
    prompt=prompt, 
    negative_prompt=negative_prompt,
    cfg_scale=7.5,
    num_inference_steps=100, width=1024, height=1024,
)
image.save("image_1024.jpg")
```
