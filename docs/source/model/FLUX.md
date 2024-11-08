# FLUX

## 相关链接

* 技术报告：https://blackforestlabs.ai/announcing-black-forest-labs/
* 模型
    * FLUX.1-dev
        * [HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-dev)
        * [ModelScope](https://modelscope.cn/models/AI-ModelScope/FLUX.1-dev)
    
* 项目页面: https://github.com/black-forest-labs/flux

## 模型介绍

FLUX.1  是由 The Black Forest Team 发布的一系列文生图模型，该模型在图像细节、提示一致性、风格多样性和文本到图像合成的场景复杂性方面定义了新的最先进技术。FLUX.1 提供了三个变体：FLUX.1 [pro]、FLUX.1 [dev] 和 FLUX.1 [schnell]，我们在这里用到的是从 FLUX.1 [pro] 蒸馏出来的用于非商业应用的开放权重的 FLUX.1 [dev]。
FLUX.1 模型均基于多模态和并行扩散变压器块的混合架构，并可缩放至 12B 参数。通过建立流匹配来改进以前最先进的扩散模型，流匹配是一种通用且概念上简单的训练生成模型的方法，其中包括作为特殊情况的扩散。此外，通过结合旋转位置嵌入和并行注意层来提高模型性能并提高硬件效率。

FLUX.1 定义了图像合成领域的最新技术，FLUX.1 [pro] 和 [dev] 在以下各个方面超越了 Midjourney v6.0、DALL·E 3 (HD) 和 SD3-Ultra 等流行模型：视觉质量、提示跟随、尺寸/方面可变性、版式和输出多样性。 FLUX.1 [schnell] 是迄今为止最先进的几步模型，其性能不仅优于同类竞争对手，而且还优于 Midjourney v6.0 和 DALL·E 3 (HD) 等强大的非蒸馏模型。FLUX.1 经过专门微调，以保留预训练的整个输出多样性。与当前最先进的技术相比，它们提供了极大改进的可能性，如下所示：

![image](https://github.com/user-attachments/assets/cff34a82-6f5d-4b6d-9c30-d7959d3ef2fc)

Flux 的生成效果：

![image](https://github.com/user-attachments/assets/68f4888e-0574-402a-ac7a-362198a7b867)

## 代码样例

```python
import torch
from diffsynth import ModelManager, FluxImagePipeline, download_models


download_models(["FLUX.1-dev"])
model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda")
model_manager.load_models([
    "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
    "models/FLUX/FLUX.1-dev/text_encoder_2",
    "models/FLUX/FLUX.1-dev/ae.safetensors",
    "models/FLUX/FLUX.1-dev/flux1-dev.safetensors"
])
pipe = FluxImagePipeline.from_model_manager(model_manager)

prompt = "CG. Full body. A captivating fantasy magic woman portrait in the deep sea. The woman, with blue spaghetti strap silk dress, swims in the sea. Her flowing silver hair shimmers with every color of the rainbow and cascades down, merging with the floating flora around her. Smooth, delicate and fair skin."
negative_prompt = "dark, worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, dim, fuzzy, depth of Field, nsfw,"

# Disable classifier-free guidance (consistent with the original implementation of FLUX.1)
torch.manual_seed(6)
image = pipe(
    prompt=prompt,
    num_inference_steps=30, embedded_guidance=3.5
)
image.save("image_1024.jpg")
```
