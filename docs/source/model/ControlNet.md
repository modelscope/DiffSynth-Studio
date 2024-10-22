# ControlNet

## 相关链接

* 论文：
    * [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)
* 模型
    * ControlNet-Union-SDXL
        * [HuggingFace](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0)
        * [ModelScope](https://modelscope.cn/models/AI-ModelScope/controlnet-union-sdxl-1.0)
    * ControlNet-V11-SD15
        * [HuggingFace](https://huggingface.co/lllyasviel/ControlNet-v1-1)
        * [ModelScope](https://modelscope.cn/models/AI-ModelScope/ControlNet-v1-1)

## 模型介绍
ControlNet 是一种辅助性的模型架构，它能够与已经训练好的 Diffusion 模型相结合。通过给模型额外添加可训练的 ControlNet 模块，我们得以在图像生成过程中施加额外的控制条件。比如，我们可以加入深度图、语义图和人体关键点等额外条件，控制生成图像的画面结构和布局。值得注意的是，针对不同的 Diffusion 模型， ControlNet 的具体结构可能会有所差异。

### ControlNet-V11-SD15
ControlNet V1.1 是基于 Stable Diffusion V1.5 (SD15) 的 ControlNet 更新版本，包含 Canny, Depth, Segmentation, Inpaint, Lineart 等控制条件对应的模型。

ControlNet 原论文是针对 SD15 设计的模型结构，如下图所示。(a) 部分结构为已训练完成的 Stable Diffusion (SD) 模型，模型输入为文本 Prompt $c_t$ 与去噪时间步长 $t$。(b) 部分结构为 ControlNet，主要包括若干个零初始化的卷积层 (zero convolution) 和 SD UNet Encoder 的可训练副本，模型输入为额外的控制条件 $c_f$。

zero convolution 为 $1\times1$ 的卷积层，其权重和偏置都被初始化为0。因此，在 ControlNet 被训练之前，所有 zero convolution 模块的输出都为0，保证了 ControlNet 的输出也为0，从而不会改变 SD 模型的输出。注意，zero convolution 的权重和偏置初始化为0并不会导致其梯度也为0，因此这些卷积层是能被训练的。

ControlNet 中的可训练副本采用与 SD UNet Encoder Blocks 相同的结构，并以其与训练好的权重作为初始化。而 SD 模型本身的所有参数都处于冻结状态。在训练过程中，只有 ControlNet 的参数会进行更新。因此，我们既能通过 ControlNet 的对额外的控制条件进行学习训练，又不会破坏 SD 模型本身的能力。

给定 SD 模型参数 $\Theta$， ControlNet 参数 $\Theta_{\mathrm{c}}$， 两个 zero convolution 模块 $\Theta_{\mathrm{z1}}$ 和 $\Theta_{\mathrm{z2}}$， 模型的输出如下。

$$
\boldsymbol{y}_{\mathrm{c}}=\mathcal{F}(\boldsymbol{x} ; \Theta)+\mathcal{Z}\left(\mathcal{F}\left(\boldsymbol{x}+\mathcal{Z}\left(\boldsymbol{c} ; \Theta_{\mathrm{z} 1}\right) ; \Theta_{\mathrm{c}}\right) ; \Theta_{\mathrm{z2}}\right)
$$

![](https://github.com/user-attachments/assets/dfe2e032-1ff8-4835-b061-ffa746ab1406)

ControlNet 生成图像示例如下所示：
![](https://github.com/user-attachments/assets/b0a122b7-2610-465e-9d01-6237c3fbe0f0)

## ControlNet++
ControlNet++ 是针对 Stable Diffusion XL (SDXL) 模型设计的 ControlNet 结构，对应上文提到的 ControlNet-Union-SDXL 模型。这一模型能同时支持10多种控制条件，包括 Pose，Depth，Canny，Lineart 等。

模型结构如下图所示。相比于 ControlNet ，这一模型扩充了 Condition Encoder 的卷积通道数量，同时增加了两个新模块，分别是 Condition Transformer 和 Control Encoder。Condition Transformer 用于组合不同的图像条件特征，而 Control Encoder 则用于编码控制条件的类型。

![](https://github.com/user-attachments/assets/96c9c4e7-ed0a-49cc-8307-a6f024166e68)


## 代码样例

以下代码为 ControlNet-Union-SDXL 模型的使用样例，其中使用的 [image.jpg](https://github.com/user-attachments/assets/cc094e8f-ff6a-4f9e-ba05-7a5c2e0e609f) 为 SDXL 生成的图像，详见[精准控制技术文档](https://diffsynth-studio.readthedocs.io/zh-cn/latest/creating/AdaptersForImageSynthesis.html)

```python
from diffsynth import ModelManager, SDXLImagePipeline, ControlNetConfigUnit
import torch
from PIL import Image

model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models([
    "models/stable_diffusion_xl/bluePencilXL_v200.safetensors",
    "models/ControlNet/controlnet_union/diffusion_pytorch_model_promax.safetensors"
])
pipe = SDXLImagePipeline.from_model_manager(model_manager, controlnet_config_units=[
    ControlNetConfigUnit("depth", "models/ControlNet/controlnet_union/diffusion_pytorch_model_promax.safetensors", scale=1)
])
torch.manual_seed(2)
image = pipe(
    prompt="masterpiece, best quality, solo, long hair, wavy hair, pink hair, red eyes, red dress, medium breasts, dress, fire ball, fire background, floating hair, refraction, portrait,",
    negative_prompt="worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw, white background",
    cfg_scale=6, num_inference_steps=60,
    controlnet_image=Image.open("image.jpg")
)
image.save("image_controlnet.jpg")
```
生成效果：

![image_controlnet](https://github.com/user-attachments/assets/d50d173e-e81a-4d7e-93e3-b2787d69953e)
