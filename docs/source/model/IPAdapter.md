# IP-Adapter

## 相关链接

* 论文：
    * [IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models](https://arxiv.org/abs/2308.06721)
* 模型
    * IP-Adapter-SDXL
        * [HuggingFace](https://huggingface.co/h94/IP-Adapter)
        * [ModelScope](https://www.modelscope.cn/models/AI-ModelScope/IP-Adapter)

## 模型介绍

IP-Adapter 与 ControlNet 技术类似，是一种通过添加辅助性模型架构，为模型添加额外的控制条件的方法。与 ControlNet 类似的是，IP-Adapter 的额外控制条件也是图像输入，不同的是，IP-Adapter 的额外控制条件是通过 Cross-Attention 的方式加入到原始模型中的。

IP-Adapter 模型结构如下图所示。不考虑最上层的图像控制条件时，文本特征的信息通过 Cross-Attention 结构被加入到 Denoising U-Net 中，这就是典型的文生图 Pipeline。IP-Adapter 参考这一条件控制的范式，加入了图像控制条件。 对于一个控制图像，首先使用 Image Encoder 提取图像特征，然后使用投影网络将其映射为一个长度为 $N$ 的特征序列。在这个时候，图像特征与文本特征的特征形式已经相近，作者便使用同样的 Cross-Attention 结构来融合这一图像特征到 U-Net 中。 Image Encoder 采用经过预训练的 CLIP 模型，投影网络由一个线性层和层归一化组成，投影后的图像特征序列长度取 $N=4$。

![](https://github.com/user-attachments/assets/5ebe45a4-6877-41fe-a2e5-deb2ea33dfdb)


为了不破坏文生图基础模型的文本控制能力，IP-Adapter 采用了文图解耦的 Cross-Attention 结构，即冻结原本的文本 Cross-Attention，加入额外的图像 Cross-Attention 结构。解耦的 Cross-Attention 公式如下所示，其中 $K$ 和 $V$ 是文本的 Key 和 Value 向量， $K^{\prime}$ 和 $V^{\prime}$ 是图像的 Key 和 Value 向量。由于两个 Attention 的 Query 向量是一样的，只需要添加两个映射矩阵 $W_{K^{\prime}}$ 和 $W_{V^{\prime}}$ 作为可学习参数，这两个参数分别从 $W_{K}$ 和 $W_{V}$ 初始化而来。

$$
\mathbf{Z}^{\text {new }}=\operatorname{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^{\top}}{\sqrt{d}}\right) \mathbf{V}+\operatorname{Softmax}\left(\frac{\mathbf{Q}\left(\mathbf{K}^{\prime}\right)^{\top}}{\sqrt{d}}\right) \mathbf{V}^{\prime}
$$

综上所述，IP-Adapter 只有投影网络和部分 Cross-Attenion 参数是可学习的，一共只有 22M 可学习参数量。

## 代码样例

以下代码为 IP-Adapter-SDXL 模型的使用样例，我们使用[皮卡丘](https://github.com/user-attachments/assets/4b750148-0238-4c3c-b58c-355dc7fde8f8)作为图像控制条件，生成超人的图像如下：

![](https://github.com/user-attachments/assets/9338f4cf-aac1-4dc0-a307-d184b31133a0)

``` python
from diffsynth import ModelManager, SDXLImagePipeline, download_models
import torch
from PIL import Image
download_models(["BluePencilXL_v200", "IP-Adapter-SDXL"])

# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models([
    "models/stable_diffusion_xl/bluePencilXL_v200.safetensors",
    "models/IpAdapter/stable_diffusion_xl/image_encoder/model.safetensors",
    "models/IpAdapter/stable_diffusion_xl/ip-adapter_sdxl.bin"
])
pipe = SDXLImagePipeline.from_model_manager(model_manager)

image_pikachu = Image.open('Pikachu.png').convert("RGB").resize((1024, 1024))

torch.manual_seed(1)
print("Generating image...")
image = pipe(
    prompt="A super man",
    negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
    cfg_scale=5,
    height=1024, width=1024, num_inference_steps=50,
    ipadapter_images=[image_pikachu], ipadapter_use_instant_style=False
)
image.save(f"PikaSuperMan.jpg")
```