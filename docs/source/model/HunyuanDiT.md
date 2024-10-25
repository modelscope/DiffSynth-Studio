# Hunyuan-DiT

## 相关链接

* 论文：[Hunyuan-DiT : A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding](https://arxiv.org/pdf/2405.08748)
* 模型
    * HunyuanDiT
        * [HuggingFace](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT)
        * [ModelScope](https://modelscope.cn/models/modelscope/HunyuanDiT)
    * HunyuanDiT-v1.1
        * [HuggingFace](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.1)
    * HunyuanDiT-v1.2
        * [HuggingFace](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2)
    * Distillation
        * [HuggingFace](https://huggingface.co/Tencent-Hunyuan/Distillation)
    * Distillation-v1.1
        * [HuggingFace](https://huggingface.co/Tencent-Hunyuan/Distillation-v1.1)
    * Distillation-v1.2
        * [HuggingFace](https://huggingface.co/Tencent-Hunyuan/Distillation-v1.2)
* 项目页面: https://dit.hunyuan.tencent.com/

## 模型介绍

Hunyuan-DiT是一种基于传统DiT架构的扩散模型, 为了加强模型对中文的细粒度(fine-grained)理解能力, Hunyuan-DiT对Transformer在多个方面进行了改进. 在类别条件(class-conditional)的 DiT 中使用的自适应层归一化(Adaptive Layer Norm)在强制执行细粒度文本条件方面表现不好, 为此Hunyuan-DiT采用了与Stable Diffusion 相似的交叉注意力机制. Hunyuan-DiT接受VAE潜在空间的向量作为输入, 将它分割成小块后经过线性层得到后续用于transformer块的标记. 在每个Hunyuan-DiT Block中包含三个模块, 自注意力(self-attention), 交叉注意力(cross-attention), 和前馈网络(feed-forward network, FFN).

![image](https://github.com/user-attachments/assets/50f3eb1f-855d-4095-88fb-c03711f4c7ae)

为了加强训练的稳定性, Hunyuan-DiT采用了QK-Norm, 在注意力层计算QKV前加入层归一化, 并且在decoder block的skip module后加入层归一化避免损失爆炸(loss explosion).

Hunyuan-DiT的生成效果:
![image](https://github.com/user-attachments/assets/4c11be16-c7ac-45a1-a900-b620606eb2c4)

## 代码样例

```python
from diffsynth import ModelManager, HunyuanDiTImagePipeline, download_models
import torch

download_models(["HunyuanDiT"])

model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models([
    "models/HunyuanDiT/t2i/clip_text_encoder/pytorch_model.bin",
    "models/HunyuanDiT/t2i/mt5/pytorch_model.bin",
    "models/HunyuanDiT/t2i/model/pytorch_model_ema.pt",
    "models/HunyuanDiT/t2i/sdxl-vae-fp16-fix/diffusion_pytorch_model.bin"
])
pipe = HunyuanDiTImagePipeline.from_model_manager(model_manager)

prompt = "一幅细致的油画描绘了一只年轻獾轻轻嗅着一朵明亮的黄色玫瑰时错综复杂的皮毛。背景是一棵大树干的粗糙纹理，獾的爪子轻轻地挖进树皮。在柔和的背景中，一个宁静的瀑布倾泻而下，它的水在绿色植物中闪烁着蓝色。"

torch.manual_seed(0)
image = pipe(
    prompt=prompt,
    num_inference_steps=50, height=1024, width=1024,
)
image.save("image_1024.png")
```