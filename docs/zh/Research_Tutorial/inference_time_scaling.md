# 推理时扩展

DiffSynth-Studio 支持推理时扩展（Inference-time Scaling）技术，具体实现了 **Spectral Evolution Search (SES)** 算法。该技术允许用户在推理阶段通过增加计算量来提升生成图像质量，而无需重新训练模型。

## 1. 基本原理

传统的文生图推理过程是从一个随机的高斯噪声开始，经过固定的去噪步数生成图像。这种方式生成的质量高度依赖于初始噪声的随机性。

**SES (Spectral Evolution Search)** 将推理过程转化为一个针对初始噪声的**搜索优化问题**：

1. **搜索空间**：在小波变换的频域空间中搜索初始噪声的最低频部分。
2. **进化策略**：使用交叉熵方法迭代采样噪声群体。
3. **奖励反馈**：利用 PickScore 等奖励模型对生成的低步数预览图进行评分。
4. **结果输出**：找到得分最高的噪声，进行完整的高质量去噪。

这种方法本质上是用**推理计算时间换取生成质量**。

关于该方法的更多技术细节，请参考论文：**[Spectral Evolution Search: Efficient Inference-Time Scaling for Reward-Aligned Image Generation](https://arxiv.org/abs/2602.03208)**。

## 2. 快速开始

在 DiffSynth-Studio 中，SES 已集成到主流文生图模型的 Pipeline 中。你只需在调用 `pipe()` 时设置 `enable_ses=True` 即可开启。

以下是以 **Z-Image-Turbo** 为例的[快速上手代码](../../../examples/z_image/model_inference/Z-Image-Turbo-SES.py)：

```python
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
import torch

pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
)

prompt = "Chinese woman in red Hanfu holding a fan, with a bright yellow neon lightning bolt floating above her palm. Soft-lit outdoor night background, silhouetted tiered pagoda, blurred colorful distant lights."

image = pipe(
    prompt=prompt, 
    seed=42, 
    rand_device="cuda",
    enable_ses=True,
    ses_reward_model="pick",
    ses_eval_budget=50,
    ses_inference_steps=8
)
image.save("image_Z-Image-Turbo_ses.jpg")
```

## 3. 支持的模型与参数

### 3.1 核心参数详解

在 `pipe()` 调用中，以下参数控制 SES 的行为：

| 参数名 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `enable_ses` | `bool` | `False` | 是否开启 SES 优化。 |
| `ses_reward_model` | `str` | `"pick"` | 奖励模型选择。支持 `"pick"` (PickScore), `"hps"` (HPSv2), `"clip"`。 |
| `ses_eval_budget` | `int` | `50` | 搜索的总预算（评估样本总数）。数值越高，质量上限越高，但耗时越长。 |
| `ses_inference_steps` | `int` | `10` | 搜索阶段生成预览图使用的步数。数值越高，对于候选噪声的质量评估越准确，但耗时越长，建议设为 8～15 。 |

### 3.2 支持模型列表

目前以下文生图模型均已支持 SES：

* **[Qwen-Image](../../../examples/qwen_image/model_inference/Qwen-Image-SES.py)**
* **[FLUX.1-dev](../../../examples/flux/model_inference/FLUX.1-dev-SES.py)**
* **[FLUX.2-dev](../../../examples/flux2/model_inference/FLUX.2-dev-SES.py)**
* **[Z-Image](../../../examples/z_image/model_inference/Z-Image-SES.py) / [Z-Image-Turbo](../../../examples/z_image/model_inference/Z-Image-Turbo-SES.py)**


## 4. 效果展示

随着搜索预算（`ses_eval_budget`）的增加，SES 能够稳定地提升图像质量。以下展示了在相同随机种子下，不同计算预算带来的质量变化。

**场景 1：Qwen-Image**

* **Prompt**: *"Springtime in the style of Paul Delvaux"*
* **Reward Model**: PickScore

| **Budget = 0** | **Budget = 10** | **Budget = 30** | **Budget = 50** |
| --- | --- | --- | --- |
|  |  |  |  |
| <img width="220" alt="Image" src="https://github.com/user-attachments/assets/250a8c18-d086-49ed-98dc-5eebc5234231" /> | <img width="220" alt="Image" src="https://github.com/user-attachments/assets/3d4b9ecf-36bc-4f63-81fe-e0be9526f103" /> | <img width="220" alt="Image" src="https://github.com/user-attachments/assets/1ed69cca-bd76-43da-940b-b8da49b5a693" /> | <img width="220" alt="Image" src="https://github.com/user-attachments/assets/bd887d83-fc78-4a1c-abb9-be814aefa8f9" /> |

**场景 2：FLUX.1-dev**

* **Prompt**: *"A masterful painting of a young woman in the style of Diego Velázquez."*
* **Reward Model**: HPSv2

| **Budget = 0** | **Budget = 10** | **Budget = 30** | **Budget = 50** |
| --- | --- | --- | --- |
|  |  |  |  |
| <img width="220" alt="Image" src="https://github.com/user-attachments/assets/c34a89b8-2f5a-420d-ad23-844a2befd074" /> | <img width="220" alt="Image" src="https://github.com/user-attachments/assets/ba9cb54a-1b50-4ada-b636-237e71383617" /> | <img width="220" alt="Image" src="https://github.com/user-attachments/assets/e3327a88-a903-4c30-8ee7-9e35fb117cc4" /> | <img width="220" alt="Image" src="https://github.com/user-attachments/assets/45e5faa2-e69d-4ffc-9400-e80c0a78f1bd" /> |