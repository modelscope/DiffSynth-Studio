
# 基于Flux的文生图示例

以下是如何使用FLUX.1模型进行文生图任务的示例。该脚本提供了一个简单的设置，用于从文本描述生成图像。包括下载必要的模型、配置pipeline，以及在启用和禁用 classifier-free guidance 的情况下生成图像。

其他 DiffSynth 支持的模型详见 [模型.md](模型.md)

## 准备

首先，确保已下载并配置了必要的模型：

```python
import torch
from diffsynth import ModelManager, FluxImagePipeline, download_models

# Download the FLUX.1-dev model files
download_models(["FLUX.1-dev"])
```

下载模型的用法详见 [下载模型.md](下载模型.md)

## 加载模型

使用您的设备和数据类型初始化模型管理器

```python
model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda")
model_manager.load_models([
    "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
    "models/FLUX/FLUX.1-dev/text_encoder_2",
    "models/FLUX/FLUX.1-dev/ae.safetensors",
    "models/FLUX/FLUX.1-dev/flux1-dev.safetensors"
])
```

模型加载的用法详见 [ModelManager.md](ModelManager.md)

## 创建 Pipeline

从加载的模型管理器中创建FluxImagePipeline实例：

```python
pipe = FluxImagePipeline.from_model_manager(model_manager)
```

Pipeline 的用法详见 [Pipeline.md](Pipeline.md)

## 文生图

使用简短的提示语生成图像。以下是启用和禁用 classifier-free guidance 的图像生成示例。

### 基础文生图

```python
prompt = "A cute little turtle"
negative_prompt = ""

torch.manual_seed(6)
image = pipe(
    prompt=prompt,
    num_inference_steps=30, embedded_guidance=3.5
)
image.save("image_1024.jpg")
```

### 使用 Classifier-Free Guidance 生成
```python
torch.manual_seed(6)
image = pipe(
    prompt=prompt, negative_prompt=negative_prompt,
    num_inference_steps=30, cfg_scale=2.0, embedded_guidance=3.5
)
image.save("image_1024_cfg.jpg")
```

### 高分辨率修复

```python
torch.manual_seed(7)
image = pipe(
    prompt=prompt,
    num_inference_steps=30, embedded_guidance=3.5,
    input_image=image.resize((2048, 2048)), height=2048, width=2048, denoising_strength=0.6, tiled=True
)
image.save("image_2048_highres.jpg")
```

