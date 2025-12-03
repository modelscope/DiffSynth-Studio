# `diffsynth.core.loader`: 模型下载与加载

本文档介绍 `diffsynth.core.loader` 中模型下载与加载相关的功能。

## ModelConfig

`diffsynth.core.loader` 中的 `ModelConfig` 用于标注模型下载来源、本地路径、显存管理配置等信息。

### 从远程下载并加载模型

以模型[DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny) 为例，在 `ModelConfig` 中填写 `model_id` 和 `origin_file_pattern` 后即可自动下载模型。默认下载到 `./models` 路径，该路径可通过[环境变量 DIFFSYNTH_MODEL_BASE_PATH](/docs/zh/Pipeline_Usage/Environment_Variables.md#diffsynth_model_base_path) 修改。

默认情况下，即使模型已经下载完毕，程序仍会向远程查询是否有遗漏文件，如果要完全关闭远程请求，请将[环境变量 DIFFSYNTH_SKIP_DOWNLOAD](/docs/zh/Pipeline_Usage/Environment_Variables.md#diffsynth_skip_download) 设置为 `True`。

```python
from diffsynth.core import ModelConfig

config = ModelConfig(
    model_id="DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny",
    origin_file_pattern="model.safetensors",
)
# Download models
config.download_if_necessary()
print(config.path)
```

调用 `download_if_necessary` 后，模型会自动下载，并将路径返回到 `config.path` 中。

### 从本地路径加载模型

如果从本地路径加载模型，则需要填入 `path`：

```python
from diffsynth.core import ModelConfig

config = ModelConfig(path="models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny/model.safetensors")
```

如果模型包含多个分片文件，以列表的形式输入即可：

```python
from diffsynth.core import ModelConfig

config = ModelConfig(path=[
    "models/Qwen/Qwen-Image/text_encoder/model-00001-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00002-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00003-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00004-of-00004.safetensors"
])
```

### 显存管理配置

`ModelConfig` 也包含了显存管理配置信息，详见[显存管理](/docs/zh/Pipeline_Usage/VRAM_management.md#更多使用方式)。

## 模型文件加载

`diffsynth.core.loader` 提供了统一的 `load_state_dict`，用于加载模型文件中的 state dict。

加载单个模型文件：

```python
from diffsynth.core import load_state_dict

state_dict = load_state_dict("models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny/model.safetensors")
```

加载多个模型文件（合并为一个 state dict）：

```python
from diffsynth.core import load_state_dict

state_dict = load_state_dict([
    "models/Qwen/Qwen-Image/text_encoder/model-00001-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00002-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00003-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00004-of-00004.safetensors"
])
```

## 模型哈希

模型哈希是用于判断模型类型的，哈希值可通过 `hash_model_file` 获取：

```python
from diffsynth.core import hash_model_file

print(hash_model_file("models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny/model.safetensors"))
```

也可计算多个模型文件的哈希值，等价于合并 state dict 后计算模型哈希值：

```python
from diffsynth.core import hash_model_file

print(hash_model_file([
    "models/Qwen/Qwen-Image/text_encoder/model-00001-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00002-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00003-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00004-of-00004.safetensors"
]))
```

模型哈希值只与模型文件中 state dict 的 keys 和 tensor shape 有关，与模型参数的数值、文件保存时间等信息无关。在计算 `.safetensors` 格式文件的模型哈希值时，`hash_model_file` 是几乎瞬间完成的，无需读取模型的参数；但在计算 `.bin`、`.pth`、`.ckpt` 等二进制文件的模型哈希值时，则需要读取全部模型参数，因此**我们不建议开发者继续使用这些格式的文件。**

通过[编写模型 Config](/docs/zh/Developer_Guide/Integrating_Your_Model.md#step-3-编写模型-config)并将模型哈希值等信息填入 `diffsynth/configs/model_configs.py`，开发者可以让 `DiffSynth-Studio` 自动识别模型类型并加载。

## 模型加载

`load_model` 是 `diffsynth.core.loader` 中加载模型的外部入口，它会调用 [skip_model_initialization](/docs/zh/API_Reference/core/vram.md#跳过模型参数初始化) 跳过模型参数初始化。如果启用了 [Disk Offload](/docs/zh/Pipeline_Usage/VRAM_management.md#disk-offload)，则调用 [DiskMap](/docs/zh/API_Reference/core/vram.md#state-dict-硬盘映射) 进行惰性加载；如果没有启用 Disk Offload，则调用 [load_state_dict](#模型文件加载) 加载模型参数。如果需要的话，还会调用 [state dict converter](/docs/zh/Developer_Guide/Integrating_Your_Model.md#step-2-模型文件格式转换) 进行模型格式转换。最后调用 `model.eval()` 将其切换到推理模式。

以下是一个启用了 Disk Offload 的使用案例：

```python
from diffsynth.core import load_model, enable_vram_management, AutoWrappedLinear, AutoWrappedModule
from diffsynth.models.qwen_image_dit import QwenImageDiT, RMSNorm
import torch

prefix = "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model"
model_path = [prefix + f"-0000{i}-of-00009.safetensors" for i in range(1, 10)]

model = load_model(
    QwenImageDiT,
    model_path,
    module_map={
        torch.nn.Linear: AutoWrappedLinear,
        RMSNorm: AutoWrappedModule,
    },
    vram_config={
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": "disk",
        "onload_device": "disk",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    },
    vram_limit=0,
)
```
