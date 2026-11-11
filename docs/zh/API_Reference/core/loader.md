# `diffsynth.core.loader`: 模型下载与加载

本文档介绍 `diffsynth.core.loader` 中模型下载与加载相关的功能。

## ModelConfig

`diffsynth.core.loader` 中的 `ModelConfig` 用于标注模型下载来源、本地路径、显存管理配置等信息。

### 从远程下载并加载模型

以模型[DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny) 为例，在 `ModelConfig` 中填写 `model_id` 和 `origin_file_pattern` 后即可自动下载模型。默认下载到 `./models` 路径，该路径可通过[环境变量 DIFFSYNTH_MODEL_BASE_PATH](../../Pipeline_Usage/Environment_Variables.md#diffsynth_model_base_path) 修改。

默认情况下，即使模型已经下载完毕，程序仍会向远程查询是否有遗漏文件，如果要完全关闭远程请求，请将[环境变量 DIFFSYNTH_SKIP_DOWNLOAD](../../Pipeline_Usage/Environment_Variables.md#diffsynth_skip_download) 设置为 `True`。

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

### 加载预量化模型

预量化模型通常会随权重一起发布一个 JSON 文件，用于描述重建模型所需的信息（即 `MODEL_CONFIGS` 中包含 `quant_config` 的条目）。在 `ModelConfig` 中用 `origin_file_pattern` 填写权重文件，用 `origin_config_file` 填写该 JSON 文件；调用 `download_if_necessary` 后，JSON 文件会被下载到 `config.config_file`，其中的条目也会自动注册，此后该预量化模型可以像普通模型一样加载：

```python
from diffsynth.core import ModelConfig

config = ModelConfig(
    model_id="DiffSynth-Studio/MiniMax-H3-Image-EntroPack",
    origin_file_pattern="models/dit_extreme_3bpp.safetensors",
    origin_config_file="models/dit_extreme_3bpp.json",
)
config.download_if_necessary()
print(config.path, config.config_file)
```

如果该 JSON 文件已经在本地，则填入 `config_file`，并在加载模型前调用 `config.register_prequantized()`。

#### 配置文件格式

该 JSON 文件包含一个对象，即一个 `MODEL_CONFIGS` 条目：

| 字段 | 是否必填 | 说明 |
| --- | --- | --- |
| `model_hash` | 是 | 量化权重文件的 hash，由 [`hash_model_file`](#模型哈希) 计算（张量名与形状的 MD5）。加载器用它识别模型类型，必须与权重文件一致。 |
| `model_name` | 是 | 模型名称，pipeline 用它取用模型，例如 `qwen_image_dit`。 |
| `model_class` | 是 | 模型类的导入路径，例如 `diffsynth.models.qwen_image_dit.QwenImageDiT`。 |
| `quant_config` | 是 | 传递给 [`QuantizeConfig`](../../Pipeline_Usage/Quantization.md) 的参数，其中 `load_prequantized` 必须为 `true`，其余字段需与保存权重时的量化配置一致。 |
| `state_dict_converter` | 否 | 权重格式转换器的导入路径，详见[模型集成](../../Developer_Guide/Integrating_Your_Model.md)。 |
| `extra_kwargs` | 否 | 模型构造函数的额外参数。 |

只使用一种量化方法时，`quant_config` 直接填写该方法及其参数：

```json
{
    "model_hash": "8c6f5ccfe9ffe861675ca0da95bb9295",
    "model_name": "minimax_h3_text_encoder",
    "model_class": "diffsynth.models.minimax_h3_text_encoder.MiniMaxH3TextEncoder",
    "quant_config": {
        "method": "entropack_lossy_quant",
        "load_prequantized": true,
        "backend_config_kwargs": {"target_bpp": 4.0, "row_rdo_iterations": 2, "row_rdo_candidates": 20}
    }
}
```

混合量化时用 `configs` 代替 `method`，每一项是一个 `QuantizeConfig`，通过 `target_modules` / `exclude_modules` 选择它负责的模块；所有子项需使用相同的 `mode`，`load_prequantized` 只写在外层：

```json
{
    "model_hash": "adab44d835193ce1b2bd0f85f0586f0e",
    "model_name": "minimax_h3_image_dit",
    "model_class": "diffsynth.models.minimax_h3_image_dit.MiniMaxH3ImageDiTPruned",
    "quant_config": {
        "load_prequantized": true,
        "configs": [
            {"method": "entropack_lossy_quant", "backend_config_kwargs": {"target_bpp": 2.91}, "exclude_modules": ["video_patch_proj", "condition_proj"]},
            {"method": "entropack_lossy_quant", "backend_config_kwargs": {"target_bpp": 8.5}, "target_modules": ["video_patch_proj", "condition_proj"]}
        ]
    }
}
```

文件名没有限制，通常按权重文件名命名（如 `model.safetensors` 对应 `model.json`）。如果 `MODEL_CONFIGS` 中已经存在相同 `model_hash` 的条目，文件中的条目会被忽略。

### 显存管理配置

`ModelConfig` 也包含了显存管理配置信息，详见[显存管理](../../Pipeline_Usage/VRAM_management.md#更多使用方式)。

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

通过[编写模型 Config](../../Developer_Guide/Integrating_Your_Model.md#step-3-编写模型-config)并将模型哈希值等信息填入 `diffsynth/configs/model_configs.py`，开发者可以让 `DiffSynth-Studio` 自动识别模型类型并加载。

## 模型加载

`load_model` 是 `diffsynth.core.loader` 中加载模型的外部入口，它会调用 [skip_model_initialization](../../API_Reference/core/vram.md#跳过模型参数初始化) 跳过模型参数初始化。如果启用了 [Disk Offload](../../Pipeline_Usage/VRAM_management.md#disk-offload)，则调用 [DiskMap](../../API_Reference/core/vram.md#state-dict-硬盘映射) 进行惰性加载；如果没有启用 Disk Offload，则调用 [load_state_dict](#模型文件加载) 加载模型参数。如果需要的话，还会调用 [state dict converter](../../Developer_Guide/Integrating_Your_Model.md#step-2-模型文件格式转换) 进行模型格式转换。最后调用 `model.eval()` 将其切换到推理模式。

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
