# 接入模型结构

本文档介绍如何将模型接入到 `DiffSynth-Studio` 框架中，供 `Pipeline` 等模块调用。

## Step 1: 集成模型结构代码

`DiffSynth-Studio` 中的所有模型结构实现统一在 `diffsynth/models` 中，每个 `.py` 代码文件分别实现一个模型结构，所有模型通过 `diffsynth/models/model_loader.py` 中的 `ModelPool` 来加载。在接入新的模型结构时，请在这个路径下建立新的 `.py` 文件。

```shell
diffsynth/models/
├── general_modules.py
├── model_loader.py
├── qwen_image_controlnet.py
├── qwen_image_dit.py
├── qwen_image_text_encoder.py
├── qwen_image_vae.py
└── ...
```

在大多数情况下，我们建议用 `PyTorch` 原生代码的形式集成模型，让模型结构类直接继承 `torch.nn.Module`，例如：

```python
import torch

class NewDiffSynthModel(torch.nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)
        self.activation = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x
```

如果模型结构的实现中包含额外的依赖，我们强烈建议将其删除，否则这会导致沉重的包依赖问题。在我们现有的模型中，Qwen-Image 的 Blockwise ControlNet 是以这种方式集成的，其代码很轻量，请参考 `diffsynth/models/qwen_image_controlnet.py`。

如果模型已被 Huggingface Library （[`transformers`](https://huggingface.co/docs/transformers/main/index)、[`diffusers`](https://huggingface.co/docs/diffusers/main/index) 等）集成，我们能够以更简单的方式集成模型：

<details>
<summary>集成 Huggingface Library 风格模型结构代码</summary>

这类模型在 Huggingface Library 中的加载方式为：

```python
from transformers import XXX_Model

model = XXX_Model.from_pretrained("path_to_your_model")
```

`DiffSynth-Studio` 不支持通过 `from_pretrained` 加载模型，因为这与显存管理等功能是冲突的，请将模型结构改写成以下格式：

```python
import torch

class DiffSynth_XXX_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import XXX_Config, XXX_Model
        config = XXX_Config(**{
            "architectures": ["XXX_Model"],
            "other_configs": "Please copy and paste the other configs here.",
        })
        self.model = XXX_Model(config)
        
    def forward(self, x):
        outputs = self.model(x)
        return outputs
```

其中 `XXX_Config` 为模型对应的 Config 类，例如 `Qwen2_5_VLModel` 的 Config 类为 `Qwen2_5_VLConfig`，可通过查阅其源代码找到。Config 内部的内容通常可以在模型库中的 `config.json` 中找到，`DiffSynth-Studio` 不会读取 `config.json` 文件，因此需要将其中的内容复制粘贴到代码中。

在少数情况下，`transformers` 和 `diffusers` 的版本更新会导致部分的模型无法导入，因此如果可能的话，我们仍建议使用 Step 1.1 中的模型集成方式。

在我们现有的模型中，Qwen-Image 的 Text Encoder 是以这种方式集成的，其代码很轻量，请参考 `diffsynth/models/qwen_image_text_encoder.py`。

</details>

## Step 2: 模型文件格式转换

由于开源社区中开发者提供的模型文件格式多种多样，因此我们有时需对模型文件格式进行转换，从而形成格式正确的 [state dict](https://docs.pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html)，常见于以下几种情况：

* 模型文件由不同代码库构建，例如 [Wan-AI/Wan2.1-T2V-1.3B](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B) 和 [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)。
* 模型在接入中做了修改，例如 [Qwen/Qwen-Image](https://www.modelscope.cn/models/Qwen/Qwen-Image) 的 Text Encoder 在 `diffsynth/models/qwen_image_text_encoder.py` 中增加了 `model.` 前缀。
* 模型文件包含多个模型，例如 [Wan-AI/Wan2.1-VACE-14B](https://www.modelscope.cn/models/Wan-AI/Wan2.1-VACE-14B) 的 VACE Adapter 和基础 DiT 模型混合存储在同一组模型文件中。

在我们的开发理念中，我们希望尽可能尊重模型原作者的意愿。如果对模型文件进行重新封装，例如 [Comfy-Org/Qwen-Image_ComfyUI](https://www.modelscope.cn/models/Comfy-Org/Qwen-Image_ComfyUI)，虽然我们可以更方便地调用模型，但流量（模型页面浏览量和下载量等）会被引向他处，模型的原作者也会失去删除模型的权力。因此，我们在框架中增加了 `diffsynth/utils/state_dict_converters` 这一模块，用于在模型加载过程中进行文件格式转换。

这部分逻辑是非常简单的，以 Qwen-Image 的 Text Encoder 为例，只需要 10 行代码即可：

```python
def QwenImageTextEncoderStateDictConverter(state_dict):
    state_dict_ = {}
    for k in state_dict:
        v = state_dict[k]
        if k.startswith("visual."):
            k = "model." + k
        elif k.startswith("model."):
            k = k.replace("model.", "model.language_model.")
        state_dict_[k] = v
    return state_dict_
```

## Step 3: 编写模型 Config

模型 Config 位于 `diffsynth/configs/model_configs.py`，用于识别模型类型并进行加载。需填入以下字段：

* `model_hash`：模型文件哈希值，可通过 `hash_model_file` 函数获取，此哈希值仅与模型文件中 state dict 的 keys 和张量 shape 有关，与文件中的其他信息无关。
* `model_name`: 模型名称，用于给 `Pipeline` 识别所需模型。如果不同结构的模型在 `Pipeline` 中发挥的作用相同，则可以使用相同的 `model_name`。在接入新模型时，只需保证 `model_name` 与现有的其他功能模型不同即可。在 `Pipeline` 的 `from_pretrained` 中通过 `model_name` 获取对应的模型。
* `model_class`: 模型结构导入路径，指向在 Step 1 中实现的模型结构类，例如 `diffsynth.models.qwen_image_text_encoder.QwenImageTextEncoder`。
* `state_dict_converter`: 可选参数，如需进行模型文件格式转换，则需填入模型转换逻辑的导入路径，例如 `diffsynth.utils.state_dict_converters.qwen_image_text_encoder.QwenImageTextEncoderStateDictConverter`。
* `extra_kwargs`: 可选参数，如果模型初始化时需传入额外参数，则需要填入这些参数，例如模型 [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny) 与 [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint) 都采用了 `diffsynth/models/qwen_image_controlnet.py` 中的 `QwenImageBlockWiseControlNet` 结构，但后者还需额外的配置 `additional_in_dim=4`，因此这部分配置信息需填入 `extra_kwargs` 字段。

我们提供了一份代码，以便快速理解模型是如何通过这些配置信息加载的：

```python
from diffsynth.core import hash_model_file, load_state_dict, skip_model_initialization
from diffsynth.models.qwen_image_text_encoder import QwenImageTextEncoder
from diffsynth.utils.state_dict_converters.qwen_image_text_encoder import QwenImageTextEncoderStateDictConverter
import torch

model_hash = "8004730443f55db63092006dd9f7110e"
model_name = "qwen_image_text_encoder"
model_class = QwenImageTextEncoder
state_dict_converter = QwenImageTextEncoderStateDictConverter
extra_kwargs = {}

model_path = [
    "models/Qwen/Qwen-Image/text_encoder/model-00001-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00002-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00003-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00004-of-00004.safetensors",
]
if hash_model_file(model_path) == model_hash:
    with skip_model_initialization():
        model = model_class(**extra_kwargs)
    state_dict = load_state_dict(model_path, torch_dtype=torch.bfloat16, device="cuda")
    state_dict = state_dict_converter(state_dict)
    model.load_state_dict(state_dict, assign=True)
    print("Done!")
```

> Q: 上述代码的逻辑看起来很简单，为什么 `DiffSynth-Studio` 中的这部分代码极为复杂？
> 
> A: 因为我们提供了激进的显存管理功能，与模型加载逻辑耦合，这导致框架结构的复杂性，我们已尽可能简化暴露给开发者的接口。

`diffsynth/configs/model_configs.py` 中的 `model_hash` 不是唯一存在的，同一模型文件中可能存在多个模型。对于这种情况，请使用多个模型 Config 分别加载每个模型，编写相应的 `state_dict_converter` 分离每个模型所需的参数。

## Step 4: 检验模型是否能被识别和加载

模型接入之后，可通过以下代码验证模型是否能够被正确识别和加载，以下代码会试图将模型加载到内存中：

```python
from diffsynth.models.model_loader import ModelPool

model_pool = ModelPool()
model_pool.auto_load_model(
    [
        "models/Qwen/Qwen-Image/text_encoder/model-00001-of-00004.safetensors",
        "models/Qwen/Qwen-Image/text_encoder/model-00002-of-00004.safetensors",
        "models/Qwen/Qwen-Image/text_encoder/model-00003-of-00004.safetensors",
        "models/Qwen/Qwen-Image/text_encoder/model-00004-of-00004.safetensors",
    ],
)
```

如果模型能够被识别和加载，则会看到以下输出内容：

```
Loading models from: [
    "models/Qwen/Qwen-Image/text_encoder/model-00001-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00002-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00003-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00004-of-00004.safetensors"
]
Loaded model: {
    "model_name": "qwen_image_text_encoder",
    "model_class": "diffsynth.models.qwen_image_text_encoder.QwenImageTextEncoder",
    "extra_kwargs": null
}
```

## Step 5: 编写模型显存管理方案

`DiffSynth-Studio` 支持复杂的显存管理，详见[启用显存管理](/docs/zh/Developer_Guide/Enabling_VRAM_management.md)。
