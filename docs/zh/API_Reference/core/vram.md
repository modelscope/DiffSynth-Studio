# `diffsynth.core.vram`: 显存管理

本文档介绍 `diffsynth.core.vram` 中的显存管理底层功能，如果你希望将这些功能用于其他的代码库中，可参考本文档。

## 跳过模型参数初始化

在 `PyTorch` 中加载模型时，模型的参数默认会占用显存或内存并进行参数初始化，而这些参数会在加载预训练权重后被覆盖掉，这导致了冗余的计算。`PyTorch` 中没有提供接口来跳过这些冗余的计算，我们在 `diffsynth.core.vram` 中提供了 `skip_model_initialization` 用于跳过模型参数初始化。

默认的模型加载方式：

```python
from diffsynth.core import load_state_dict
from diffsynth.models.qwen_image_controlnet import QwenImageBlockWiseControlNet

model = QwenImageBlockWiseControlNet() # Slow
path = "models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny/model.safetensors"
state_dict = load_state_dict(path, device="cpu")
model.load_state_dict(state_dict, assign=True)
```

跳过参数初始化的模型加载方式：

```python
from diffsynth.core import load_state_dict, skip_model_initialization
from diffsynth.models.qwen_image_controlnet import QwenImageBlockWiseControlNet

with skip_model_initialization():
    model = QwenImageBlockWiseControlNet() # Fast
path = "models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny/model.safetensors"
state_dict = load_state_dict(path, device="cpu")
model.load_state_dict(state_dict, assign=True)
```

在 `DiffSynth-Studio` 中，所有预训练模型都遵循这一加载逻辑。开发者在[接入模型](/docs/zh/Developer_Guide/Integrating_Your_Model.md)完毕后即可直接以这种方式快速加载模型。

## State Dict 硬盘映射

对于某个模型的预训练权重文件，如果我们只需要读取其中的一组参数，而非全部参数，State Dict 硬盘映射可以加速这一过程。我们在 `diffsynth.core.vram` 中提供了 `DiskMap` 用于按需加载模型参数。

默认的权重加载方式：

```python
from diffsynth.core import load_state_dict

path = "models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny/model.safetensors"
state_dict = load_state_dict(path, device="cpu") # Slow
print(state_dict["img_in.weight"])
```

使用 `DiskMap` 只加载特定参数：

```python
from diffsynth.core import DiskMap

path = "models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny/model.safetensors"
state_dict = DiskMap(path, device="cpu") # Fast
print(state_dict["img_in.weight"])
```

`DiskMap` 是 `DiffSynth-Studio` 中 Disk Offload 的基本组件，开发者在[配置细粒度显存管理方案](/docs/zh/Developer_Guide/Enabling_VRAM_management.md)后即可直接启用 Disk Offload。

`DiskMap` 是利用 `.safetensors` 文件的特性实现的功能，因此在使用 `.bin`、`.pth`、`.ckpt` 等二进制文件时，模型的参数是全量加载的，这也导致 Disk Offload 不支持这些格式的文件。**我们不建议开发者继续使用这些格式的文件。**

## 显存管理可替换模块

在启用 `DiffSynth-Studio` 的显存管理后，模型内部的模块会被替换为 `diffsynth.core.vram.layers` 中的可替换模块，其使用方式详见[细粒度显存管理方案](/docs/zh/Developer_Guide/Enabling_VRAM_management.md#编写细粒度显存管理方案)。
