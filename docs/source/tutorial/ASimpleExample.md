# 快速开始

在这篇文档中，我们通过一段代码为你介绍如何快速上手使用 DiffSynth-Studio 进行创作。

## 安装

使用以下命令从 GitHub 克隆并安装 DiffSynth-Studio。更多信息请参考[安装](./Installation.md)。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

## 一键运行！

通过运行以下代码，我们将会下载模型、加载模型、生成图像。

```python
import torch
from diffsynth import ModelManager, FluxImagePipeline

model_manager = ModelManager(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_id_list=["FLUX.1-dev"]
)
pipe = FluxImagePipeline.from_model_manager(model_manager)

torch.manual_seed(0)
image = pipe(
    prompt="In a forest, a wooden plank sign reading DiffSynth",
    height=576, width=1024,
)
image.save("image.jpg")
```

![image](https://github.com/user-attachments/assets/15a52a2b-2f18-46fe-810c-cb3ad2853919)

从这个例子中，我们可以看到，DiffSynth 中有两个关键模块：`ModelManager` 和 `Pipeline`，接下来我们详细介绍。

## 下载和加载模型

`ModelManager` 负责下载和加载模型，通过以下代码可以直接一步完成。

```python
import torch
from diffsynth import ModelManager

model_manager = ModelManager(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_id_list=["FLUX.1-dev"]
)
```

当然，我们也支持分步完成，以下代码和上述代码的行为是等价的。

```python
import torch
from diffsynth import download_models, ModelManager

download_models(["FLUX.1-dev"])
model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda")
model_manager.load_models([
    "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
    "models/FLUX/FLUX.1-dev/text_encoder_2",
    "models/FLUX/FLUX.1-dev/ae.safetensors",
    "models/FLUX/FLUX.1-dev/flux1-dev.safetensors"
])
```

下载模型时，我们支持从 [ModelScope](https://www.modelscope.cn/) 和 [HuggingFace](https://huggingface.co/) 下载模型，也支持下载非预置的模型，关于模型下载的更多信息请参考[模型下载](./DownloadModels.md)。

加载模型时，你可以把所有想要加载的模型路径放入其中。对于 `.safetensors` 等格式的模型权重文件，`ModelManager` 在加载后会自动判断模型类型；对于文件夹格式的模型，`ModelManager` 会尝试解析其中的 `config.json` 文件并尝试调用 `transformers` 等第三方库中的对应模块。关于 DiffSynth-Studio 支持的模型，请参考[支持的模型](./Models.md)。

## 构建 Pipeline

DiffSynth-Studio 提供了多个推理 `Pipeline`，这些 `Pipeline` 可以直接通过 `ModelManager` 获取所需的模型并初始化。例如，FLUX.1-dev 模型的文生图 `Pipeline` 可以这样构建：

```python
pipe = FluxImagePipeline.from_model_manager(model_manager)
```

更多用于图像生成和视频生成的 `Pipeline` 详见[推理流水线](./Pipelines.md)。
