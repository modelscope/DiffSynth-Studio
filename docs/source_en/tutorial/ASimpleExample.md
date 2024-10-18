# Quick Start

In this document, we introduce how to quickly get started with DiffSynth-Studio for creation through a piece of code.

## Installation

Use the following command to clone and install DiffSynth-Studio from GitHub. For more information, please refer to [Installation](./Installation.md).

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

## One-click Run!

By running the following code, we will download the model, load the model, and generate an image.

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

From this example, we can see that there are two key modules in DiffSynth: `ModelManager` and `Pipeline`. We will introduce them in detail next.

## Downloading and Loading Models

`ModelManager` is responsible for downloading and loading models, which can be done in one step with the following code.

```python
import torch
from diffsynth import ModelManager

model_manager = ModelManager(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_id_list=["FLUX.1-dev"]
)
```

Of course, we also support completing this step by step, and the following code is equivalent to the above.

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

When downloading models, we support downloading from [ModelScope](https://www.modelscope.cn/) and [HuggingFace](https://huggingface.co/), and we also support downloading non-preset models. For more information about model downloading, please refer to [Model Download](./DownloadModels.md).

When loading models, you can put all the model paths you want to load into it. For model weight files in formats such as `.safetensors`, `ModelManager` will automatically determine the model type after loading; for folder format models, `ModelManager` will try to parse the `config.json` file within and try to call the corresponding module in third-party libraries such as `transformers`. For models supported by DiffSynth-Studio, please refer to [Supported Models](./Models.md).

## Building Pipeline

DiffSynth-Studio provides multiple inference `Pipeline`s, which can be directly obtained through `ModelManager` to get the required models and initialize. For example, the text-to-image `Pipeline` for the FLUX.1-dev model can be constructed as follows:

```python
pipe = FluxImagePipeline.from_model_manager(model_manager)
```

For more `Pipeline`s used for image generation and video generation, see [Inference Pipelines](./Pipelines.md).
