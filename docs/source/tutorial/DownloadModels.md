# 下载模型

我们在 DiffSynth-Studio 中预置了一些主流 Diffusion 模型的下载链接，你可以下载并使用这些模型。

## 下载预置模型

你可以直接使用 `download_models` 函数下载预置的模型文件，其中模型 ID 可参考 [config file](https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/configs/model_config.py)。

```python
from diffsynth import download_models

download_models(["FLUX.1-dev"])
```

对于 VSCode 用户，激活 Pylance 或其他 Python 语言服务后，在代码中输入 `""` 即可显示支持的所有模型 ID。

![image](https://github.com/user-attachments/assets/2bbfec32-e015-45a7-98d9-57af13200b7c)

## 下载非预置模型

你可以选择 [ModelScope](https://modelscope.cn/models) 和 [HuggingFace](https://huggingface.co/models) 两个下载源中的模型。当然，你也可以通过浏览器等工具选择手动下载自己所需的模型。

```python
from diffsynth import download_customized_models

download_customized_models(
    model_id="Kwai-Kolors/Kolors",
    origin_file_path="vae/diffusion_pytorch_model.fp16.bin",
    local_dir="models/kolors/Kolors/vae",
    downloading_priority=["ModelScope", "HuggingFace"]
)
```

在这段代码中，我们将会按照下载的优先级，优先从 `ModelScope` 下载，在 ID 为 `Kwai-Kolors/Kolors` 的[模型库](https://modelscope.cn/models/Kwai-Kolors/Kolors)中，把文件 `vae/diffusion_pytorch_model.fp16.bin` 下载到本地的路径 `models/kolors/Kolors/vae` 中。
