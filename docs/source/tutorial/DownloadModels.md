# 下载模型

我们在 DiffSynth-Studio 中预置了一些主流 Diffusion 模型的下载链接，你可以轻松地下载并使用这些模型。

## 下载预置模型

你可以直接使用 `download_models` 函数下载预置的模型文件，其中模型 ID 可参考 [config file](/diffsynth/configs/model_config.py)。

```python
from diffsynth import download_models

download_models(["FLUX.1-dev"])
```

对于 VSCode 用户，激活 Pylance 或其他 Python 语言服务后，在代码中输入 `""` 即可显示支持的所有模型 ID。

![image](https://github.com/user-attachments/assets/2bbfec32-e015-45a7-98d9-57af13200b7c)

## 下载非预置模型

你可以选择 [ModelScope](https://modelscope.cn/models) 和 [HuggingFace](https://huggingface.co/models) 两个下载源中的模型。当然，你也可以通过浏览器等工具选择手动下载自己所需的模型。

```python
from diffsynth.models.downloader import download_from_huggingface, download_from_modelscope

# From Modelscope (recommended)
download_from_modelscope("Kwai-Kolors/Kolors", "vae/diffusion_pytorch_model.fp16.bin", "models/kolors/Kolors/vae")
# From Huggingface
download_from_huggingface("Kwai-Kolors/Kolors", "vae/diffusion_pytorch_model.fp16.safetensors", "models/kolors/Kolors/vae")
```
