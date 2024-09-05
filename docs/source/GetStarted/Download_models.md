# 下载模型

下载预设模型，模型ID可参考 [config file](/diffsynth/configs/model_config.py).

```python
from diffsynth import download_models

download_models(["FLUX.1-dev", "Kolors"])
```

下载非预设模型，可以选择 [ModelScope](https://modelscope.cn/models) 和 [HuggingFace](https://huggingface.co/models) 两个下载源中的模型。

```python
from diffsynth.models.downloader import download_from_huggingface, download_from_modelscope

# From Modelscope (recommended)
download_from_modelscope("Kwai-Kolors/Kolors", "vae/diffusion_pytorch_model.fp16.bin", "models/kolors/Kolors/vae")
# From Huggingface
download_from_huggingface("Kwai-Kolors/Kolors", "vae/diffusion_pytorch_model.fp16.safetensors", "models/kolors/Kolors/vae")
```
