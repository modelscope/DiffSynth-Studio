# Download Models

We have preset some mainstream Diffusion model download links in DiffSynth-Studio, which you can download and use.

## Download Preset Models

You can directly use the `download_models` function to download the preset model files, where the model ID can refer to the [config file](/diffsynth/configs/model_config.py).

```python
from diffsynth import download_models

download_models(["FLUX.1-dev"])
```

For VSCode users, after activating Pylance or other Python language services, typing `""` in the code will display all supported model IDs.

![image](https://github.com/user-attachments/assets/2bbfec32-e015-45a7-98d9-57af13200b7c)

## Download Non-Preset Models

You can select models from two download sources: [ModelScope](https://modelscope.cn/models) and [HuggingFace](https://huggingface.co/models). Of course, you can also manually download the models you need through browsers or other tools.

```python
from diffsynth import download_customized_models

download_customized_models(
    model_id="Kwai-Kolors/Kolors",
    origin_file_path="vae/diffusion_pytorch_model.fp16.bin",
    local_dir="models/kolors/Kolors/vae",
    downloading_priority=["ModelScope", "HuggingFace"]
)
```

In this code snippet, we will prioritize downloading from `ModelScope` according to the download priority, and download the file `vae/diffusion_pytorch_model.fp16.bin` from the model repository with ID `Kwai-Kolors/Kolors` in the [model library](https://modelscope.cn/models/Kwai-Kolors/Kolors) to the local path `models/kolors/Kolors/vae`.
