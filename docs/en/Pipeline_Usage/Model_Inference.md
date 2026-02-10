# Model Inference

This document uses the Qwen-Image model as an example to introduce how to use `DiffSynth-Studio` for model inference.

## Loading Models

Models are loaded through `from_pretrained`:

```python
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
```

Where `torch_dtype` and `device` are computation precision and computation device (not model precision and device). `model_configs` can be configured in multiple ways for model paths. For how models are loaded internally in this project, please refer to [`diffsynth.core.loader`](/docs/en/API_Reference/core/loader.md).

<details>

<summary>Download and load models from remote sources</summary>

> `DiffSynth-Studio` downloads and loads models from [ModelScope](https://www.modelscope.cn/) by default. You need to fill in `model_id` and `origin_file_pattern`, for example:
> 
> ```python
> ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
> ```
> 
> Model files are downloaded to the `./models` path by default, which can be modified through [environment variable DIFFSYNTH_MODEL_BASE_PATH](/docs/en/Pipeline_Usage/Environment_Variables.md#diffsynth_model_base_path).

</details>

<details>

<summary>Load models from local file paths</summary>

> Fill in `path`, for example:
> 
> ```python
> ModelConfig(path="models/xxx.safetensors")
> ```
> 
> For models loaded from multiple files, use a list, for example:
> 
> ```python
> ModelConfig(path=[
>     "models/Qwen/Qwen-Image/text_encoder/model-00001-of-00004.safetensors",
>     "models/Qwen/Qwen-Image/text_encoder/model-00002-of-00004.safetensors",
>     "models/Qwen/Qwen-Image/text_encoder/model-00003-of-00004.safetensors",
>     "models/Qwen/Qwen-Image/text_encoder/model-00004-of-00004.safetensors",
> ])
> ```

</details>

By default, even after models have been downloaded, the program will still query remotely for missing files. To completely disable remote requests, set [environment variable DIFFSYNTH_SKIP_DOWNLOAD](/docs/en/Pipeline_Usage/Environment_Variables.md#diffsynth_skip_download) to `True`.

```shell
import os
os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
import diffsynth
```

To download models from [HuggingFace](https://huggingface.co/), set [environment variable DIFFSYNTH_DOWNLOAD_SOURCE](/docs/en/Pipeline_Usage/Environment_Variables.md#diffsynth_download_source) to `huggingface`.

```shell
import os
os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"
import diffsynth
```

## Starting Inference

Input a prompt to start the inference process and generate an image.

```python
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
prompt = "Exquisite portrait, underwater girl, blue dress flowing, hair floating, translucent light, bubbles surrounding, peaceful face, intricate details, dreamy and ethereal."
image = pipe(prompt, seed=0, num_inference_steps=40)
image.save("image.jpg")
```

Each model `Pipeline` has different input parameters. Please refer to the documentation for each model.

If the model parameters are too large, causing insufficient VRAM, please enable [VRAM management](/docs/en/Pipeline_Usage/VRAM_management.md).

## Loading LoRA

LoRA is a lightweight model training method that produces a small number of parameters to extend model capabilities. DiffSynth-Studio supports two ways to load LoRA: cold loading and hot loading.

* Cold loading: When the base model does not have [VRAM management](/docs/en/Pipeline_Usage/VRAM_management.md) enabled, LoRA will be fused into the base model weights. In this case, inference speed remains unchanged, but LoRA cannot be unloaded after loading.

```python
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
lora = ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-LoRA-ArtAug-v1", origin_file_pattern="model.safetensors")
pipe.load_lora(pipe.dit, lora, alpha=1)
prompt = "Exquisite portrait, underwater girl, blue dress flowing, hair floating, translucent light, bubbles surrounding, peaceful face, intricate details, dreamy and ethereal."
image = pipe(prompt, seed=0, num_inference_steps=40)
image.save("image.jpg")
```

* Hot loading: When the base model has [VRAM management](/docs/en/Pipeline_Usage/VRAM_management.md) enabled, LoRA will not be fused into the base model weights. In this case, inference speed will be slower, but LoRA can be unloaded through `pipe.clear_lora()` after loading.

```python
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cuda",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cuda",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
lora = ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-LoRA-ArtAug-v1", origin_file_pattern="model.safetensors")
pipe.load_lora(pipe.dit, lora, alpha=1)
prompt = "Exquisite portrait, underwater girl, blue dress flowing, hair floating, translucent light, bubbles surrounding, peaceful face, intricate details, dreamy and ethereal."
image = pipe(prompt, seed=0, num_inference_steps=40)
image.save("image.jpg")
pipe.clear_lora()
```
