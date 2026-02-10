# 模型推理

本文档以 Qwen-Image 模型为例，介绍如何使用 `DiffSynth-Studio` 进行模型推理。

## 加载模型

模型通过 `from_pretrained` 加载：

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

其中 `torch_dtype` 和 `device` 是计算精度和计算设备（不是模型的精度和设备）。`model_configs` 可通过多种方式配置模型路径，关于本项目内部是如何加载模型的，请参考 [`diffsynth.core.loader`](/docs/zh/API_Reference/core/loader.md)。

<details>

<summary>从远程下载模型并加载</summary>

> `DiffSynth-Studio` 默认从[魔搭社区](https://www.modelscope.cn/)下载并加载模型，需填写 `model_id` 和 `origin_file_pattern`，例如
> 
> ```python
> ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
> ```
> 
> 模型文件默认下载到 `./models` 路径，该路径可通过[环境变量 DIFFSYNTH_MODEL_BASE_PATH](/docs/zh/Pipeline_Usage/Environment_Variables.md#diffsynth_model_base_path) 修改。

</details>

<details>

<summary>从本地文件路径加载模型</summary>

> 填写 `path`，例如
> 
> ```python
> ModelConfig(path="models/xxx.safetensors")
> ```
> 
> 对于从多个文件加载的模型，使用列表即可，例如
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

默认情况下，即使模型已经下载完毕，程序仍会向远程查询是否有遗漏文件，如果要完全关闭远程请求，请将[环境变量 DIFFSYNTH_SKIP_DOWNLOAD](/docs/zh/Pipeline_Usage/Environment_Variables.md#diffsynth_skip_download) 设置为 `True`。

```shell
import os
os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
import diffsynth
```

如需从 [HuggingFace](https://huggingface.co/) 下载模型，请将[环境变量 DIFFSYNTH_DOWNLOAD_SOURCE](/docs/zh/Pipeline_Usage/Environment_Variables.md#diffsynth_download_source) 设置为 `huggingface`。

```shell
import os
os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"
import diffsynth
```

## 启动推理

输入提示词，即可启动推理过程，生成一张图片。

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
prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
image = pipe(prompt, seed=0, num_inference_steps=40)
image.save("image.jpg")
```

每个模型 `Pipeline` 的输入参数不同，请参考各模型的文档。

如果模型参数量太大，导致显存不足，请开启[显存管理](/docs/zh/Pipeline_Usage/VRAM_management.md)。

## 加载 LoRA

LoRA 是一种轻量化的模型训练方式，产生少量参数，扩展模型的能力。DiffSynth-Studio 的 LoRA 加载有两种方式：冷加载和热加载。

* 冷加载：当基础模型未开启[显存管理](/docs/zh/Pipeline_Usage/VRAM_management.md)时，LoRA 会融合进基础模型权重，此时推理速度没有变化，LoRA 加载后无法卸载。

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
prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
image = pipe(prompt, seed=0, num_inference_steps=40)
image.save("image.jpg")
```

* 热加载：当基础模型开启[显存管理](/docs/zh/Pipeline_Usage/VRAM_management.md)时，LoRA 不会融合进基础模型权重，此时推理速度会变慢，LoRA 加载后可通过 `pipe.clear_lora()` 卸载。

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
prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
image = pipe(prompt, seed=0, num_inference_steps=40)
image.save("image.jpg")
pipe.clear_lora()
```
