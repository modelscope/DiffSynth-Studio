# 显存管理

显存管理是 `DiffSynth-Studio` 的特色功能，能够让低显存的 GPU 能够运行参数量巨大的模型推理。本文档以 Qwen-Image 为例，介绍显存管理方案的使用。

## 基础推理

以下代码中没有启用任何显存管理，显存占用 56G，作为参考。

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

## CPU Offload

由于模型 `Pipeline` 包括多个组件，这些组件并非同时调用的，因此我们可以在某些组件不需要参与计算时将其移至内存，减少显存占用，以下代码可以实现这一逻辑，显存占用 40G。

```python
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
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
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
image = pipe(prompt, seed=0, num_inference_steps=40)
image.save("image.jpg")
```

## FP8 量化

在 CPU Offload 的基础上，我们进一步启用 FP8 量化来减少显存需求，以下代码可以令模型参数以 FP8 精度存储在显存中，并在推理时临时转为 BF16 精度计算，显存占用 21G。但这种量化方案有微小的图像质量下降问题。

```python
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch

vram_config = {
    "offload_dtype": torch.float8_e4m3fn,
    "offload_device": "cpu",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cuda",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
image = pipe(prompt, seed=0, num_inference_steps=40)
image.save("image.jpg")
```

> Q: 为什么要在推理时临时转为 BF16 精度，而不是以 FP8 精度计算？
> 
> A: FP8 的原生计算仅在 Hopper 架构的 GPU（例如 H20）支持，且计算误差很大，我们目前暂不开放 FP8 精度计算。目前的 FP8 量化仅能减少显存占用，不会提高计算速度。

## 动态显存管理

在 CPU Offload 中，我们对模型组件进行控制，事实上，我们支持做到 Layer 级别的 Offload，将一个模型拆分为多个 Layer，令一部分常驻显存，令一部分存储在内存中按需移至显存计算。这一功能需要模型开发者针对每个模型提供详细的显存管理方案，相关配置在 `diffsynth/configs/vram_management_module_maps.py` 中。

通过在 `Pipeline` 中增加 `vram_limit` 参数，框架可以自动感知设备的剩余显存并决定如何拆分模型到显存和内存中。`vram_limit` 越小，占用显存越少，速度越慢。
* `vram_limit=None` 时，即默认状态，框架认为显存无限，动态显存管理是不启用的
* `vram_limit=10` 时，框架会在显存占用超过 10G 之后限制模型，将超出的部分移至内存中存储。
* `vram_limit=0` 时，框架会尽全力减少显存占用，所有模型参数都存储在内存中，仅在必要时移至显存计算

在显存不足以运行模型推理的情况下，框架会试图超出 `vram_limit` 的限制从而让模型推理运行下去，因此显存管理框架并不能总是保证占用的显存小于 `vram_limit`，我们建议将其设置为略小于实际可用显存的数值，例如 GPU 显存为 16G 时，设置为 `vram_limit=15.5`。`PyTorch` 中可用 `torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3)` 获取 GPU 的显存。

```python
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch

vram_config = {
    "offload_dtype": torch.float8_e4m3fn,
    "offload_device": "cpu",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
image = pipe(prompt, seed=0, num_inference_steps=40)
image.save("image.jpg")
```

## Disk Offload

在更为极端的情况下，当内存也不足以存储整个模型时，Disk Offload 功能可以让模型参数惰性加载，即，模型中的每个 Layer 仅在调用 forward 时才会从硬盘中读取相应的参数。启用这一功能时，我们建议使用高速的 SSD 硬盘。

Disk Offload 是极为特殊的显存管理方案，只支持 `.safetensors` 格式文件，不支持 `.bin`、`.pth`、`.ckpt` 等二进制文件，不支持带 Tensor reshape 的 [state dict converter](/docs/zh/Developer_Guide/Integrating_Your_Model.md#step-2-模型文件格式转换)。

```python
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
    vram_limit=10,
)
prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
image = pipe(prompt, seed=0, num_inference_steps=40)
image.save("image.jpg")
```

## 更多使用方式

`vram_config` 中的信息可自行填写，例如不开 FP8 量化的 Disk Offload：

```python
vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
```

具体地，显存管理模块会将模型的 Layer 分为以下四种状态：

* Offload：短期内不调用这个模型，这个状态由 `Pipeline` 控制切换
* Onload：接下来随时要调用这个模型，这个状态由 `Pipeline` 控制切换
* Preparing：Onload 和 Computation 的中间状态，在显存允许的前提下的暂存状态，这个状态由显存管理机制控制切换，当且仅当【vram_limit 设置为无限制】或【vram_limit 已设置且有空余显存】时会进入这一状态
* Computation：模型正在计算过程中，这个状态由显存管理机制控制切换，仅在 `forward` 中临时进入

如果你是模型开发者，希望自行控制某个模型的显存管理粒度，请参考[../Developer_Guide/Enabling_VRAM_management.md](/docs/zh/Developer_Guide/Enabling_VRAM_management.md)。

## 最佳实践

* 显存足够 -> 使用[基础推理](#基础推理)
* 显存不足
    * 内存足够 -> 使用[动态显存管理](#动态显存管理)
    * 内存不足 -> 使用[Disk Offload](#disk-offload)
