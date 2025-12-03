# 细粒度显存管理方案

本文档介绍如何为模型编写合理的细粒度显存管理方案，以及如何将 `DiffSynth-Studio` 中的显存管理功能用于外部的其他代码库，在阅读本文档前，请先阅读文档[显存管理](/docs/zh/Pipeline_Usage/VRAM_management.md)。

## 20B 模型需要多少显存？

以 Qwen-Image 的 DiT 模型为例，这一模型的参数量达到了 20B，以下代码会加载这一模型并进行推理，需要约 40G 显存，这个模型在显存较小的消费级 GPU 上显然是无法运行的。

```python
from diffsynth.core import load_model
from diffsynth.models.qwen_image_dit import QwenImageDiT
from modelscope import snapshot_download
import torch

snapshot_download(
    model_id="Qwen/Qwen-Image",
    local_dir="models/Qwen/Qwen-Image",
    allow_file_pattern="transformer/*"
)
prefix = "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model"
model_path = [prefix + f"-0000{i}-of-00009.safetensors" for i in range(1, 10)]
inputs = {
    "latents": torch.randn((1, 16, 128, 128), dtype=torch.bfloat16, device="cuda"),
    "timestep": torch.zeros((1,), dtype=torch.bfloat16, device="cuda"),
    "prompt_emb": torch.randn((1, 5, 3584), dtype=torch.bfloat16, device="cuda"),
    "prompt_emb_mask": torch.ones((1, 5), dtype=torch.int64, device="cuda"),
    "height": 1024,
    "width": 1024,
}

model = load_model(QwenImageDiT, model_path, torch_dtype=torch.bfloat16, device="cuda")
with torch.no_grad():
    output = model(**inputs)
```

## 编写细粒度显存管理方案

为了编写细粒度的显存管理方案，我们需用 `print(model)` 观察和分析模型结构：

```
QwenImageDiT(
  (pos_embed): QwenEmbedRope()
  (time_text_embed): TimestepEmbeddings(
    (time_proj): TemporalTimesteps()
    (timestep_embedder): DiffusersCompatibleTimestepProj(
      (linear_1): Linear(in_features=256, out_features=3072, bias=True)
      (act): SiLU()
      (linear_2): Linear(in_features=3072, out_features=3072, bias=True)
    )
  )
  (txt_norm): RMSNorm()
  (img_in): Linear(in_features=64, out_features=3072, bias=True)
  (txt_in): Linear(in_features=3584, out_features=3072, bias=True)
  (transformer_blocks): ModuleList(
    (0-59): 60 x QwenImageTransformerBlock(
      (img_mod): Sequential(
        (0): SiLU()
        (1): Linear(in_features=3072, out_features=18432, bias=True)
      )
      (img_norm1): LayerNorm((3072,), eps=1e-06, elementwise_affine=False)
      (attn): QwenDoubleStreamAttention(
        (to_q): Linear(in_features=3072, out_features=3072, bias=True)
        (to_k): Linear(in_features=3072, out_features=3072, bias=True)
        (to_v): Linear(in_features=3072, out_features=3072, bias=True)
        (norm_q): RMSNorm()
        (norm_k): RMSNorm()
        (add_q_proj): Linear(in_features=3072, out_features=3072, bias=True)
        (add_k_proj): Linear(in_features=3072, out_features=3072, bias=True)
        (add_v_proj): Linear(in_features=3072, out_features=3072, bias=True)
        (norm_added_q): RMSNorm()
        (norm_added_k): RMSNorm()
        (to_out): Sequential(
          (0): Linear(in_features=3072, out_features=3072, bias=True)
        )
        (to_add_out): Linear(in_features=3072, out_features=3072, bias=True)
      )
      (img_norm2): LayerNorm((3072,), eps=1e-06, elementwise_affine=False)
      (img_mlp): QwenFeedForward(
        (net): ModuleList(
          (0): ApproximateGELU(
            (proj): Linear(in_features=3072, out_features=12288, bias=True)
          )
          (1): Dropout(p=0.0, inplace=False)
          (2): Linear(in_features=12288, out_features=3072, bias=True)
        )
      )
      (txt_mod): Sequential(
        (0): SiLU()
        (1): Linear(in_features=3072, out_features=18432, bias=True)
      )
      (txt_norm1): LayerNorm((3072,), eps=1e-06, elementwise_affine=False)
      (txt_norm2): LayerNorm((3072,), eps=1e-06, elementwise_affine=False)
      (txt_mlp): QwenFeedForward(
        (net): ModuleList(
          (0): ApproximateGELU(
            (proj): Linear(in_features=3072, out_features=12288, bias=True)
          )
          (1): Dropout(p=0.0, inplace=False)
          (2): Linear(in_features=12288, out_features=3072, bias=True)
        )
      )
    )
  )
  (norm_out): AdaLayerNorm(
    (linear): Linear(in_features=3072, out_features=6144, bias=True)
    (norm): LayerNorm((3072,), eps=1e-06, elementwise_affine=False)
  )
  (proj_out): Linear(in_features=3072, out_features=64, bias=True)
)
```

在显存管理中，我们只关心包含参数的 Layer。在这个模型结构中，`QwenEmbedRope`、`TemporalTimesteps`、`SiLU` 等 Layer 都是不包含参数的，`LayerNorm` 也因为设置了 `elementwise_affine=False` 不包含参数。包含参数的 Layer 只有 `Linear` 和 `RMSNorm`。

`diffsynth.core.vram` 中提供了两个用于替换的模块用于显存管理：
* `AutoWrappedLinear`: 用于替换 `Linear` 层
* `AutoWrappedModule`: 用于替换其他任意层

编写一个 `module_map`，将模型中的 `Linear` 和 `RMSNorm` 映射到对应的模块上：

```python
module_map={
    torch.nn.Linear: AutoWrappedLinear,
    RMSNorm: AutoWrappedModule,
}
```

此外，还需要提供 `vram_config` 与 `vram_limit`，这两个参数在[显存管理](/docs/zh/Pipeline_Usage/VRAM_management.md#更多使用方式)中已有介绍。

调用 `enable_vram_management` 即可启用显存管理，注意此时模型加载时的 `device` 为 `cpu`，与 `offload_device` 一致：

```python
from diffsynth.core import load_model, enable_vram_management, AutoWrappedLinear, AutoWrappedModule
from diffsynth.models.qwen_image_dit import QwenImageDiT, RMSNorm
import torch

prefix = "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model"
model_path = [prefix + f"-0000{i}-of-00009.safetensors" for i in range(1, 10)]
inputs = {
    "latents": torch.randn((1, 16, 128, 128), dtype=torch.bfloat16, device="cuda"),
    "timestep": torch.zeros((1,), dtype=torch.bfloat16, device="cuda"),
    "prompt_emb": torch.randn((1, 5, 3584), dtype=torch.bfloat16, device="cuda"),
    "prompt_emb_mask": torch.ones((1, 5), dtype=torch.int64, device="cuda"),
    "height": 1024,
    "width": 1024,
}

model = load_model(QwenImageDiT, model_path, torch_dtype=torch.bfloat16, device="cpu")
enable_vram_management(
    model,
    module_map={
        torch.nn.Linear: AutoWrappedLinear,
        RMSNorm: AutoWrappedModule,
    },
    vram_config = {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    },
    vram_limit=0,
)
with torch.no_grad():
    output = model(**inputs)
```

以上代码只需要 2G 显存就可以运行 20B 模型的 `forward`。

## Disk Offload

[Disk Offload](/docs/zh/Pipeline_Usage/VRAM_management.md#disk-offload) 是特殊的显存管理方案，需在模型加载过程中启用，而非模型加载完毕后。通常，在以上代码能够顺利运行的前提下，Disk Offload 可以直接启用：

```python
from diffsynth.core import load_model, enable_vram_management, AutoWrappedLinear, AutoWrappedModule
from diffsynth.models.qwen_image_dit import QwenImageDiT, RMSNorm
import torch

prefix = "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model"
model_path = [prefix + f"-0000{i}-of-00009.safetensors" for i in range(1, 10)]
inputs = {
    "latents": torch.randn((1, 16, 128, 128), dtype=torch.bfloat16, device="cuda"),
    "timestep": torch.zeros((1,), dtype=torch.bfloat16, device="cuda"),
    "prompt_emb": torch.randn((1, 5, 3584), dtype=torch.bfloat16, device="cuda"),
    "prompt_emb_mask": torch.ones((1, 5), dtype=torch.int64, device="cuda"),
    "height": 1024,
    "width": 1024,
}

model = load_model(
    QwenImageDiT,
    model_path,
    module_map={
        torch.nn.Linear: AutoWrappedLinear,
        RMSNorm: AutoWrappedModule,
    },
    vram_config={
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": "disk",
        "onload_device": "disk",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    },
    vram_limit=0,
)
with torch.no_grad():
    output = model(**inputs)
```

Disk Offload 是极为特殊的显存管理方案，只支持 `.safetensors` 格式文件，不支持 `.bin`、`.pth`、`.ckpt` 等二进制文件，不支持带 Tensor reshape 的 [state dict converter](/docs/zh/Developer_Guide/Integrating_Your_Model.md#step-2-模型文件格式转换)。

如果出现非 Disk Offload 能正常运行但 Disk Offload 不能正常运行的情况，请在 GitHub 上给我们提 issue。

## 写入默认配置

为了让用户能够更方便地使用显存管理功能，我们将细粒度显存管理的配置写在 `diffsynth/configs/vram_management_module_maps.py` 中，上述模型的配置信息为：

```python
"diffsynth.models.qwen_image_dit.QwenImageDiT": {
    "diffsynth.models.qwen_image_dit.RMSNorm": "diffsynth.core.vram.layers.AutoWrappedModule",
    "torch.nn.Linear": "diffsynth.core.vram.layers.AutoWrappedLinear",
}
```
