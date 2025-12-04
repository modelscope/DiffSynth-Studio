# Fine-Grained VRAM Management Scheme

This document introduces how to write reasonable fine-grained VRAM management schemes for models, and how to use the VRAM management functions in `DiffSynth-Studio` for other external code libraries. Before reading this document, please read the document [VRAM Management](/docs/en/Pipeline_Usage/VRAM_management.md).

## How Much VRAM Does a 20B Model Need?

Taking Qwen-Image's DiT model as an example, this model has reached 20B parameters. The following code will load this model and perform inference, requiring about 40G VRAM. This model obviously cannot run on consumer-grade GPUs with smaller VRAM.

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

## Writing Fine-Grained VRAM Management Scheme

To write a fine-grained VRAM management scheme, we need to use `print(model)` to observe and analyze the model structure:

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

In VRAM management, we only care about layers containing parameters. In this model structure, `QwenEmbedRope`, `TemporalTimesteps`, `SiLU` and other Layers do not contain parameters. `LayerNorm` also does not contain parameters because `elementwise_affine=False` is set. Layers containing parameters are only `Linear` and `RMSNorm`.

`diffsynth.core.vram` provides two replacement modules for VRAM management:
* `AutoWrappedLinear`: Used to replace `Linear` layers
* `AutoWrappedModule`: Used to replace any other layer

Write a `module_map` to map `Linear` and `RMSNorm` in the model to the corresponding modules:

```python
module_map={
    torch.nn.Linear: AutoWrappedLinear,
    RMSNorm: AutoWrappedModule,
}
```

In addition, `vram_config` and `vram_limit` are also required, which have been introduced in [VRAM Management](/docs/en/Pipeline_Usage/VRAM_management.md#more-usage-methods).

Call `enable_vram_management` to enable VRAM management. Note that the `device` when loading the model is `cpu`, consistent with `offload_device`:

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

The above code only requires 2G VRAM to run the `forward` of a 20B model.

## Disk Offload

[Disk Offload](/docs/en/Pipeline_Usage/VRAM_management.md#disk-offload) is a special VRAM management scheme that needs to be enabled during the model loading process, not after the model is loaded. Usually, when the above code can run smoothly, Disk Offload can be directly enabled:

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

Disk Offload is an extremely special VRAM management scheme. It only supports `.safetensors` format files, not binary files such as `.bin`, `.pth`, `.ckpt`, and does not support [state dict converter](/docs/en/Developer_Guide/Integrating_Your_Model.md#step-2-model-file-format-conversion) with Tensor reshape.

If there are situations where Disk Offload cannot run normally but non-Disk Offload can run normally, please submit an issue to us on GitHub.

## Writing Default Configuration

To make it easier for users to use the VRAM management function, we write the fine-grained VRAM management configuration in `diffsynth/configs/vram_management_module_maps.py`. The configuration information for the above model is:

```python
"diffsynth.models.qwen_image_dit.QwenImageDiT": {
    "diffsynth.models.qwen_image_dit.RMSNorm": "diffsynth.core.vram.layers.AutoWrappedModule",
    "torch.nn.Linear": "diffsynth.core.vram.layers.AutoWrappedLinear",
}
```# Fine-Grained VRAM Management Scheme

This document introduces how to write reasonable fine-grained VRAM management schemes for models, and how to use the VRAM management functions in `DiffSynth-Studio` for other external code libraries. Before reading this document, please read the document [VRAM Management](/docs/en/Pipeline_Usage/VRAM_management.md).

## How Much VRAM Does a 20B Model Need?

Taking Qwen-Image's DiT model as an example, this model has reached 20B parameters. The following code will load this model and perform inference, requiring about 40G VRAM. This model obviously cannot run on consumer-grade GPUs with smaller VRAM.

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

## Writing Fine-Grained VRAM Management Scheme

To write a fine-grained VRAM management scheme, we need to use `print(model)` to observe and analyze the model structure:

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

In VRAM management, we only care about layers containing parameters. In this model structure, `QwenEmbedRope`, `TemporalTimesteps`, `SiLU` and other Layers do not contain parameters. `LayerNorm` also does not contain parameters because `elementwise_affine=False` is set. Layers containing parameters are only `Linear` and `RMSNorm`.

`diffsynth.core.vram` provides two replacement modules for VRAM management:
* `AutoWrappedLinear`: Used to replace `Linear` layers
* `AutoWrappedModule`: Used to replace any other layer

Write a `module_map` to map `Linear` and `RMSNorm` in the model to the corresponding modules:

```python
module_map={
    torch.nn.Linear: AutoWrappedLinear,
    RMSNorm: AutoWrappedModule,
}
```

In addition, `vram_config` and `vram_limit` are also required, which have been introduced in [VRAM Management](/docs/en/Pipeline_Usage/VRAM_management.md#more-usage-methods).

Call `enable_vram_management` to enable VRAM management. Note that the `device` when loading the model is `cpu`, consistent with `offload_device`:

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

The above code only requires 2G VRAM to run the `forward` of a 20B model.

## Disk Offload

[Disk Offload](/docs/en/Pipeline_Usage/VRAM_management.md#disk-offload) is a special VRAM management scheme that needs to be enabled during the model loading process, not after the model is loaded. Usually, when the above code can run smoothly, Disk Offload can be directly enabled:

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

Disk Offload is an extremely special VRAM management scheme. It only supports `.safetensors` format files, not binary files such as `.bin`, `.pth`, `.ckpt`, and does not support [state dict converter](/docs/en/Developer_Guide/Integrating_Your_Model.md#step-2-model-file-format-conversion) with Tensor reshape.

If there are situations where Disk Offload cannot run normally but non-Disk Offload can run normally, please submit an issue to us on GitHub.

## Writing Default Configuration

To make it easier for users to use the VRAM management function, we write the fine-grained VRAM management configuration in `diffsynth/configs/vram_management_module_maps.py`. The configuration information for the above model is:

```python
"diffsynth.models.qwen_image_dit.QwenImageDiT": {
    "diffsynth.models.qwen_image_dit.RMSNorm": "diffsynth.core.vram.layers.AutoWrappedModule",
    "torch.nn.Linear": "diffsynth.core.vram.layers.AutoWrappedLinear",
}
```