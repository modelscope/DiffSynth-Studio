# 模型量化

量化通过降低模型权重的存储精度（如 4bit、8bit）来减少显存占用，让大模型能在更小的显卡上运行。`DiffSynth-Studio` 提供了统一的量化接口 `QuantizeConfig`，支持多种后端（bitsandbytes、torchao 等），既可以在线量化一个 fp 模型，也可以直接加载预量化的权重，还支持在量化模型上做 LoRA 训练（QLoRA）。

本文以 `Z-Image` 为例，介绍量化功能的使用。

> **量化 与 显存管理 FP8 的区别**
>
> [显存管理](./VRAM_management.md)中的 FP8 是通过 `offload_dtype` / `onload_dtype` 等参数控制权重在显存中的存储精度，它作用于所有参数、无需依赖第三方量化库，但只支持简单的 FP8 转换。
>
> 本文的量化（`QuantizeConfig`）是针对 `nn.Linear` 层的专门量化方案，支持 NF4、INT8、INT4 等多种更精细的量化格式，并支持保存/加载量化权重、QLoRA 训练。两者可以组合使用。

## 快速开始

在任意 `ModelConfig` 上传入 `quantize` 参数即可对该模型启用在线量化。以下代码将 Z-Image 的 DiT 用 NF4 量化后加载：

```python
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
from diffsynth.core.quant import QuantizeConfig
import torch

pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="Tongyi-MAI/Z-Image", origin_file_pattern="transformer/*.safetensors",
            quantize=QuantizeConfig(method="bitsandbytes_nf4"),
        ),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
)
prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
image = pipe(prompt=prompt, seed=42, num_inference_steps=50, cfg_scale=4)
image.save("image_z_image_nf4.jpg")
```

> bitsandbytes、torchao 等后端需要安装对应的第三方库，可通过 `pip install "diffsynth[quant]"` 一次性安装。

## 支持的量化方法

框架内置的量化方法如下（`method` 即传入 `QuantizeConfig` 的名称）：

| method | 后端 | 位宽 | 依赖库 | 可保存 | 可训练 (QLoRA) |
| --- | --- | --- | --- | --- | --- |
| `bitsandbytes_nf4` | bitsandbytes | 4bit (NF4) | bitsandbytes | ✅ | ✅ |
| `bitsandbytes_fp4` | bitsandbytes | 4bit (FP4) | bitsandbytes | ✅ | ✅ |
| `torchao_int8_w8a16` | torchao | 8bit (INT8, weight-only) | torchao | ✅ | ✅ |
| `torchao_int4_w4a16` | torchao | 4bit (INT4, weight-only) | torchao | ✅ | ❌ |
| `torchao_fp8_w8a16` | torchao | 8bit (FP8, weight-only) | torchao | ✅ | ✅ |
| `torchao_int8_w8a8` | torchao | W8A8 (INT8 权重 + INT8 动态激活) | torchao | ❌ | ❌ |
| `torchao_fp8_w8a8` | torchao | W8A8 (FP8 权重 + FP8 动态激活) | torchao | — | — |
| `torchao_int4_w4a8` | torchao | W4A8 (INT4 权重 + FP8 动态激活) | torchao + mslk | — | — |
| `torchao_mxfp8_w8a8` | torchao | W8A8 (MXFP8 microscaling) | torchao | ✅ | ❌ |
| `torchao_mxfp4_w4a4` | torchao | W4A4 (MXFP4 microscaling) | torchao | ✅ | ❌ |
| `torchao_nvfp4_w4a4` | torchao | W4A4 (NVFP4) | torchao | — | — |

### 激活量化的硬件要求

上表中 `w8a8` / `w4a8` / `w4a4` 这类**激活量化**方法除了量化权重，还会量化激活，因此对显卡算力
（compute capability，简称 SM）有额外要求。框架会在量化前检查，不满足时直接报错并说明原因，
而不是让底层 kernel 抛出难以理解的错误：

| 方法 | 最低 SM | 说明 |
| --- | --- | --- |
| `torchao_int8_w8a8` | 无 | 任意 CUDA 设备可用；但不支持保存量化权重，也无法训练 |
| `torchao_fp8_w8a8` | 8.9 | 需要 fp8 tensor core |
| `torchao_int4_w4a8` | 9.0 | 需要 mslk 的 int4 kernel（WGMMA / TMA 指令），另需 `pip install mslk` |
| `torchao_mxfp8_w8a8` / `torchao_mxfp4_w4a4` / `torchao_nvfp4_w4a4` | 10.0 | 需要块缩放（blockwise scale）的 fp4/fp8 矩阵乘 |

标注 `—` 的方法（`torchao_fp8_w8a8`、`torchao_int4_w4a8`、`torchao_nvfp4_w4a4`）受硬件限制，
尚未在实机上验证其保存 / 训练能力。MX 两项的结论是通过下面的数值仿真路径实测得到的。

MX 系列还支持**数值仿真**：传入 `kernel_preference="emulated"` 会先反量化再走普通矩阵乘，
可在任意显卡上评估该量化方案的精度损失（但没有加速效果）：

```python
QuantizeConfig(method="torchao_mxfp4_w4a4",
               backend_config_kwargs={"kernel_preference": "emulated"})
```

> `torchao_int4_w4a16` 默认的 `int4_packing_format="plain"` 同样依赖 mslk 的 Hopper kernel。
> 在更早的显卡（如 A100）上请改用 torch 原生打包格式：
> `backend_config_kwargs={"int4_packing_format": "tile_packed_to_4d"}`。

你可以在代码中查看某个方法支持的后端配置参数：

```python
from diffsynth.core.quant import describe_quant_method, QUANT_METHODS

# 列出所有已注册的方法名
print(list(QUANT_METHODS.keys()))

# 查看某个方法的后端、说明，以及可接受的 backend_config_kwargs 及其默认值
describe_quant_method("bitsandbytes_nf4")
```

## QuantizeConfig 详解

`QuantizeConfig` 描述了"用哪种方法、量化哪些层、量化后如何运行"。常用字段：

- **`method`**：量化方法名，见上表，必填。
- **`mode`**：量化层的运行方式。
    - `"dynamic"`（默认）：保留后端原生的量化 Linear，每次 forward 时临时反量化计算，显存占用低。
    - `"dequant_once"`：量化/加载完成后，一次性把量化层还原成普通的 fp `nn.Linear`（保留量化引入的误差）。适合需要标准 `nn.Linear` 的场景，不再省显存。
- **`target_modules` / `exclude_modules`**：按层名过滤要量化的 `nn.Linear`。层名匹配规则为：完整点分名称完全相等，或以 `"." + 名称` 结尾（例如 `"img_mod.1"` 能匹配 `transformer_blocks.0.img_mod.1`）。一般把对量化敏感的层（如 embedding、输出层、调制层）放进 `exclude_modules`。
- **`backend_config_kwargs`**：透传给后端配置的参数，例如 NF4 的 `blocksize`、`compress_statistics` 等，具体可用 `describe_quant_method(method)` 查看。
- **`load_prequantized`**：是否加载已经量化好的权重（见下一节）。

示例：只量化注意力和 MLP 层，排除对精度敏感的层：

```python
from diffsynth.core.quant import QuantizeConfig

quantize = QuantizeConfig(
    method="bitsandbytes_nf4",
    mode="dynamic",
    exclude_modules=["time_embedder.proj_in", "time_embedder.proj_out", "proj_out"],
    backend_config_kwargs={"compress_statistics": False},
)
```

## 加载预量化权重

除了在线量化，框架也支持直接加载已经量化好的 checkpoint，省去每次加载时的量化开销。

对于官方发布的量化模型，其配置中已经写好了 `quant_config`，框架会自动识别并加载，你无需做任何额外配置，像加载普通模型一样即可：

```python
ModelConfig(model_id="ideogram-ai/ideogram-4-nf4", origin_file_pattern="transformer/diffusion_pytorch_model.safetensors")
```

对于自己保存的量化 checkpoint（见下一节），加载时需要显式传入 `quantize`，并设置 `load_prequantized=True`：

```python
from diffsynth.core.quant import QuantizeConfig

ModelConfig(
    path="models/z-image-nf4/transformer.safetensors",
    quantize=QuantizeConfig(method="bitsandbytes_nf4", load_prequantized=True),
)
```

> `load_prequantized=True` 表示 checkpoint 中已经是量化后的权重，框架会先构建对应的量化"空壳"再把权重直接赋值进去，而不会再对 fp 权重做一次在线量化。

## 保存量化模型

如果你想把一次在线量化的结果保存下来，反复使用，可以用 `save_quantized_model`：

```python
from diffsynth.core.loader import ModelConfig
from diffsynth.core.quant import QuantizeConfig
from diffsynth.utils.quant.serialization import save_quantized_model

quantize = QuantizeConfig(method="bitsandbytes_nf4")
model_config = ModelConfig(
    model_id="Tongyi-MAI/Z-Image",
    origin_file_pattern="transformer/*.safetensors",
    quantize=quantize,
)
save_quantized_model(model_config, "models/z-image-nf4/transformer.safetensors")
```

它会下载并加载原始 fp 权重、执行量化、再把量化后的 state dict 以 `.safetensors` 格式保存下来。保存后即可用上一节的方式（`load_prequantized=True`）加载。

> 只有后端声明了 `is_serializable=True` 才能保存量化权重（内置的 bitsandbytes、torchao 均支持）。

## 混合量化

不同层对量化的敏感程度不同。`MixedQuantizeConfig` 允许对不同的层集合应用不同的量化方法，例如对精度敏感的调制层用 INT8、其余层用 NF4：

```python
from diffsynth.core.quant import QuantizeConfig, MixedQuantizeConfig

mod_layers = [
    "img_mod.1", "txt_mod.1", "norm_out.linear", "img_in", "txt_in", "proj_out",
]
quantize = MixedQuantizeConfig(configs=[
    QuantizeConfig(method="bitsandbytes_nf4", exclude_modules=mod_layers),
    QuantizeConfig(method="torchao_int8_w8a16", target_modules=mod_layers),
])
```

各个子配置会按顺序依次执行，且它们匹配到的层集合必须互不重叠，框架会在量化前校验并在冲突时报错。`MixedQuantizeConfig` 对外的接口与单个 `QuantizeConfig` 完全一致，可以直接传给 `ModelConfig(quantize=...)`。

> 注意：加载混合量化的预量化 checkpoint 时，`load_prequantized=True` 要设置在 `MixedQuantizeConfig` 上，而不是子配置上。

## 量化 + LoRA 训练（QLoRA）

量化后的 Linear 是 `torch.nn.Linear` 的子类，且内置后端的量化层都是可微的（`is_differentiable=True`），梯度可以穿过冻结的量化权重反传到 LoRA 分支。因此你可以在量化的底模上直接注入并训练 LoRA，即 QLoRA，从而在很小的显存里训练大模型。

典型做法是：**用预量化的底模进行训练**。训练脚本通过 `--model_id_with_origin_paths` 指向预量化模型，框架会自动识别其 `quant_config` 完成量化，再由 `--lora_base_model` / `--lora_target_modules` 注入 LoRA：

```bash
accelerate launch examples/.../train.py \
  --model_id_with_origin_paths "DiffSynth-Studio/MiniMax-H3-NF4:minimax-h3-fl2va-nf4.safetensors,..." \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --output_path "./models/train/xxx-nf4"
```

- 训练过程中量化底模的权重保持冻结，只有 LoRA 分支参与更新，因此保存下来的是 fp 精度的 LoRA 权重（不是量化权重）。
- 推理时按"量化底模 + LoRA"的方式加载：先像[加载预量化权重](#加载预量化权重)一样加载量化底模，再 `pipe.load_lora(pipe.dit, "epoch-x.safetensors")`。

## 自定义量化后端

如果内置方法不能满足需求，你可以实现自己的量化后端。核心是继承 `QuantBackend`，产出一个满足契约的量化 `nn.Linear`，并注册方法名。可参考 `diffsynth/models/ideogram4_dit.py` 中的 `Fp8Linear` / `Ideogram4Fp8QuantBackend`。

一个量化后端产出的量化 Linear 必须满足以下契约（详见 `diffsynth/core/quant/base.py`）：

- **(a)** 是 `nn.Linear` 的即插即用替代：`forward(x)` 内部完成反量化 + 矩阵乘。
- **(b)** `.to(...)` 只移动设备、不改变打包权重/量化状态的 dtype（dtype 转换需保持其存储格式与数值不变）。
- **(c)** `state_dict()` / `load_state_dict(assign=True)` 可往返（必要时通过 `flatten_state_dict` / `unflatten_state_dict`）。
- **(d)**（仅训练场景）`forward` 对输入可微，梯度能穿过冻结的量化层到达 LoRA 分支。

一个最小后端骨架：

```python
import torch
from diffsynth.core.quant import QuantBackend, register_quant_backend, register_quant_method


class MyQuantLinear(torch.nn.Linear):
    """自定义的量化 Linear，需满足上述契约 (a)-(d)。"""
    # 若持有需要保护 dtype 的打包张量，可参考 ideogram4_dit.py 的 `_apply` 写法。


@register_quant_backend("my_backend")
class MyQuantBackend(QuantBackend):
    def capabilities(self):
        return {**super().capabilities(), "is_serializable": True, "is_differentiable": True}

    def quantized_linear_classes(self):
        return (MyQuantLinear,)

    def create_quantized_linear(self, linear, compute_device=None, model_device=None):
        # 在线量化：把一个 fp nn.Linear 转成 MyQuantLinear
        ...

    def create_quantized_linear_shell(self, linear, compute_dtype):
        # 构建空壳，用于加载预量化 checkpoint
        ...

    def dequantize_to_linear(self, module, compute_dtype, compute_device=None, model_device=None):
        # 还原回普通 nn.Linear（支持 mode="dequant_once"）
        ...


# 注册一个量化方法名，指向该后端；config_factory 把 backend_config_kwargs 转为后端配置
register_quant_method("my_method", "my_backend", lambda kwargs: dict(kwargs), label="my custom method")
```

注册后即可像内置方法一样使用：`QuantizeConfig(method="my_method")`。

框架提供了两个校验工具，建议在实现后运行以确保后端行为正确：

```python
from diffsynth.core.quant import QUANT_BACKENDS, QUANT_METHODS
from diffsynth.core.quant.base import check_backend_contract, check_differentiable

spec = QUANT_METHODS["my_method"]
backend = QUANT_BACKENDS[spec.backend](spec.config_factory({}))
check_backend_contract(backend)   # 校验类声明、工厂方法、state dict 键覆盖等契约
# check_differentiable(quantized_linear)  # 校验梯度可穿过量化层（QLoRA 训练需要）
```

## 最佳实践

- **优先加载官方预量化模型**：省去在线量化开销，且量化配置已调好。
- **敏感层用 `exclude_modules` 排除**：embedding、时间步嵌入、输出层、调制层等对量化较敏感，量化后可能明显掉质量，建议排除或用更高精度（配合混合量化）。
- **量化与显存管理组合**：在 `ModelConfig` 上同时设置 `quantize` 和 `vram_config` / `vram_limit`，可进一步压低显存。参见[显存管理](./VRAM_management.md)。
- **精度与显存权衡**：4bit（NF4/INT4）省显存最多但质量损失更大；8bit（INT8/FP8）质量损失较小。可先用 8bit，显存仍不够再降到 4bit。
