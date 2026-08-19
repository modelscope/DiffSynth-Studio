# `diffsynth.core.quant`: 模型量化

本文档介绍 `diffsynth.core.quant` 中的量化底层接口，如果你希望将这些功能用于其他的代码库中，可参考本文档。若只想在 `Pipeline` 中启用量化，请参考[模型量化](../../Pipeline_Usage/Quantization.md)。

模块通过 `diffsynth.core.quant` 导出以下接口，分为三类：

| 分类 | 接口 |
| --- | --- |
| 用户接口 | `QuantizeConfig`、`MixedQuantizeConfig`、`describe_quant_method`、`QUANT_METHODS` |
| 扩展接口 | `QuantBackend`、`BackendConfig`、`register_quant_backend`、`register_quant_method`、`QuantMethodSpec`、`QUANT_BACKENDS` |
| 验证工具 | `check_differentiable`、`check_backend_contract` |

量化的作用对象是模型中的 `nn.Linear`：框架遍历模型、把命中的 `nn.Linear` 替换成后端提供的量化 Linear（它们都是 `nn.Linear` 的子类，因此 LoRA 注入、显存管理等机制无需改动即可识别）。后端只负责单层的量化，模型级的遍历与替换由 `QuantizeConfig` 完成。

## 用户接口

### QuantizeConfig

`QuantizeConfig` 既是量化配置，也是作用于任意 `nn.Module` 的操作入口。

字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `method` | `str` | 量化方法名，取自 `QUANT_METHODS`，决定后端、量化方案与后端配置。必填 |
| `mode` | `str` | `"dynamic"`（默认）保留后端原生量化 Linear，每次 forward 反量化；`"dequant_once"` 在权重量化或加载后立刻还原为普通 fp `nn.Linear` |
| `target_modules` | `list` | 只量化命中的层；`None` 表示不限制 |
| `exclude_modules` | `list` | 排除命中的层 |
| `backend_config_kwargs` | `dict` | 传给该方法后端配置工厂的参数，决定量化行为，例如 nf4 的 `blocksize` |
| `load_prequantized` | `bool` | checkpoint 中已是量化权重，直接加载而不在线量化 |

`target_modules` / `exclude_modules` 的匹配规则：层的完整点分名称与列表项相等，或以 `"." + 列表项` 结尾。例如 `"img_mod.1"` 能匹配 `transformer_blocks.0.img_mod.1`。

构造 `QuantizeConfig` 时会校验后端依赖与参数，不满足时立即报错（含安装指引），不会推迟到推理时才失败。

主要方法：

#### `quantize_model(model, compute_device=None, model_device=None)`

原地量化 `model` 中命中的 `nn.Linear`，保持每层原有的 dtype。必须在 `load_state_dict` **之后**调用。`load_prequantized=True` 时该方法什么都不做（这种 checkpoint 本来就是量化的）。

- `compute_device`：量化计算发生的设备；`None` 表示就地量化。
- `model_device`：量化完成后每层的存放设备；`None` 表示留在 `compute_device` 上。

fp 模型放在 CPU、配合 `compute_device="cuda", model_device="cpu"`，可以逐层流式量化，加速器上同时只驻留一层：

```python
import torch
from diffsynth.core.quant import QuantizeConfig

cfg = QuantizeConfig(method="bitsandbytes_nf4")
model.load_state_dict(fp_state_dict)
cfg.quantize_model(model, compute_device="cuda", model_device="cpu")
```

#### `prepare_for_prequantized_load(model, compute_dtype=torch.bfloat16)`

把命中的 `nn.Linear` 换成与预量化 checkpoint 结构一致的空量化层（"空壳"）。必须在 `load_state_dict(assign=True)` **之前**调用。`compute_dtype` 是量化层在 forward 时反量化到的 dtype。

#### `unflatten_state_dict(state_dict, metadata)` / `flatten_state_dict(state_dict)`

量化权重往往是"打包张量 + 量化状态"的复合结构，而 `.safetensors` 只能存普通张量，这两个方法负责在两种形态间转换。

- `unflatten_state_dict(state_dict, metadata)`：把从 checkpoint 读出的扁平张量重建为复合量化张量，结果可交给 `load_state_dict(assign=True)`。
- `flatten_state_dict(state_dict)`：把量化模型的 state dict 摊平为普通张量与纯字符串 metadata，返回 `(tensors, metadata)`，可直接交给 `safetensors.torch.save_file(tensors, path, metadata=metadata)`。后端未声明 `is_serializable` 时抛出 `NotImplementedError`。

加载预量化 checkpoint 的完整流程：

```python
import torch
from diffsynth.core.quant import QuantizeConfig

cfg = QuantizeConfig(method="bitsandbytes_nf4", load_prequantized=True)
cfg.prepare_for_prequantized_load(model, compute_dtype=torch.bfloat16)
state_dict = cfg.unflatten_state_dict(state_dict, metadata)
model.load_state_dict(state_dict, assign=True)
```

#### `dequantize_model(model, compute_dtype=torch.bfloat16, compute_device=None, model_device=None)`

把模型中所有量化 Linear 换回普通 fp `nn.Linear`，还原出的权重带有量化误差。**仅当 `mode="dequant_once"` 时生效**，否则直接返回。可在上面两种流程之后调用：

```python
cfg.dequantize_model(model, compute_dtype=torch.bfloat16)
```

#### `is_quantized_linear(module)`

判断 `module` 是否为本配置后端产出的量化 Linear。

#### `build_quantized_shell(module, compute_dtype)`

构建与 `module` 形状、bias 一致的空量化 Linear。用于在保持层可路由的前提下释放其权重，以及在计算设备上暂存一份副本，是显存管理的配套接口。

### MixedQuantizeConfig

把多个 `QuantizeConfig` 组合成一次混合量化，每个子配置负责一组互不重叠的层，对外暴露与单个 `QuantizeConfig` 相同的接口（`quantize_model`、`prepare_for_prequantized_load`、`dequantize_model`、`flatten_state_dict`、`unflatten_state_dict`、`is_quantized_linear`、`build_quantized_shell`，以及 `method` / `mode` 两个只读属性）。

```python
from diffsynth.core.quant import QuantizeConfig, MixedQuantizeConfig

mod_layers = ["img_mod.1", "txt_mod.1", "norm_out.linear", "img_in", "txt_in", "proj_out"]
cfg = MixedQuantizeConfig(configs=[
    QuantizeConfig(method="bitsandbytes_nf4", exclude_modules=mod_layers),
    QuantizeConfig(method="torchao_int8_w8a16", target_modules=mod_layers),
])
cfg.quantize_model(model, compute_device="cuda")
```

字段与约束：

- `configs`：`QuantizeConfig` 列表，按顺序执行。所有子配置必须共享同一个 `mode`，且它们的 `load_prequantized` 必须为 `False`。
- `load_prequantized`：加载混合量化 checkpoint 时设置在本包装类上，而非子配置上。
- 子配置匹配到的层集合必须两两不相交。`quantize_model` 与 `prepare_for_prequantized_load` 会在改动模型之前校验，冲突时报错并指出重叠的层名。

`build_quantized_shell(module, compute_dtype, layer_name=None)` 在这里多了 `layer_name` 参数：当多个子配置共用同一后端时，它们产出的量化 Linear 是同一个类，只能靠层名判断归属。

### describe_quant_method 与 QUANT_METHODS

`QUANT_METHODS` 是 `{方法名: QuantMethodSpec}` 的注册表。`QuantMethodSpec` 有三个字段：`backend`（后端名）、`config_factory`（把 `backend_config_kwargs` 转成后端配置的可调用对象）、`label`（人类可读的说明）。

列举全部方法前需先调用 `backends.load_all_backends()`：

```python
from diffsynth.core.quant import QUANT_METHODS, backends

backends.load_all_backends()
print(sorted(QUANT_METHODS))
```

`describe_quant_method(name)` 打印某个方法的后端、说明，以及它接受的 `backend_config_kwargs` 及默认值（内部会自动加载后端）：

```python
from diffsynth.core.quant import describe_quant_method

describe_quant_method("comfy_kitchen_int8_w8a8")
```

```
method: comfy_kitchen_int8_w8a8
backend: comfy_kitchen
detail: W8A8, int8 weight + int8 dynamic activation (ComfyUI int8_tensorwise)
backend config: diffsynth.core.quant.backends.comfy_kitchen.ComfyKitchenInt8Config
backend_config_kwargs (user-tunable):
  per_channel       = True
  convrot           = True
  convrot_groupsize = 256
  orig_dtype        = torch.bfloat16
pinned by method (not overridable):
  format = 'int8_tensorwise'
```

其中 `user-tunable` 是可以通过 `backend_config_kwargs` 修改的参数，`pinned by method` 是该方法固定、不可修改的部分（例如 `comfy_kitchen_int8_w8a8` 与 `comfy_kitchen_fp8_w8a8` 共用一个后端，靠 `format` 区分）。传入未被接受的键会直接报错并列出可用键。

## 扩展接口：自定义后端

### QuantBackend 契约

`QuantBackend` 是框架与具体量化库（bitsandbytes / torchao / 自定义）之间的适配层。子类通过 `register_quant_backend` 注册到 `QUANT_BACKENDS`，由 `QuantizeConfig` 实例化并注入方法对应的后端配置。

后端产出的量化 Linear 必须满足以下四条契约：

- **(a)** 是 `nn.Linear` 的即插即用替代：`forward(x)` 内部完成反量化 + 矩阵乘。
- **(b)** `.to(...)` 只移动设备，绝不改变打包权重 / 量化状态的类型：`.to(dtype)`、`.half()`、`.float()` 等 dtype 转换必须让它们的存储格式与数值保持原样。
- **(c)** `state_dict()` 与 `load_state_dict(assign=True)` 可往返（必要时借助 `flatten_state_dict` / `unflatten_state_dict`）。
- **(d)**（仅训练场景）`forward` 对输入可微，梯度能穿过冻结的量化层到达 LoRA 分支。静态上由 `capabilities()["is_differentiable"]` 声明，运行时可用 `check_differentiable` 验证。

契约 (b) 之所以必要，是因为显存管理会对模型做 dtype/device 转换，若打包权重被误转成 bf16，量化状态就被破坏了。可参考 `diffsynth/models/ideogram4_dit.py` 中 `Fp8Linear._apply` 的写法：把需要保护的张量名登记下来，在 `_apply` 里把会改变其 dtype 的转换降级为纯设备迁移。

需要实现或覆盖的成员：

| 成员 | 说明 |
| --- | --- |
| `name` | 由 `register_quant_backend` 自动设置 |
| `project_url` | 后端所属库的项目地址，`announce_environment()` 会打印它，把硬件兼容性问题指向上游 |
| `capabilities()` | 返回 `is_serializable` / `is_differentiable` / `is_compileable` / `requires_calibration` 四个布尔标志，默认全为 `False` |
| `validate_environment()` | 检查依赖库与硬件，缺失时抛出带安装指引的异常。在 `QuantizeConfig` 构造时调用 |
| `quantized_linear_classes()` | 声明本后端产出的 Linear 类，必须都是 `torch.nn.Linear` 的子类。`is_quantized_linear` 默认基于它做 `isinstance` 判断 |
| `create_quantized_linear(linear, compute_device, model_device)` | 在线量化：把一个 fp `nn.Linear` 转成量化 Linear。不实现则该后端不支持在线量化 |
| `create_quantized_linear_shell(linear, compute_dtype)` | 构建空壳，用于加载预量化 checkpoint。不实现则该后端不支持预量化加载 |
| `dequantize_to_linear(module, compute_dtype, compute_device, model_device)` | 还原成普通 `nn.Linear`。不实现则 `mode="dequant_once"` 不可用 |
| `flatten_state_dict` / `unflatten_state_dict` | 量化 state dict 与扁平张量之间的转换，`is_serializable=True` 时需要实现 |

基类对未实现的方法给出了明确的报错信息，因此只支持部分能力的后端可以只实现自己需要的那几个。

### BackendConfig

`BackendConfig` 是后端类型化配置的基类。用户可调的参数写成普通 dataclass 字段，方法固定的值用 `field(init=False, default=...)` 声明，这样它们既能被 `describe_quant_method` 区分展示，也无法通过 `backend_config_kwargs` 修改。

类方法 `from_kwargs(kwargs)` 会校验传入的键：出现未声明的键时抛出 `ValueError` 并列出全部可接受的键。它通常直接作为 `register_quant_method` 的 `config_factory`。

bitsandbytes 后端的写法就是这个模式的典型示例——共享的 4bit 参数放在基类，`quant_type` 由每个方法的子类钉死：

```python
from dataclasses import dataclass, field
import torch
from diffsynth.core.quant import BackendConfig, register_quant_method


@dataclass
class BitsAndBytes4bitConfig(BackendConfig):
    compress_statistics: bool = True
    blocksize: int = None
    quant_storage: torch.dtype = torch.uint8


@dataclass
class BitsAndBytesNF4Config(BitsAndBytes4bitConfig):
    quant_type: str = field(init=False, default="nf4")


register_quant_method("bitsandbytes_nf4", "bitsandbytes", BitsAndBytesNF4Config.from_kwargs, label="4bit, nf4, weight-only")
```

`config_factory` 不强制返回 `BackendConfig`：若后端直接消费第三方库的配置对象，也可以传入任意把 `dict` 转成该对象的函数（torchao 后端就是这样直接构建 `Int8WeightOnlyConfig` 等）。

### register_quant_backend 与 register_quant_method

- `register_quant_backend(name)`：类装饰器，把后端类注册到 `QUANT_BACKENDS` 并设置其 `name`。
- `register_quant_method(name, backend, config_factory, label="")`：注册一个方法名到 `QUANT_METHODS`，指明它使用哪个后端、如何构建后端配置。一个后端可以注册多个方法，用固定字段区分量化方案。

一个最小后端的完整骨架：

```python
import torch
from diffsynth.core.quant import QuantBackend, register_quant_backend, register_quant_method


class MyQuantLinear(torch.nn.Linear):
    """自定义量化 Linear，需满足契约 (a)-(d)。"""


@register_quant_backend("my_backend")
class MyQuantBackend(QuantBackend):
    project_url = "https://example.com/my-quant-lib"

    def capabilities(self):
        return {**super().capabilities(), "is_serializable": True, "is_differentiable": True}

    def validate_environment(self):
        ...   # 依赖缺失时抛出 ImportError

    def quantized_linear_classes(self):
        return (MyQuantLinear,)

    def create_quantized_linear(self, linear, compute_device=None, model_device=None):
        ...

    def create_quantized_linear_shell(self, linear, compute_dtype):
        ...

    def dequantize_to_linear(self, module, compute_dtype, compute_device=None, model_device=None):
        ...


register_quant_method("my_method", "my_backend", lambda kwargs: dict(kwargs), label="my custom method")
```

注册后即可像内置方法一样使用：`QuantizeConfig(method="my_method")`。若后端定义在 `diffsynth/core/quant/backends/` 之外（例如与某个模型放在一起），只要该模块在构造 `QuantizeConfig` 之前被导入过即可。

## 验证工具

### check_differentiable

```python
check_differentiable(module, example_input=None, verbose=True) -> bool
```

检查梯度能否穿过 `module` 到达其输入：从输出真实反向一次（`torch.autograd.grad`），并确认输入端收到了有限的梯度。这正是 LoRA 训练对冻结（量化）层的要求。模块会被原地转为 bfloat16 并用 bfloat16 输入探测；`example_input` 为 `None` 时会为暴露了 `in_features` 的模块自动构造随机输入。

```python
import torch
from diffsynth.core.quant import check_differentiable
from torchao.quantization import quantize_, Int8WeightOnlyConfig

linear = torch.nn.Linear(1024, 1024, dtype=torch.bfloat16, device="cuda")
quantize_(linear, Int8WeightOnlyConfig(version=2))
check_differentiable(linear)
```

### check_backend_contract

```python
check_backend_contract(backend, in_features=512, out_features=512,
                       compute_dtype=torch.bfloat16, compute_device="cuda", verbose=True) -> bool
```

新后端的准入自检：验证它声明了自己的 Linear 类、两个工厂方法都返回这些类的实例、每个声明的类都是 `torch.nn.Linear` 的子类（否则 LoRA 目标探测与显存管理都看不见它），并检查后端实际写出的 checkpoint 键是否都落在层名之下——键名模式漏掉某个 scale 会让 Disk Offload 静默加载出损坏的层。不支持的工厂方法会被跳过而不算失败。

```python
from diffsynth.core.quant import QUANT_BACKENDS, QUANT_METHODS, check_backend_contract

spec = QUANT_METHODS["bitsandbytes_nf4"]
check_backend_contract(QUANT_BACKENDS[spec.backend](spec.config_factory({})))
```
