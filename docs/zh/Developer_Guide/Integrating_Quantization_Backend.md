# 接入量化后端

`DiffSynth-Studio` 的量化框架由 `diffsynth.core.quant` 提供，内置了 bitsandbytes、torchao、comfy-kitchen 等后端（见[模型量化](../Pipeline_Usage/Quantization.md)）。如果你有自研的量化算法或想接入新的量化库，只需实现一个 `QuantBackend`，框架的在线量化、预量化 checkpoint 保存/加载、混合量化、显存管理、量化 + LoRA 训练都能直接复用。

本文以一个玩具后端 **INT9**（9bit 对称权重量化，真实按 9bit 打包存储，每个输出通道一个 fp32 scale）为例，走完接入的全过程。INT9 在硬件上并不存在，这里只是为了让示例代码足够短、又能覆盖所有需要实现的接口。完整的接口签名与约定见 [`diffsynth.core.quant` API 文档](../API_Reference/core/quant.md#扩展接口自定义后端)。

## 框架结构

量化框架分成三层：

- **`QuantizeConfig`**：面向用户的配置与入口，负责在模型中遍历、匹配并替换 `nn.Linear`。你不需要改动它。
- **`QuantBackend`**：适配层，只处理**单个** `nn.Linear`：怎么量化、怎么造空壳、怎么反量化、怎么读写 state dict。这是你要实现的部分。
- **量化 Linear**：实际承载量化权重并在 `forward` 中完成反量化 + 矩阵乘的模块。

量化 Linear 必须满足四条契约：

- **(a)** 是 `nn.Linear` 的替代品，`forward(x)` 内部完成反量化 + 矩阵乘；且必须是 `torch.nn.Linear` 的子类，否则 LoRA 注入与显存管理无法识别它。
- **(b)** `.to(...)` 只搬设备，不改打包权重与量化状态的 dtype。显存管理会对模型做 dtype 转换，若打包权重被转成 bf16，量化状态就损坏了。
- **(c)** `state_dict()` 与 `load_state_dict(assign=True)` 可以往返，必要时通过 `flatten_state_dict` / `unflatten_state_dict` 转换。
- **(d)**（仅训练需要）`forward` 对输入可微，梯度能穿过冻结的量化层到达 LoRA 分支。

## 第一步：编写量化 Linear

INT9 的存储布局需要一点设计：9bit 没有对应的原生 dtype，如果直接把它塞进 int16 张量，每个权重仍然占 16bit，和 bf16 一样大，量化就白做了。因此这里把每个权重拆成两部分存放——低 8 位放进 uint8 的 `weight`，第 9 位（最高位）单独构成一个位平面，8 个权重打包进 1 个字节存进 `weight_msb`，再加上每个输出通道一个 fp32 的 `weight_scale`。这样每个权重实际占用 9bit，是 bf16 的 56%。

另外注意两个细节：

- 删掉 `nn.Linear` 原有的 `weight` 参数，改为注册同名 buffer，这样 checkpoint 的键名依然是 `层名.weight`，磁盘 offload 与混合量化的键归属判断才能正常工作。
- 通过重写 `_apply` 守护打包张量的 dtype，即契约 (b)。`.to()` / `.half()` / `.float()` 等所有转换都会走到 `_apply`，把会改变 dtype 的转换降级为纯搬设备即可。

```python
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from diffsynth.core.quant import BackendConfig, QuantBackend, register_quant_backend, register_quant_method


def pack_msb(bits):
    """把 0/1 位平面按 8 个权重 1 字节打包，每个权重只占 1bit。"""
    flat = bits.reshape(-1)
    padding = (-flat.numel()) % 8
    if padding:
        flat = torch.cat([flat, flat.new_zeros(padding)])
    groups = flat.view(-1, 8)
    packed = torch.zeros(groups.shape[0], dtype=torch.uint8, device=flat.device)
    for index in range(8):
        packed |= groups[:, index] << index
    return packed


def unpack_msb(packed, numel):
    bits = torch.stack([(packed >> index) & 1 for index in range(8)], dim=1)
    return bits.reshape(-1)[:numel]


class Int9Linear(torch.nn.Linear):
    """int9 权重：低 8 位存在 uint8 的 `weight` 中，第 9 位打包进 `weight_msb`，
    每个输出通道一个 fp32 scale。每个权重占 9bit，是 bf16 的 56%。"""

    dtype_guarded_tensor_names = ("weight", "weight_msb", "weight_scale")

    def __init__(self, in_features, out_features, bias, compute_dtype):
        with torch.device("meta"):
            super().__init__(in_features, out_features, bias=bias, dtype=compute_dtype)
        del self.weight
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.uint8, device="meta"))
        self.register_buffer("weight_msb", torch.empty((in_features * out_features + 7) // 8, dtype=torch.uint8, device="meta"))
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32, device="meta"))
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def _apply(self, fn, recurse=True):
        protected = {id(tensor) for name in self.dtype_guarded_tensor_names
                     if (tensor := getattr(self, name, None)) is not None}

        def guard(tensor):
            converted = fn(tensor)
            if id(tensor) in protected and converted.dtype != tensor.dtype:
                return tensor.to(device=converted.device)
            return converted

        return super()._apply(guard, recurse)

    def dequantize_weight(self, dtype):
        msb = unpack_msb(self.weight_msb, self.weight.numel()).view_as(self.weight)
        codes = self.weight.to(torch.int16) | (msb.to(torch.int16) << 8)
        return ((codes - 256).float() * self.weight_scale.unsqueeze(1)).to(dtype)

    def forward(self, x):
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.dequantize_weight(x.dtype), bias)
```

`forward` 中的反量化用的是常规张量运算，梯度可以经 `F.linear` 传回输入 `x`，因此契约 (d) 自动满足。这里的解包是用 PyTorch 算子逐位拼出来的，只为示例简洁；真实后端通常把解包与矩阵乘融合进一个 kernel，避免每次 forward 都物化一份 fp 权重。

`dequantize_weight` 中有一个容易踩的坑：整数码的还原必须在 fp32 中做。bf16 只有 8bit 有效精度，超过 256 的整数无法精确表示，如果直接把码值转成 bf16 再乘 scale，第 9 位就被舍入掉了，精度收益会白白丢失（实测误差从 int8 的 2.25 倍优势退化到 1.15 倍）。凡是码值位宽超过计算精度的有效位数的量化格式，都要注意这一点。

## 第二步：编写后端

后端的每个方法只处理一层 `nn.Linear`：

- `capabilities()`：声明能力，四个开关默认全为 `False`。`is_serializable=True` 才允许保存量化权重，`is_differentiable=True` 才允许量化 + LoRA 训练。
- `quantized_linear_classes()`：声明本后端产出的 Linear 类，`is_quantized_linear` 默认用它做 `isinstance` 判断。
- `create_quantized_linear()`：在线量化，把 fp 的 `nn.Linear` 变成量化 Linear。`compute_device` 是量化计算所在设备，`model_device` 是量化完成后存放的设备，两者配合可以逐层流式量化，显存里一次只放一层。
- `create_quantized_linear_shell()`：造一个空壳，用于加载预量化 checkpoint 以及磁盘 offload。空壳会在每次 offload 时重建，所以要建在 `meta` 设备上，保持廉价。
- `dequantize_to_linear()`：反量化回普通 `nn.Linear`，供 `mode="dequant_once"` 使用。
- `flatten_state_dict` / `unflatten_state_dict`：state dict 与扁平张量之间的转换。INT9 的 state dict 本身就是普通张量，直接用基类实现即可，无需重写；只有像 bitsandbytes、torchao 那样含复合张量（张量子类、嵌套量化状态）的后端才需要重写。

未实现的方法会由基类抛出带说明的异常，因此只支持部分能力的后端只实现自己需要的即可。`self.config` 是框架注入的后端配置实例，即下一步要写的 `Int9WeightOnlyConfig`。

```python
@register_quant_backend("toy_int9")
class Int9QuantBackend(QuantBackend):
    project_url = "https://example.com/toy-int9"

    def capabilities(self):
        return {**super().capabilities(), "is_serializable": True, "is_differentiable": True}

    def quantized_linear_classes(self):
        return (Int9Linear,)

    def create_quantized_linear(self, linear, compute_device=None, model_device=None):
        weight = linear.weight.data
        if compute_device is not None:
            weight = weight.to(device=compute_device)
        amax = weight.abs().amax(dim=1) if self.config.per_channel else weight.abs().amax().expand(weight.shape[0])
        scale = (amax.float() / 255).clamp(min=1e-8)
        codes = (weight.float() / scale.unsqueeze(1)).round().clamp(-256, 255).to(torch.int16) + 256

        quant_linear = Int9Linear(linear.in_features, linear.out_features, bias=linear.bias is not None, compute_dtype=weight.dtype)
        quant_linear.weight = (codes & 0xFF).to(torch.uint8)
        quant_linear.weight_msb = pack_msb((codes >> 8).to(torch.uint8))
        quant_linear.weight_scale = scale
        if linear.bias is not None:
            quant_linear.bias = torch.nn.Parameter(linear.bias.data.to(device=scale.device), requires_grad=False)
        return quant_linear if model_device is None else quant_linear.to(device=model_device)

    def create_quantized_linear_shell(self, linear, compute_dtype):
        return Int9Linear(linear.in_features, linear.out_features, bias=linear.bias is not None, compute_dtype=compute_dtype)

    def dequantize_to_linear(self, module, compute_dtype, compute_device=None, model_device=None):
        if compute_device is not None:
            module = module.to(device=compute_device)
        fp_weight = module.dequantize_weight(compute_dtype)
        linear = torch.nn.Linear(module.in_features, module.out_features, bias=module.bias is not None, device="meta")
        linear.weight = torch.nn.Parameter(fp_weight, requires_grad=False)
        if module.bias is not None:
            linear.bias = torch.nn.Parameter(module.bias.data.to(dtype=compute_dtype, device=fp_weight.device), requires_grad=False)
        return linear if model_device is None else linear.to(device=model_device)
```

## 第三步：编写后端配置

后端配置继承 `BackendConfig`：用户可调的参数写成普通 dataclass 字段，由方法固定、不允许用户覆盖的值用 `field(init=False, default=...)` 声明。`describe_quant_method` 会分别展示这两类参数，`from_kwargs` 则会在用户传入未知的 `backend_config_kwargs` 时报错。

```python
@dataclass
class Int9WeightOnlyConfig(BackendConfig):
    per_channel: bool = True                    # 用户可调：per-channel 还是 per-tensor
    bits: int = field(init=False, default=9)     # 方法固定，不可覆盖
```

## 第四步：注册量化方法

一个后端可以注册多个方法，通过配置中被固定的字段区分（例如 bitsandbytes 后端用 `quant_type` 区分 nf4 与 fp4）。方法名建议遵循 `<后端>_<格式>_w<权重位宽>a<激活位宽>` 的命名约定：

```python
register_quant_method("toy_int9_w9a16", "toy_int9", Int9WeightOnlyConfig.from_kwargs, label="9bit, int9, weight-only (toy)")
```

注册后端和方法有两种方式：

**方式一：写在自己的代码里（推荐，即插即用）**。把上面的代码放在任意模块中，只要在构造 `QuantizeConfig` 之前 import 过这个模块，方法就已经注册进 `QUANT_METHODS`，可以像内置方法一样使用，无需改动框架代码：

```python
import my_project.toy_int9   # 触发 register_quant_backend / register_quant_method

from diffsynth.core.quant import QuantizeConfig

quantize = QuantizeConfig(method="toy_int9_w9a16", backend_config_kwargs={"per_channel": True})
```

**方式二：作为内置后端（永久生效）**。把后端文件放到 `diffsynth/core/quant/backends/` 下，并在 `diffsynth/core/quant/backends/__init__.py` 的 `_LAZY_BACKENDS` 中登记，框架就会在需要时按需 import，用户无需手动 import：

```python
_LAZY_BACKENDS = {
    "bitsandbytes": ".bitsandbytes",
    "torchao": ".torchao",
    "comfy_kitchen": ".comfy_kitchen",
    "toy_int9": ".toy_int9",
}
```

如果你的量化算法或量化库有通用价值，欢迎按方式二提 PR 给我们，让更多用户直接用上。需要第三方依赖的后端请在 `validate_environment()` 中检查依赖并给出安装提示，在 `project_url` 中填写上游项目地址。

## 第五步：自检

框架提供了两个自检工具，建议在接入后立刻跑一遍。`check_backend_contract` 会检查后端是否声明了 Linear 类、两个工厂方法是否返回声明的类、所有类是否都是 `nn.Linear` 的子类，以及后端实际写出的 checkpoint 键是否都落在层名之下（漏掉一个 scale 会让磁盘 offload 静默加载出损坏的层）。不支持的工厂方法会被跳过，不计为失败。

```python
from diffsynth.core.quant import QUANT_BACKENDS, QUANT_METHODS, check_backend_contract, check_differentiable, describe_quant_method

describe_quant_method("toy_int9_w9a16")

spec = QUANT_METHODS["toy_int9_w9a16"]
check_backend_contract(QUANT_BACKENDS[spec.backend](spec.config_factory({})), compute_device="cpu")
```

输出如下，`describe_quant_method` 同时验证了用户可调参数与固定参数的划分是否符合预期：

```
method: toy_int9_w9a16
backend: toy_int9
detail: 9bit, int9, weight-only (toy)
backend config: my_project.toy_int9.Int9WeightOnlyConfig
backend_config_kwargs (user-tunable):
  per_channel = True
pinned by method (not overridable):
  bits = 9
check_backend_contract (toy_int9):
  [PASS] quantized_linear_classes() is non-empty: ['Int9Linear']
  [PASS] Int9Linear subclasses torch.nn.Linear
  [PASS] a plain nn.Linear is not reported as quantized
  [PASS] create_quantized_linear_shell() returns a declared class, got Int9Linear
  [PASS] the shell is recognized before load_state_dict (disk offload routing)
  [PASS] create_quantized_linear() returns a declared class, got Int9Linear
  [PASS] every stored key lives under the layer name; uncovered: []
  => OK
```

接着在一个小模型上验证数值误差、真实的显存收益、契约 (b) 的 dtype 守护、以及契约 (d) 的可微性：

```python
import torch
from diffsynth.core.quant import QuantizeConfig, check_differentiable


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(256, 512)
        self.fc2 = torch.nn.Linear(512, 256, bias=False)

    def forward(self, x):
        return self.fc2(torch.nn.functional.silu(self.fc1(x)))


def footprint(model):
    return sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))


torch.manual_seed(0)
model = ToyModel().to(torch.bfloat16)
x = torch.randn(4, 256, dtype=torch.bfloat16)
reference = model(x)
fp_bytes = footprint(model)

QuantizeConfig(method="toy_int9_w9a16").quantize_model(model, compute_device="cpu")
print("relative error:", ((model(x) - reference).norm() / reference.norm()).item())
print(f"footprint: {fp_bytes} -> {footprint(model)} bytes ({footprint(model) / fp_bytes:.3f} of bf16)")

model.to(torch.float32)                                   # 契约 (b)：打包权重的 dtype 不应改变
print(model.fc1.weight.dtype, model.fc1.weight_msb.dtype, model.fc1.weight_scale.dtype, model.fc1.bias.dtype)

check_differentiable(model.fc1)                           # 契约 (d)
```

```
2 nn.Linear layers quantized (method: toy_int9_w9a16).
relative error: 0.004150390625
footprint: 525312 -> 299008 bytes (0.569 of bf16)
torch.uint8 torch.uint8 torch.float32 torch.float32
check_differentiable (Int9Linear): OK -- gradients pass through the module to its input
```

实测占用是 bf16 的 0.569，略高于 9/16 = 0.5625，差值来自 fp32 的 scale 和未量化的 bias。如果这个比例接近 1，说明打包格式没有真正压缩权重，需要回到第一步检查存储布局。

### 在真实模型上推理：Z-Image

小模型验证通过后，就可以直接在真实模型上用了——自定义后端和内置方法的用法完全一致，只要在构造 `QuantizeConfig` 之前 import 过注册后端的模块，把它传给 `ModelConfig(quantize=...)` 即可：

```python
import torch

import my_project.toy_int9   # 注册 toy_int9 后端与 toy_int9_w9a16 方法
from diffsynth.core.quant import QuantizeConfig
from diffsynth.pipelines.z_image import ModelConfig, ZImagePipeline

pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="Tongyi-MAI/Z-Image-Turbo",
            origin_file_pattern="transformer/*.safetensors",
            quantize=QuantizeConfig(method="toy_int9_w9a16"),
        ),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
)

dit_bytes = sum(t.numel() * t.element_size() for t in list(pipe.dit.parameters()) + list(pipe.dit.buffers()))
print(f"dit weights: {dit_bytes / 1024 ** 3:.3f} GiB")

prompt = "A delicate portrait of an underwater girl, blue dress flowing, hair gently drifting, light and shadow clear, surrounded by bubbles, serene expression, exquisite details, dreamlike and beautiful."
image = pipe(prompt=prompt, seed=42, rand_device="cuda")
image.save("z_image_toy_int9.jpg")
```

实测 Z-Image Turbo 的 DiT 权重占用（8 步 Turbo 出图正常，画质与 bf16 无明显差异）：

| | DiT 权重 |
| --- | --- |
| bf16 | 11.464 GiB |
| `toy_int9_w9a16` | 6.456 GiB（0.563×） |

需要注意峰值显存与权重占用不是一回事：这个 toy 的 forward 每次都会临时物化一份 fp 权重，所以峰值的节省会小于权重的节省。在一个 48 个 Linear 的合成模型上（权重全部常驻 GPU）实测：

| | 权重 | forward 峰值 |
| --- | --- | --- |
| bf16 | 1.500 GiB | 1.527 GiB |
| `toy_int9_w9a16` | 0.845 GiB（0.563×） | 1.036 GiB（0.678×） |

这份临时权重只与**最大的那一层**有关，不随层数增长，所以模型越深、收益越接近权重的比例；真实后端把解包与矩阵乘融进一个 kernel 后就不需要它了。想进一步压低峰值，可以叠加[显存管理](../Pipeline_Usage/VRAM_management.md)按层搬运权重（把 `vram_config` 传给上面的每个 `ModelConfig`，实测峰值可降到 2.1 GiB）。

### 精度对比：int9 vs int8

多出来的 1 bit 是否真的换来了精度？把同一个权重用**完全相同**的 per-channel 对称量化方案分别做 8bit 与 9bit，对比反量化后的权重误差与层输出误差即可。这也是给新后端做精度回归的通用做法：控制其他变量，只改位宽。

```python
import torch
from my_project.toy_int9 import Int9QuantBackend, Int9WeightOnlyConfig


def quantize_int8(linear):
    """同样的 per-channel 对称方案，只少 1 bit：码值范围 [-128, 127]。"""
    weight = linear.weight.data
    scale = (weight.abs().amax(dim=1).float() / 127).clamp(min=1e-8)
    codes = (weight.float() / scale.unsqueeze(1)).round().clamp(-128, 127)
    return (codes * scale.unsqueeze(1)).to(weight.dtype)


def relative_error(reference, value):
    return ((value.float() - reference.float()).norm() / reference.float().norm()).item()


torch.manual_seed(0)
backend = Int9QuantBackend(Int9WeightOnlyConfig())
linear = torch.nn.Linear(2048, 2048, bias=False).to(torch.bfloat16)
fp_weight = linear.weight.data.clone()

int9_weight = backend.create_quantized_linear(linear).dequantize_weight(torch.bfloat16)
int8_weight = quantize_int8(linear)
error8, error9 = relative_error(fp_weight, int8_weight), relative_error(fp_weight, int9_weight)
print(f"weight error: int8 {error8:.6f} | int9 {error9:.6f} ({error8 / error9:.2f}x lower)")

x = torch.randn(64, 2048, dtype=torch.bfloat16)
reference = torch.nn.functional.linear(x, fp_weight)
out8 = relative_error(reference, torch.nn.functional.linear(x, int8_weight))
out9 = relative_error(reference, torch.nn.functional.linear(x, int9_weight))
print(f"output error: int8 {out8:.6f} | int9 {out9:.6f} ({out8 / out9:.2f}x lower)")
```

```
weight error: int8 0.004353 | int9 0.001937 (2.25x lower)
output error: int8 0.004947 | int9 0.002816 (1.76x lower)
```

结论符合预期：码值范围从 255 级扩到 511 级，量化步长减半，权重误差随之降到约 1/2（实测 2.25 倍，均匀量化下误差与步长成正比）。端到端的层输出收益略小（1.76 倍），因为激活值本身是 bf16，矩阵乘自带的舍入噪声会占掉一部分收益——这也提示：位宽收益要放到实际计算精度下评估，而不是只看权重误差。

最后验证契约 (c)：保存量化权重，再用空壳加载回来，两者的输出应完全一致。

```python
from safetensors.torch import load_file, save_file

save_config = QuantizeConfig(method="toy_int9_w9a16")
tensors, metadata = save_config.flatten_state_dict(model.state_dict())
save_file(tensors, "toy_int9.safetensors", metadata=metadata)

loaded = ToyModel().to(torch.bfloat16)
load_config = QuantizeConfig(method="toy_int9_w9a16", load_prequantized=True)
load_config.prepare_for_prequantized_load(loaded, compute_dtype=torch.bfloat16)
loaded.load_state_dict(load_config.unflatten_state_dict(load_file("toy_int9.safetensors"), metadata), assign=True)
print("reload match:", torch.equal(loaded(x.float()), model(x.float())))
```

```
reload match: True
```

### 与 Disk Offload 组合验证

[显存管理](../Pipeline_Usage/VRAM_management.md)中的 Disk Offload 对量化后端的要求最严格：模型常驻内存中只保留 `meta` 空壳，每次 forward 时才按层把张量从磁盘流式读回来，用完即丢。它依赖两件事：

- 只支持**预量化 checkpoint**，因此必须 `load_prequantized=True`，并先经 `prepare_for_prequantized_load` 把目标层换成空壳。
- 某一层需要哪些张量，是用层的点分名做前缀扫描从 checkpoint 键里找出来的，然后以 `load_state_dict(assign=True)` 严格加载。因此后端只要满足「所有张量都在 `层名.` 之下」这一条（无论是 `层名.weight_scale` 这样的平级张量，还是 bnb 那样的嵌套量化状态），就能被正确切分；键少了或多了会直接报错，而不会静默加载出错误的层。

```python
import torch
from safetensors.torch import save_file

from diffsynth.core.loader.model import load_metadata_from_safetensors
from diffsynth.core.quant import QuantizeConfig
from diffsynth.core.vram.disk_map import DiskMap
from diffsynth.core.vram.layers import AutoWrappedLinear, enable_vram_management_recursively

resident = ToyModel().to(torch.bfloat16)
x = torch.randn(2, 256, dtype=torch.bfloat16, device="cuda")

save_config = QuantizeConfig(method="toy_int9_w9a16")
save_config.quantize_model(resident, compute_device="cuda")
resident = resident.to("cuda")
reference = resident(x)

tensors, metadata = save_config.flatten_state_dict(resident.state_dict())
save_file({key: value.cpu() for key, value in tensors.items()}, "toy_int9.safetensors", metadata=metadata)

fresh = ToyModel().to(torch.bfloat16)
load_config = QuantizeConfig(method="toy_int9_w9a16", load_prequantized=True)
load_config.prepare_for_prequantized_load(fresh, compute_dtype=torch.bfloat16)
enable_vram_management_recursively(
    fresh,
    module_map={torch.nn.Linear: AutoWrappedLinear},
    vram_config={
        "offload_dtype": "disk", "offload_device": "disk",
        "onload_dtype": "disk", "onload_device": "disk",
        "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16, "computation_device": "cuda",
    },
    disk_map=DiskMap(["toy_int9.safetensors"], "cuda", torch_dtype=None),
    quantize=load_config,
    metadata=load_metadata_from_safetensors("toy_int9.safetensors"),
)

for name, module in fresh.named_modules():
    if getattr(module, "disk_offload", False):
        print(f"{name}: {module._disk_required_keys()}")

resident_bytes = sum(t.numel() * t.element_size() for t in list(resident.parameters()) + list(resident.buffers()))
offloaded_bytes = sum(t.numel() * t.element_size() for t in list(fresh.parameters()) + list(fresh.buffers()) if not t.is_meta)
print(f"resident {resident_bytes} bytes -> in memory after disk offload {offloaded_bytes} bytes")
print("output matches:", torch.equal(fresh(x), reference), "| repeatable:", torch.equal(fresh(x), reference))
```

在前面那个 `ToyModel` 上实测（`DiskMap` 的 `torch_dtype=None` 很关键，它保证打包张量不会在读取时被转换精度）：

```
2 nn.Linear layers replaced for loading the pre-quantized checkpoint (method: toy_int9_w9a16).
fc1: ['fc1.bias', 'fc1.weight', 'fc1.weight_msb', 'fc1.weight_scale']
fc2: ['fc2.weight', 'fc2.weight_msb', 'fc2.weight_scale']
resident 299008 bytes -> in memory after disk offload 0 bytes
output matches: True | repeatable: True
```

每层的 `weight` / `weight_msb` / `weight_scale` / `bias` 都被正确归到该层名下，常驻占用降到 0 字节（全部是 `meta` 空壳），输出与常驻量化模型逐位相同，且多次 forward 结果稳定——说明空壳的反复重建与流式加载没有副作用。

在真实模型上，则可以用[模型量化](../Pipeline_Usage/Quantization.md)中的通用流程做端到端验证：把 `QuantizeConfig(method="toy_int9_w9a16")` 传给 `ModelConfig(quantize=...)` 做在线量化推理，用 `save_quantized_model` 保存量化权重并注册 hash 后加载，以及在量化模型上注入 LoRA 训练。

## 接入检查清单

- 打包格式真的减小了权重体积：量化前后实测占用之比应接近理论位宽比，而不是接近 1。
- 精度收益经过验证：与少 1 bit 的同方案对比，误差确实下降；否则说明反量化路径中丢失了精度。
- 量化 Linear 是 `torch.nn.Linear` 的子类，`state_dict` 的键都在层名之下。
- `_apply` 守护了所有打包张量与量化状态的 dtype。
- `capabilities()` 与实际能力一致：声明 `is_serializable` 就要保证 state dict 能往返，声明 `is_differentiable` 就要能通过 `check_differentiable`。
- `create_quantized_linear` 尊重 `compute_device` / `model_device`，以支持逐层流式量化。
- 能与 Disk Offload 组合：空壳建在 `meta` 上且重建代价低，所有张量都在层名之下，且 `unflatten_state_dict` 能接受「单层子字典 + 整文件 metadata」的调用方式。
- 依赖第三方库时，`validate_environment()` 给出明确的安装提示，`project_url` 指向上游项目。
- `check_backend_contract` 全部通过。
