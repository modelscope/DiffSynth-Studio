# 实施计划：量化后端契约 + comfy-kitchen int8 后端接入

> 自包含施工说明，不需要读其他报告。所有「实测」结论均在
> A100-80GB / torch 2.10.0+cu128 / triton 3.6.0 / driver 570.133.20 / py3.10 上验证过，
> **请直接采信，不要重新试错**。验证脚本在 `workspace/20260808_comfy_kitchen_backend/`。
>
> ⚠️ 本计划涉及共用框架代码（`core/quant`、`core/vram`）与三个现有后端。
> **每一处 core 改动动手前把 diff 给用户过一遍**（项目规则）。

---

## 0. 目标与范围

分两部分，**Part A 先落地**（它定义了 Part B 要遵守的接口）：

| | 内容 | 产出 |
| --- | --- | --- |
| **Part A** | 给 `QuantBackend` 加一条**可机器验证的契约**：每个后端必须声明自己产出的 Linear 类；据此改造 bitsandbytes / torchao / ideogram4_fp8 | 判据统一到一处，顺带修掉两个现存静默 bug |
| **Part B** | 新增 comfy-kitchen 后端，支持 ComfyUI 的 `int8_tensorwise` 格式（预量化加载 + 在线动态量化） | MiniMax-H3 int8_convrot 能跑通 |

### 明确不做

- ❌ fp8 / nvfp4 / mxfp8 / convrot_w4a4（架构留口，见 §B7）
- ❌ `pruned` 变体（那是 adaLN 曲线表的模型结构改动，与量化无关）
- ❌ 训练 / QLoRA（comfy-kitchen 的 `QuantizedTensor` 前向不可微）
- ❌ disk offload（需要额外 core 改动，见 §A6）

---

# Part A：量化后端契约

## A1. 问题：三种判据，两个静默 bug

`enable_vram_management_recursively`（`core/vram/layers.py:558-575`）靠
`quantize.is_quantized_linear(module)` 决定一个层走哪条路：

```python
if quantize is not None and quantize.is_quantized_linear(module):
    module_ = AutoWrappedQuantizedModule(...)      # 量化路径
    continue
for source_module, target_module in module_map.items():
    if isinstance(module, source_module):          # torch.nn.Linear -> AutoWrappedLinear
        module_ = target_module(...)               # 普通路径
```

**判错不报错** —— 量化层被当普通层包进 `AutoWrappedLinear`，输出 finite、无异常（已实测）。

现有三个后端用了三种判据，可靠性完全不同：

| 后端 | 判据 | shell 阶段（`load_state_dict` 之前）能认出吗 |
| --- | --- | --- |
| bitsandbytes | `quantized_linear_classes()` → isinstance `Linear4bit` | ✅ 能 |
| **torchao** | 看 weight 类型字符串 `"torchao" in type(weight).__module__` | ❌ **不能**，shell 是普通 `nn.Linear`，weight 是普通 `Parameter` |
| ideogram4_fp8 | isinstance `Fp8Linear` | ✅ 能，但 `Fp8Linear` 不是 `nn.Linear` 子类 |

而 `load_model` 的 **disk offload 分支**（`core/loader/model.py:34-42`）建壳后**立刻**调
`enable_vram_management`，中间没有 `load_state_dict`。实测后果：

```
torchao_int8_w8a16 : shell 类型=Linear      is_quantized_linear=False -> AutoWrappedLinear         ❌ 静默走错
bitsandbytes_nf4   : shell 类型=Linear4bit  is_quantized_linear=True  -> AutoWrappedQuantizedModule ✅
```

**Bug 1**：torchao + disk offload 静默走错路径。
**Bug 2**：`ideogram4_fp8` + disk offload 缺 `weight_scale`（见 §A6，`_disk_required_keys` 只收 `weight` / `weight.*` / `bias`）。

## A2. 契约定义（两个**必需**方法）

改 `diffsynth/core/quant/base.py`：

```python
class QuantBackend(ABC):
    def quantized_linear_classes(self) -> tuple:
        """The Linear classes this backend produces. MUST be non-empty.

        Both `create_quantized_linear` and `create_quantized_linear_shell` must return an
        instance of one of these, and each must subclass `torch.nn.Linear`. A quantized
        layer is then recognizable by isinstance alone -- before its weights are loaded,
        after any `.to()`, and after offload -- which is what VRAM management relies on.
        """
        raise NotImplementedError(
            f"Backend `{self.name}` must declare the Linear classes it produces."
        )

    def checkpoint_key_patterns(self) -> tuple:
        """Checkpoint entries one quantized Linear needs, relative to its dotted name.

        Each item is either an exact relative key ("weight_scale") or a prefix ending in
        "." ("weight.") meaning that entry and everything nested below it. Disk offload
        uses this to fetch exactly one layer out of a whole-file key index, so every key
        `unflatten_state_dict` consumes must be listed here.

        The default covers backends that nest their quant state under the weight
        (bitsandbytes: `<layer>.weight.absmax`, ...). Backends whose side tensors are
        siblings of the weight (`<layer>.weight_scale`, `<layer>.comfy_quant`, ...) must
        override it, or disk offload will silently load those layers without their scales.
        """
        return ("weight", "weight.", "bias")
```

配套改动：

1. **`is_quantized_linear` 收敛到基类唯一实现**，删掉所有 per-backend 覆盖：
   ```python
   def is_quantized_linear(self, module) -> bool:
       return isinstance(module, self.quantized_linear_classes())
   ```
   删除 `backends/torchao.py:29-31` 的覆盖。
2. `quantized_linear_classes()` 从「可选，默认 `()`」变成「必需」。仓库内只有 3 个后端，迁移面可控。
3. **新增 `QuantizeConfig.checkpoint_keys()` 与 `build_quantized_shell()`**，把 `core/vram/layers.py`
   对 backend 的反向依赖摘干净（见 §A2.1）。

**契约的子要求**（都要被自检覆盖）：

- (a) 两个工厂方法返回的实例都 `isinstance` 于声明的类
- (b) 声明的每个类都是 `torch.nn.Linear` 的子类
- (c) `checkpoint_key_patterns()` 能覆盖该后端存盘时产生的全部键

(b) 的收益：`auto_detect_lora_target_modules`（`diffsynth/diffusion/training_module.py:231` 的
`isinstance(module, torch.nn.Linear)`）能识别量化层，peft 也能注入。

### A2.1 `core/vram/layers.py` 的三行替换

现状两个问题：

1. **`_disk_required_keys()`（`layers.py:484-491`）硬编码了 bitsandbytes 的嵌套键形**：只收
   `<layer>.weight` / `<layer>.weight.*` / `<layer>.bias`。`weight_scale` / `input_scale` /
   `comfy_quant` 这类**下划线兄弟键**收不到 ⇒ `ideogram4_fp8` 与 comfy-kitchen 的 disk offload
   会静默缺 scale。
2. **`layers.py:512` 与 `538` 直接访问 `self.quantize.backend`**，而 `MixedQuantizeConfig`
   没有 `.backend` 属性 ⇒ 混合量化 + disk offload 现在就是 `AttributeError`。

改法：

```python
# core/quant/config.py — QuantizeConfig / MixedQuantizeConfig 各新增
def checkpoint_keys(self, module, layer_name, available_keys) -> list[str]:
    """Resolve the backend's `checkpoint_key_patterns` for `layer_name` against
    `available_keys` (anything supporting `in` and iteration, including `DiskMap`).
    Raises when the packed weight is absent -- silently loading a layer without its
    scale corrupts it without any error."""

def build_quantized_shell(self, module, compute_dtype):
    """Delegate to the backend's `create_quantized_linear_shell`."""
#   Mixed 版两个方法都按 `config.is_quantized_linear(module)` 派发到拥有该层的子 config

# core/vram/layers.py — 三行替换
# _disk_required_keys                        -> self.quantize.checkpoint_keys(self.module, self.name, self.disk_map)
# 两处 .backend.create_quantized_linear_shell -> self.quantize.build_quantized_shell(...)
```

`DiskMap` 已有 `__iter__` / `__contains__`，精确键走 O(1) 的 `in`；只有带 `.` 前缀的模式
需要扫一遍键表（也就是只有 bitsandbytes 付这个代价）。

## A3. 契约自检（机器验证，这是契约成立的关键）

放在 `core/quant/base.py`，与现有的 `check_differentiable` 并列：

```python
def check_backend_contract(backend, in_features: int = 512, out_features: int = 512,
                           compute_dtype: torch.dtype = torch.bfloat16,
                           compute_device: str = "cuda", verbose: bool = True) -> bool:
    """Verify a backend satisfies the quantized-Linear contract: it declares its classes,
    both factory methods return instances of them, and every declared class subclasses
    `torch.nn.Linear` so LoRA target detection and VRAM management can see it.
    """
```

检查项：

1. `quantized_linear_classes()` 非空
2. 每个类 `issubclass(cls, torch.nn.Linear)`
3. `create_quantized_linear_shell(plain_linear, dtype)` 的返回值 isinstance 于声明的类
4. `create_quantized_linear(plain_linear, ...)` 的返回值同上（后端不支持在线量化时跳过，不算失败）
5. shell 上 `is_quantized_linear()` 为 True（**这一条直接锁死 Bug 1**）
6. 普通 `nn.Linear` 上 `is_quantized_linear()` 为 False
7. **`checkpoint_key_patterns()` 非空且能覆盖存盘键**（**锁死 Bug 2**）：
   拿一个真实量化层跑 `flatten_state_dict`（或直接取其 `state_dict()`）得到存盘键集合，
   断言每一个键都能被 `checkpoint_keys()` 解析出来；反之则该后端的 disk offload 一定缺料。

## A4. 三个后端的改造

### A4.1 bitsandbytes —— 已符合，零改动

`bnb.nn.Linear4bit` 本身就是 `nn.Linear` 子类，`quantized_linear_classes()` 已返回它；
quant state 嵌在 `<layer>.weight.absmax` 等键下，**恰好是 `checkpoint_key_patterns()` 的默认值**。
仅需跑一次自检确认，以及 §A5 的 disk offload 回归（**它是唯一有现成示例覆盖 disk 路径的后端**）。

### A4.2 torchao —— 加一个空标记类

```python
class TorchaoLinear(torch.nn.Linear):
    """Marker class for torchao-quantized Linears.

    torchao keeps the quantization in the weight tensor subclass, so forward needs no
    override; the class exists so the layer is recognizable by isinstance even before
    `load_state_dict` fills in the quantized weight.
    """
```

改三处：

| 方法 | 改动 |
| --- | --- |
| `create_quantized_linear` | `quantize_` 后把 weight 装进 `TorchaoLinear`（或直接在 `TorchaoLinear` 实例上调 `quantize_`，两者等价） |
| `create_quantized_linear_shell` | 返回 `TorchaoLinear(..., device="meta")` 而不是 `torch.nn.Linear` |
| `quantized_linear_classes` | 返回 `(TorchaoLinear,)` |
| `is_quantized_linear` | **删掉**（用基类实现） |
| `checkpoint_key_patterns` | torchao 的 safetensors 存盘键形需**实测确认**后再定：跑一层的 `flatten_state_dict` 看它产生哪些键（`torchao.prototype.safetensors` 会拆出 `_data` / `_scale` 类后缀），再决定默认值够不够。**不要猜** |

`dequantize_to_linear` 不用改（它本来就返回普通 `nn.Linear`，语义正确）。

**已实测**（`probe_torchao_detection.py`）：三条路线数值完全相同 `rel_l2=0.00898`：

| 路线 | rel_l2 |
| --- | --- |
| 原做法（普通 `nn.Linear` + `quantize_`） | 0.00898 |
| 量化后把 weight 搬进 `TorchaoLinear` 壳 | 0.00898 |
| 直接对 `TorchaoLinear` 实例调 `quantize_` | 0.00898 |

`state_dict` 键仍是 `['weight']` ⇒ **磁盘格式零破坏**；`.to(dtype)` 后 weight 仍是 `Int8Tensor`。

### A4.3 ideogram4_fp8 —— 改成 `nn.Linear` 子类（已实测可行）

`Fp8Linear` 当前是纯 `torch.nn.Module`。改造后：

```python
class Fp8Linear(torch.nn.Linear):
    """Linear layer holding an e4m3 float8 weight + per-row float32 scale."""

    dtype_guarded_tensor_names: tuple = ("weight", "weight_scale")

    def __init__(self, in_features, out_features, bias, compute_dtype):
        with torch.device("meta"):
            super().__init__(in_features, out_features, bias=bias, dtype=compute_dtype)
        del self.weight                      # drop the float Parameter nn.Linear created
        self.compute_dtype = compute_dtype
        self.register_buffer("weight", torch.empty(out_features, in_features,
                                                   dtype=FP8_WEIGHT_DTYPE, device="meta"))
        self.register_buffer("weight_scale",
                             torch.empty(out_features, dtype=torch.float32, device="meta"))
        if self.bias is not None:
            self.bias.requires_grad_(False)   # nn.Linear creates it trainable by default

    # _apply dtype guard and forward stay exactly as they are today
```

**已实测**（`probe_fp8linear_contract.py`）当前实现与改造版逐项对比：

| | 当前（纯 `nn.Module`） | 改造版（`nn.Linear` 子类） |
| --- | --- | --- |
| `isinstance(nn.Linear)` | False | **True** ✓ |
| forward rel_l2（bias=False / True） | 0.02669 / 0.01435 | **完全相同** ✓ |
| `state_dict` 键 | `['weight','weight_scale']` / `[+'bias']` | **完全相同** ✓ |
| `.to(bfloat16)` dtype 守卫 | GUARD HOLDS | **GUARD HOLDS** ✓ |
| cpu→cuda 往返 | 一致 | 一致 ✓ |
| bias 存储 | buffer | Parameter（键名不变） |

⚠️ **唯一行为差异**：bias 从 buffer 变 Parameter，默认 `requires_grad=True`
⇒ 上面代码里那句 `requires_grad_(False)` **不能省**，否则会意外进入优化器的可训练集合。

⚠️ `del self.weight` 的顺序不能变：必须先删 `nn.Linear` 建的 float Parameter，再
`register_buffer` 同名 buffer（同名冲突会报错）。

另需新增 `checkpoint_key_patterns`：

```python
def checkpoint_key_patterns(self):
    return ("weight", "weight_scale", "bias")      # 兄弟键，不是 weight 子树
```

这一行就是 **Bug 2 的修正** —— `ideogram4_fp8` 的 disk offload 以前收不到 `weight_scale`。

### A4.4 关键实现细节（三个后端共用，容易踩）

**① `module_map` 的顺序依赖必须写注释。** 任何 `nn.Linear` 子类都会同时匹配 `module_map` 里的
`torch.nn.Linear → AutoWrappedLinear`；正确性依赖 `layers.py:563` 的 quantize 检查在 568 行的
循环**之前**。这个顺序是正确性的一部分，代码里目前**没有任何注释**说明 —— 谁重排一下就静默坏掉。
建议在 `enable_vram_management_recursively` 里加一句注释锁住这个前提。

**② 类的粒度是 per-backend，不是 per-format。** 一个后端一个类 + 字段区分格式即可
（如 comfy-kitchen 用 `layout` 字段）。从来不需要 isinstance-per-format。

**③ 重复量化的防护。** 契约后所有量化层都是 `nn.Linear` 子类，
`QuantizeConfig._should_quantize` 的 `isinstance(module, torch.nn.Linear)` 会命中它们
⇒ 对同一模型重复调 `quantize_model()` 会二次量化。建议加一行守卫：
```python
if self.is_quantized_linear(module):
    return False
```

## A5. Part A 的回归验证

torchao 与 ideogram4_fp8 现在的**非 disk 路径是好的**，改造必须证明没退化。

### A5.1 单层四条路径（每个后端）

| 路径 | 断言 |
| --- | --- |
| 在线量化（`load_prequantized=False`） | 输出与改造前逐元素一致或 rel_l2 差异 < 1e-6 |
| 预量化加载（`load_prequantized=True`） | 同上；且 `state_dict` 键集合完全不变 |
| `mode="dequant_once"` | 还原后是普通 `nn.Linear`，`is_quantized_linear` 为 False |
| CPU offload（`enable_vram_management`，非 disk） | routing 是 `AutoWrappedQuantizedModule`，前向数值不变 |

加上 `check_backend_contract` 对 3 个后端全绿。

**基线怎么取**：改造前先把四条路径的输出张量存盘（`workspace/20260808_comfy_kitchen_backend/baseline/`），
改造后逐一比对。不要凭肉眼看"差不多"。

### A5.2 ❗ NF4 disk offload 端到端回归（本次改动的关键验收）

我们改了 `_disk_required_keys` 与 shell 构造的调用方式，而 **NF4 是现在唯一真在跑 disk offload
的后端** —— 所以它是防止回归的主防线。基准脚本：

```
examples/minimax_h3/model_inference_low_vram/MiniMax-H3-NF4-FL2VA.py
```

它的 vram_config 就是 disk offload 的正宗写法，且 **DiT / TE / video VAE / audio VAE 四个模型全部**
走 `offload_device="disk"`：

```python
vram_config = {
    "offload_dtype": "disk",  "offload_device": "disk",
    "onload_dtype": torch.bfloat16, "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16, "computation_device": "cuda",
}
... vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
```

这条路径会贯穿我们改动的全部代码：
`AutoWrappedQuantizedModule(disk_offload=True)` → `_disk_required_keys`（→ `checkpoint_keys`）
→ `_load_from_disk` → `unflatten_state_dict` → `offload()` / `computation_module()` 里的
`create_quantized_linear_shell`（→ `build_quantized_shell`）。

**执行方式**：按项目惯例做最小必要测试 —— 先把脚本里的 `num_inference_steps` 改小（如 4）
跑通链路，再跑一次原参数。产物存本任务目录。

**验收**：

1. 改造**前**先跑一次，存下视频/音频产物与日志作基线（同 seed）
2. 改造**后**同 seed 重跑，产物应与基线一致（允许 bf16 舍入级差异）
3. 日志里 4 个模型都正确报出量化层数量，无 missing / unexpected keys
4. 峰值显存与改造前同量级（证明 disk 路径真的在走，没有退化成全量驻留）

⚠️ **不要用 `examples/minimax_h3/model_training/lora/MiniMax-H3-NF4-FL2VA.sh` 验证 disk offload** ——
那是训练脚本，而且它**没有传 `--offload_models`**，所以根本不会进 disk 分支
（训练侧的 disk 配置在 `training_module.py:164-175`，由 `--offload_models` 触发，
目前全仓库**没有任何 .sh 脚本用过它**）。

### A5.3 ❗ 动态（在线）量化回归 —— 用 Z-Image-Turbo

前面两节盖的是预量化加载；**在线量化是另一条独立代码路径**
（`create_quantized_linear` → `quantize_model`），必须单独回归。

基准驱动脚本（**已存在，无需新写**）：

```
examples/z_image/model_quantize/Z-Image-Turbo-Quantize.py
```

它是参数化通用驱动：`--method` 的 choices 是 `sorted(QUANT_METHODS) + ["none"]`，
`--mode` 支持 `dynamic` / `dequant_once`，还有 `--no_quantize_text_encoder` / `--seed` / `--output`。
⇒ **`ck_int8` 一注册就自动出现在 choices 里**，Part B 也不需要新写动态量化脚本。

另有一组固定方法的脚本（它们已内置指标打印，是报告的数据源）：

```
examples/z_image/model_quantize/Z-Image-Turbo-{bnb_nf4,bnb_fp4,torchao_int8_w8a16,torchao_fp8_w8a16,torchao_int4_w4a16}.py
```

**Part A 要跑的矩阵**（改造前采基线、改造后重跑对比）：

| method | mode | 为何要跑 |
| --- | --- | --- |
| `none` | — | bf16 基线，给其他行做参系 |
| `torchao_int8_w8a16` | dynamic + dequant_once | **torchao 改造的主回归** |
| `torchao_fp8_w8a16` | dynamic | 同上，另一个 layout |
| `torchao_int4_w4a16` | dynamic | 同上（需 `mslk` 或改 packing format，不可用则记录跳过原因） |
| `bitsandbytes_nf4` | dynamic | bnb 应零改动，跑一次确认契约没误伤 |
| `ideogram4_fp8` | — | 该后端**不支持在线量化**（无 `create_quantized_linear`），走 §A5.1 的单层路径即可 |

**验收**：同 seed 下出图与基线一致（允许 bf16 舍入级差异）；加载耗时 / 显存 / 每步耗时同量级。

⚠️ **在线量化的显存陷阱（既有经验，不要踏）**：脚本头部那行
`os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")` **必须在 CUDA 初始化前设置**。
在线量化是逐层释放 fp 权重，默认分配器下被释放的 segment 会被交错的小量化张量钉住、
永不归还驱动（A100 上实测约 10 GB 泄漏）。新写任何在线量化脚本都要带上这行。

### A5.4 环境就绪情况

| 依赖 | 状态 |
| --- | --- |
| NF4 权重 | ✅ 已在 `models/DiffSynth-Studio/MiniMax-H3-NF4/`（fl2va 17.2 GB / TE 15.3 GB / VAE） |
| processor | 需 `models/MiniMax/MiniMax-H3/FL2VA/processor/`（bf16 官方仓已在本地） |
| 示例数据集 | ❌ `data/diffsynth_example_dataset/minimax_h3/` **不在本地**，脚本里的 `dataset_snapshot_download` 会拉（或预先下好） |
| Z-Image-Turbo | ✅ 已在 `models/Tongyi-MAI/Z-Image-Turbo/`（31 GB） |
| comfy-kitchen | 需 `pip install comfy-kitchen`（见 §B10） |

## A6. 不在 Part A 范围内的 core 改动

只剩一项，本次**不做**：

**`computation_module()` 的 `clone_to_device` 钩子** —— 给小显存卡的 `vram_limit`
细粒度逐层上卡用。comfy-kitchen 的 `QuantizedTensor` 无法 `copy.deepcopy`（见 §B8），
但本次靠示例脚本的 `onload_device="cuda"` 规避，不需要动 core。

---

# Part B：comfy-kitchen int8 后端

## B1. `int8_tensorwise` 与 convrot 的关系

**只有一个 format，convrot 是它的参数。** ComfyUI `QUANT_ALGOS` 里没有 `int8_convrot` 这个键。

```jsonc
{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}   // 旋转
{"format": "int8_tensorwise"}                                              // 不旋转
```

- 上游规则：`in_features % convrot_groupsize != 0` 的层会**降低 groupsize 或不旋转**
- ⇒ **同一个 checkpoint 里就是混的**，预量化加载必须**逐层读 marker**，不能用全局配置代替
- marker 里 convrot 字段也可能嵌在 `"params": {...}` 下（旧形态），两种都要吃
- **缺 `format` 字段的是已废弃的 `int8_w8a8` 格式 → 必须报错，不要猜**

### ⛑ 实测：目标 checkpoint 真的有两种 groupsize

`minimax_h3_fl2va_int8_convrot.safetensors` 的 250 个量化层：

```
x200  {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}
x50   {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 64}    <- blocks.*.adaln_proj.linear
```

原因：`adaln_proj.linear` 的 `in_features = 2688`，而 **2688 % 256 = 128 ≠ 0**，
上游降到 64（2688/64 = 42 ✓，且 64 = 4³ 满足 regular Hadamard 的 4 次幂要求）。

### ⛑ groupsize 填错的后果（已实测）

| 情况 | 结果 |
| --- | --- |
| gs=256 用在 in=2688 上（256 **不整除** 2688） | ✅ 抛 `ValueError`，大声报错 |
| gs=64 用在 gs=256 的权重上（**两者都整除** 5376） | ❌ **无任何报错**，`rel_l2 = 1.0010`、`cosine = 0.4999` |

第二种是真陷阱：不抛异常、不出 NaN，出图"看着像但明显更糟"。而目标 checkpoint 同时含这两类层。

### 存储契约

```
<layer>.weight        int8   [out_features, in_features]
<layer>.weight_scale  fp32   [out_features, 1]（per-channel）或 []（标量）
<layer>.comfy_quant   uint8  json.dumps(...) 的字节
<layer>.bias          原 dtype，不量化
```

## B2. comfy-kitchen API（已实测）

```python
import comfy_kitchen as ck
from comfy_kitchen.tensor import QuantizedTensor, get_layout_class

# 在线量化
qt = QuantizedTensor.from_float(fp_weight, "TensorWiseINT8Layout",
                               is_weight=True, per_channel=True,
                               convrot=True, convrot_groupsize=256)

# 从 checkpoint 张量重建
cls = get_layout_class("TensorWiseINT8Layout")
params = cls.Params(scale=weight_scale, orig_dtype=torch.bfloat16, orig_shape=(out, in_),
                    is_weight=True, convrot=True, convrot_groupsize=256)
qt = QuantizedTensor(qdata, "TensorWiseINT8Layout", params)

qt.state_dict(prefix="layer.weight")   # -> {"layer.weight": qdata, "layer.weight_scale": scale}
qt.dequantize()
get_layout_class(name).get_requirements()   # {min_sm_version, current_sm_version, fast_matmul_supported}
```

### ⭐ 前向不需要自定义实现

普通 `nn.Linear` 把 `weight` 换成 `Parameter(QuantizedTensor, requires_grad=False)`，直接调用就走量化 kernel
（`aten.linear.default` 已在 layout 里注册 dispatch）。

**一个类就能同时服务三种状态，且可混在同一次 forward 里**（实测，`nn.Linear(5376,4096)`）：

| 配置 | rel_l2 | 量化状态存在哪 |
| --- | --- | --- |
| convrot gs=256 | 0.01369 | `weight._params` |
| convrot gs=64 | 0.01313 | 同上 |
| 不旋转 | 0.01354 | 同上 |

⇒ **`ComfyKitchenLinear` 对 convrot 完全无感知，不要为它写分支，也不要把它存成 module 字段。**

### ⭐ dtype / device 迁移是安全的

| 操作 | 实测 |
| --- | --- |
| `.to(dtype=torch.bfloat16)` | storage 仍 int8、`params.scale` 仍 fp32 → **PRESERVED** |
| `.to("cpu")` → `.to("cuda")` | 权重与 scale 都正确搬迁，forward 数值不变 |

⇒ **不需要 `_apply` dtype 守卫**（与 bnb / Fp8Linear 那类自定义 buffer 不同）。

### registry 门禁（**踩坑点**）

```python
if torch.version.cuda is None or tuple(int(v) for v in torch.version.cuda.split(".")[:2]) < (13, 0):
    ck.registry.disable("cuda")     # 否则调用时报 "CUDA driver version is insufficient"
```

- **`ck.list_backends()` 不可信**，会把不可用的 cuda 后端报成 available
- per-call 覆盖：`with ck.use_backend("eager"): ...`（thread-local）
- int8 在 sm ≥ 7.5 上 `supports_fast_matmul()` 为 True，**正常路径不需要 override**；
  override 只用于兜底（sm < 7.5、或 `get_cuda_capability()` 为 None 的 CPU）
- 若某 layout 需要 override，**量化与 forward 都要包**（只包量化会在 forward 崩）

### 硬件门槛

`TensorWiseINT8Layout.MIN_SM_VERSION = (7, 5)`（Turing+）。A100(sm80) ✅ triton 快路径可用。
加速 1.19×（小 shape）～ 1.59×（`[28672,5376]` @ 8192 tokens）；数值 rel_l2 ≈ 0.013。
triton 只在 **Linux** 上随 torch 自带（`triton==3.6.0; platform_system == "Linux"`），Windows 落 eager。

## B3. 文件清单

| 文件 | 动作 |
| --- | --- |
| `diffsynth/core/quant/backends/comfy_kitchen.py` | **新增**，~220 行 |
| `diffsynth/core/quant/backends/__init__.py` | 改 1 行 |
| `diffsynth/configs/model_configs.py` | 注册 model_hash + quant_config |
| `examples/minimax_h3/model_inference/MiniMax-H3-FL2VA-CK-INT8.py` | 新增 |
| `pyproject.toml` | **可选**依赖组加 `comfy-kitchen` |

## B4. Linear 子类（遵守 Part A 契约）

```python
class ComfyKitchenLinear(torch.nn.Linear):
    """`nn.Linear` whose weight is a comfy-kitchen `QuantizedTensor`.

    Forward dispatch lives in the tensor subclass and reads the per-layer quant state off
    the weight, so this class only records the layout and pins the comfy-kitchen backend
    when the layout has no fast kernel on the current device.
    """

    def __init__(self, in_features, out_features, bias, *, layout, compute_dtype, force_eager):
        with torch.device("meta"):
            super().__init__(in_features, out_features, bias=bias, dtype=compute_dtype)
        self.layout = layout
        self.compute_dtype = compute_dtype
        self.force_eager = force_eager
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, x):
        if not self.force_eager:
            return super().forward(x)
        import comfy_kitchen as ck
        with ck.use_backend("eager"):
            return super().forward(x)

    def extra_repr(self):
        params = getattr(self.weight, "_params", None)      # meta shell has no _params yet
        if params is None:
            return f"{super().extra_repr()}, layout={self.layout}"
        return (f"{super().extra_repr()}, layout={self.layout}, "
                f"convrot={getattr(params, 'convrot', None)}, "
                f"groupsize={getattr(params, 'convrot_groupsize', None)}")
```

`force_eager = not get_layout_class(layout).supports_fast_matmul()`，建壳时算一次。

## B5. `QuantBackend` 各方法

| 方法 | 实现要点 |
| --- | --- |
| `validate_environment()` | `find_spec("comfy_kitchen")` 为 None → `ImportError` 带 `pip install comfy-kitchen` 提示；随后跑幂等初始化（禁 cuda 后端）。**不要在模块 import 时就 import comfy_kitchen**（它会初始化 CUDA 上下文并打日志） |
| `capabilities()` | `is_serializable=True`、`is_differentiable=False`、`is_compileable=False`（保守）、`requires_calibration=False` |
| `quantized_linear_classes()` | `(ComfyKitchenLinear,)` |
| `create_quantized_linear` | 搬到 `compute_device` → `QuantizedTensor.from_float(...)` → 装进 `ComfyKitchenLinear` → 搬到 `model_device` |
| `create_quantized_linear_shell` | meta 上建 `ComfyKitchenLinear`；**同时把 `{layer 名: (out, in)}` 记到 backend 实例缓存**，供 `unflatten_state_dict` 取 `orig_shape` |
| `unflatten_state_dict` | 见 §B6 |
| `flatten_state_dict` | `qt.state_dict(prefix)` + 按 §B1 规则重建 marker 字节（P1，可后置） |
| `dequantize_to_linear` | `module.weight.dequantize().to(compute_dtype)` → 普通 `nn.Linear` |
| `checkpoint_key_patterns` | `("weight", "weight_scale", "comfy_quant", "bias")` —— 全是**兄弟键**，不能用默认值 |

## B6. `unflatten_state_dict`：预量化加载的核心

```
输入:  <layer>.weight / <layer>.weight_scale / <layer>.comfy_quant  (+ <layer>.bias)
输出:  <layer>.weight = QuantizedTensor(qdata, "TensorWiseINT8Layout", Params(...))
       marker 与 scale 键全部 pop（已被吸收）
```

1. 扫一遍 state_dict，收集 `.comfy_quant` 结尾的键 → 量化层名集合
2. 逐层解析 marker：`json.loads(bytes(tensor.tolist()))`
   - `format` **必需**；≠ `"int8_tensorwise"` → 报错（列出实际值与支持列表）
   - `convrot`：`conf.get("convrot", conf.get("params", {}).get("convrot", False))`
   - `convrot_groupsize`：同样两级取，默认 256
   - `full_precision_matrix_mult`（可选 bool）→ 该层设 `force_eager=True`
3. `weight_scale`：`numel() > 1` → per-channel；`== 1` → 标量
4. `orig_shape`：**不能从 `qdata.shape` 推**，从 §B5 建壳时的缓存取
   （框架调用顺序保证 `prepare_for_prequantized_load` 先于 `unflatten_state_dict`）
5. 构造 `Params(...)` → `QuantizedTensor(...)` → 写回 `<layer>.weight`
6. pop `<layer>.comfy_quant` / `<layer>.weight_scale`；`<layer>.bias` 原样留下
7. 本 format 不用的边料键（如 `input_scale`）：pop 并 warn once

**必须自己加的校验**（上游不会报）：若 `convrot=True`，断言
`in_features % convrot_groupsize == 0` **且** `convrot_groupsize` 是 4 的幂。
不整除时上游会抛 `ValueError`，但**整除但数值错的情况上游完全不测**（见 §B1）。

## B7. 注册

### B7.1 量化方法：只注册一个

```python
def _int8_config(kwargs):
    return {"layout": "TensorWiseINT8Layout",
            "is_weight": True, "per_channel": True,
            "convrot": True, "convrot_groupsize": 256,
            **kwargs}

register_quant_method(
    "ck_int8", "comfy_kitchen", _int8_config,
    label="8bit, int8 W8A8 (ComfyUI int8_tensorwise; convrot on by default, "
          "per-layer value comes from the checkpoint marker when loading prequantized)")
```

`backend_config_kwargs` 可覆盖 `convrot` / `convrot_groupsize` / `per_channel`。

### B7.2 model_hash（已实测，直接用）

```python
{
    # ModelConfig(model_id="Comfy-Org/MiniMax-H3",
    #             origin_file_pattern="diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors")
    "model_hash": "fb6c399412920c6b817d5c1f43ec62cd",
    "model_name": "minimax_h3_dit",
    "model_class": "diffsynth.models.minimax_h3_dit.MiniMaxH3DiT",
    "quant_config": {"method": "ck_int8", "load_prequantized": True,
                     "target_modules": MINIMAX_H3_CK_INT8_TARGETS},
},
```

同目录 `minimax_h3_fl2va_bf16.safetensors`（对照基线）若 hash 与已注册的官方分片版不同，
也补一条（同 `model_name`，无 `quant_config`）。

### B7.3 `target_modules`（已实测，**含 adaln_proj.linear**）

`_name_matches` 只支持「全名相等」或「以 `.pattern` 结尾」，**不支持前缀排除**。
写后缀会把 `token_refiner.blocks.*` 的 8 个未量化层套进去 → 加载当场挂。

实测该 checkpoint 的量化层是 **250 个 = 50 blocks × 5**：

```python
MINIMAX_H3_CK_INT8_TARGETS = [
    f"blocks.{i}.{name}" for i in range(50)
    for name in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2", "adaln_proj.linear")
]
```

⚠️ 这份 checkpoint **把 `adaln_proj.linear` 也量化了**（另一份常见的第三方发布则刻意不量化 adaLN）。
不量化的是：`condition_proj`、`final_layer.*`、`video/audio_patch_proj`、`time_embedder.*`、`token_refiner.*`。
**换 checkpoint 时先跑 `probe_ck_checkpoint.py` 重新确认，不要硬编码。**

## B8. ⚠️ VRAM 管理：`copy.deepcopy` 不可用（已实测）

`QuantizedTensor` 用 `_make_wrapper_subclass` 创建，wrapper 本身**没有真实 storage**（数据在 `_qdata`），
标准 `copy.deepcopy` 报 `RuntimeError: Attempted to call copy_() on an invalid python storage.`

而 `AutoWrappedQuantizedModule.computation_module()`（`layers.py:533-540`）有这条分支：

```python
device = self.preparing_device if self.state == 2 else self.onload_device
if device == self.computation_device:
    return self.module                                    # 早返回，不 deepcopy
return copy.deepcopy(self.module).to(device=self.computation_device)   # <- 崩在这里
```

实测：

| vram_config | 显存充裕（state 2） | 显存吃紧（`vram_limit` 阻止 preparing，state 1） | offload 后权重位置 |
| --- | --- | --- | --- |
| `onload=cpu, preparing=cuda`（NF4 示例那套） | ✅ rel_l2 0.0131 | ❌ **崩** | cpu ✓ |
| **`onload=cuda, preparing=cuda`（本计划采用）** | ✅ | ✅ | cpu ✓ |

❌ 那一格是**条件性崩溃**：`forward()` 里 `preparing()` 只在 `vram_limit is None or check_free_vram()`
时才调，NF4 示例传了 `vram_limit=...` ⇒ **小图能跑、大图崩**，极难定位。

**对策（零 core 改动）**：示例脚本用 `onload_device = computation_device = "cuda"`
—— state 1 时命中早返回，永不进 deepcopy 分支；`offload_device="cpu"` 保留 offload 语义（实测确认）。
代价是丢掉 `vram_limit` 的细粒度逐层上卡（那是给"单模型都装不下显存"的极端场景用的）。
对本任务无影响：int8 DiT 是 31.7 GiB，A100 80GB 能整个装下。

**要支持小显存卡**则需 core 新增可覆盖的克隆钩子（`clone_to_device`），comfy-kitchen 用
`QuantizedTensor.__tensor_flatten__` / `__tensor_unflatten__`（或 `_copy_with`）重建。**本次不做。**

对照：torchao 的 tensor subclass **支持 deepcopy**（state 0 / 1 都正常，rel_l2 0.01258），
所以这是 comfy-kitchen wrapper subclass 特有的限制，不是框架缺陷。

## B9. 示例脚本

以 `examples/minimax_h3/model_inference/MiniMax-H3-NF4-FL2VA.py` 为模板，**vram_config 必须改**：

```python
# ❗ onload_device 必须等于 computation_device，否则显存吃紧时会撞上
#    AutoWrappedQuantizedModule 的 copy.deepcopy 分支而崩（见 §B8）
vram_config = {
    "offload_dtype": torch.bfloat16, "offload_device": "cpu",
    "onload_dtype": torch.bfloat16, "onload_device": "cuda",     # <- 不是 "cpu"
    "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16, "computation_device": "cuda",
}

ModelConfig(path="models/Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors",
            **vram_config),
```

TE / VAE 用官方 bf16（`models/MiniMax/MiniMax-H3/FL2VA/{text_encoder,video_vae,audio_vae}`）。
**不要**用 `offload_device="disk"`（见 §A6）。

## B10. 环境准备

```bash
PY=/root/miniconda3/envs/debug/bin/python
$PY -m pip install comfy-kitchen        # 本次验证用 0.2.26（有 cp310 wheel）
```

`pyproject.toml` 放**可选**依赖组（如 `quant` extra），**不要设为必需** ——
它带 135 MB（cuda）+ 36 MB（hip）的预编译 `.so`，会拉低整仓库安装成功率，且 Windows / NPU 上无意义。

⚠️ 装完先回归：`python -c "import diffsynth"` + 跑一个现有 bnb / torchao 量化示例。

---

# 实施步骤与验收

> 每步都有可执行验收，**未通过不要进下一步**。
> 脚本与产物全部放 `workspace/20260808_comfy_kitchen_backend/`（本任务目录，复用，不要每跑一次新建）。
> **任何 GPU 实验前先 `nvidia-smi` 检查目标卡占用**；被占用先问用户，不要擅自换卡。

## Step A1 — 采基线（Part A 改造前，必做）

两部分，**改造前不采基线，改造后就无法证明没退化**：

1. **单层**：对 torchao 与 ideogram4_fp8，把「在线量化 / 预量化加载 / `dequant_once` / CPU offload」
   四条路径的输出张量存到 `baseline/`
2. **端到端**：跑一次 `examples/minimax_h3/model_inference_low_vram/MiniMax-H3-NF4-FL2VA.py`
   （先用小 `num_inference_steps` 跑通），存下产物、日志、峰值显存作基线。
   这是 **NF4 disk offload** 的基线，也是本次 core 改动的主防线（见 §A5.2）。

## Step A2 — 契约 + 自检

实现 §A2 的契约与 §A3 的 `check_backend_contract`。此时 3 个后端应该有 2 个不过（torchao、ideogram4_fp8）
—— 自检能红说明它有效。

## Step A3 — 改造三个后端

按 §A4 顺序做：bitsandbytes（仅确认）→ torchao → ideogram4_fp8。
每个改完立即跑 `check_backend_contract` + §A5.1 的四条路径与基线比对。

**验收**：3 个后端契约全绿；四条路径与基线差异 < 1e-6 或逐元素相同；`state_dict` 键集合不变。

## Step A4 — ❗ NF4 disk offload 回归（core 改动的关键验收）

同 seed 重跑 `examples/minimax_h3/model_inference_low_vram/MiniMax-H3-NF4-FL2VA.py`，
与 Step A1 的端到端基线比对。验收条目见 §A5.2。

**这一步不过不得进 Part B** —— 它是判断 `checkpoint_keys` / `build_quantized_shell`
两个新接口没有弄坏现有 disk offload 的唯一真实证据。

## Step B1 — 摸清 checkpoint（无需 GPU）

跑已有的 `probe_ck_checkpoint.py`。**验收**：
- 所有 marker 的 `format` 都是 `int8_tensorwise`
- 与 `MiniMaxH3DiT` 的**键差集为空**（已验证：535 vs 535）
- 若出现 `adaln_t_table` / `adaln_proj.linear` 第二维不是 2688 / 无 `time_embedder`
  → 是 `pruned` 结构，**立即停下告知用户**，本计划不覆盖

## Step B2 — 后端骨架 + 单层单测（GPU）

`test_ck_unit.py` 验收项：

1. 用**真实 checkpoint 的一层**（`blocks.0.mlp.fc1`，gs=256）重建 `QuantizedTensor`，
   forward 与 `qt.dequantize()` 后 `F.linear` 的相对误差 < 0.02
2. **再用 `blocks.0.adaln_proj.linear`（gs=64）重复一次** —— 这一条专门锁住"逐层读 marker"
3. 反量化后的权重与 bf16 文件同层权重比：余弦 > 0.999、rel_l2 < 0.02
4. `.to(dtype=bfloat16)` 后 storage 仍 int8、scale 仍 fp32
5. `.to("cpu")` → `.to("cuda")` 往返后 forward 数值不变
6. 在线量化往返：bf16 → `from_float` → `dequantize`，rel_l2 ≈ 0.01
7. 畸形 marker（缺 `format`、未知 `format`、groupsize 不整除）**必须抛异常**
8. `check_backend_contract` 通过
9. 未安装 comfy-kitchen 时 `import diffsynth` 正常、用 `ck_int8` 报清晰安装提示

## Step B3 — 整模型加载（GPU）

**验收**：250 个目标层是 `ComfyKitchenLinear`，其余仍是 `AutoWrappedLinear` / `nn.Linear`；
无 unexpected / missing keys；打印实测 CPU 内存与显存占用并与 bf16 版对比。

## Step B4 — 端到端出图（GPU）

先 8 步小样通链路，再 50 步正式产物。与 bf16 基线**同 seed** 对照，记录耗时、峰值显存、主观画质。

## Step B5 — `mode="dequant_once"` 回归

同一 checkpoint 用 `dequant_once` 加载，确认还原成普通 bf16 `nn.Linear` 且出图正常。

## Step B6 — ❗ 动态（在线）量化，用 Z-Image-Turbo

到目前为止验的都是**预量化加载**；在线量化走的是 `create_quantized_linear` →
`QuantizedTensor.from_float`，**是另一条代码路径**，必须单独验。

不需要写新脚本 —— `ck_int8` 注册后自动出现在通用驱动的 choices 里：

```bash
PY=/root/miniconda3/envs/debug/bin/python
S=examples/z_image/model_quantize/Z-Image-Turbo-Quantize.py

CUDA_VISIBLE_DEVICES=<空闲卡> $PY $S --method none                              # bf16 基线
CUDA_VISIBLE_DEVICES=<空闲卡> $PY $S --method ck_int8 --mode dynamic            # 主目标
CUDA_VISIBLE_DEVICES=<空闲卡> $PY $S --method ck_int8 --mode dequant_once       # 还原路径
CUDA_VISIBLE_DEVICES=<空闲卡> $PY $S --method ck_int8 --no_quantize_text_encoder # 只量化 DiT
```

**验收**：

1. 四条命令都能跑完并出图，图像质量与 `none` 基线主观可比（同 seed）
2. `dynamic` 的 model-resident VRAM 明显低于 `none`（量化层体积减半）
3. `dequant_once` 的显存应回到 bf16 量级（证明还原真的发生了）
4. 与 `torchao_int8_w8a16 --mode dynamic` 同 seed 对比，作为同类 int8 方法的横向参系
5. 日志里量化层数量符合预期（DiT + TE）

⚠️ 在线量化时 `convrot` 由 `backend_config_kwargs` 决定（默认 `True, gs=256`）。
Z-Image 的层尺寸不一定都被 256 整除 —— **遇到不整除的层必须自动降低 groupsize 或关旋转，
不能直接抛错**（这是上游 `in_features % 256 == 0` 规则的实现作业）。
实现时要在日志里报出每种 groupsize 各多少层，便于核对。

---

# 实验报告（交付物，必做）

完成后写 `workspace/20260808_comfy_kitchen_backend/REPORT.md`。**不是口头汇报，是带数据的文档。**

## 目录约定

```
workspace/20260808_comfy_kitchen_backend/
  REPORT.md                  ← 主报告
  logs/<script>_<时间戳>.log  ← 每个验证脚本的**完整 stdout+stderr**
  outputs/                   ← 出图 / 视频 / 音频产物
  baseline/                  ← Step A1 采的改造前基线
```

跑脚本一律用 `2>&1 | tee logs/<name>_$(date +%Y%m%d_%H%M%S).log`，**不要只贴摘取的几行** ——
后来排查靠的就是完整日志。

按项目规范：**跑量化回归前先清仓库根目录的旧出图**（`image_*.jpg` 等），
跑完把产物归档到 `outputs/`，不要留在根目录。

## REPORT.md 必须包含

1. **环境**：解释器路径、torch / triton / comfy-kitchen 版本、GPU 型号与 capability、driver
2. **每一项验收的结果表**，一行一项，列：验收项 / 命令 / 日志文件名 / 结果（✅/❌）/ 关键数字
3. **契约自检输出**：`check_backend_contract` 对 4 个后端的逐项结果（照抄日志）
4. **Part A 回归对比表**：每个后端 × 每条路径，改造前 vs 改造后的数值差异
5. **NF4 disk offload 对比**：产物一致性、峰值显存、日志里的量化层数量
6. **动态量化对比表**（Z-Image-Turbo），指标直接抄脚本已经打印的那几项：

   | method | mode | load 耗时(含量化) | model-resident VRAM (allocator / device real) | denoise 每步 | it/s | 出图 |
   | --- | --- | --- | --- | --- | --- | --- |

7. **MiniMax-H3 端到端**：bf16 基线 vs ck_int8，同 seed，含耗时 / 峰值显存 / 产物路径
8. **未通过与跳过的项**：必须列出，带原因与日志位置。
   **不得省略失败项，也不得把"没跑"写成"通过"。**
9. **发现的新问题**（如果有）：现象 / 复现命令 / 影响面 / 建议

## 报告中的数字要求

- 所有计时必须在 **GPU 空闲时**采集（跑前 `nvidia-smi` 确认，并在报告里记录当时卡状态）。
  被占用时采的数字**作废重测**，不要写进报告
- 数值对比给具体量（rel_l2 / 余弦 / max|diff|），不写"差不多""看不出区别"
- 推断与实测要分开标注：没实测过的写"未验证"，不要写成结论

---

# 交付前自检清单

- [ ] `check_backend_contract` 对 4 个后端（bnb / torchao / ideogram4_fp8 / comfy_kitchen）全绿
- [ ] Part A 的四条路径 × 2 个改造后端，与基线比对通过
- [ ] **NF4 disk offload 端到端回归通过**（`model_inference_low_vram/MiniMax-H3-NF4-FL2VA.py`，同 seed 与基线一致）
- [ ] **动态量化回归通过**（Z-Image-Turbo，Part A 的方法矩阵 + Part B 的 `ck_int8`）
- [ ] **实验报告 `REPORT.md` 已完成**，每个验证脚本的完整日志在 `logs/`、产物在 `outputs/`
- [ ] 仓库根目录没有残留的 `image_*.jpg` 等产物
- [ ] `checkpoint_key_patterns` 对每个后端都能覆盖其存盘键（自检第 7 项）
- [ ] 所有后端的 `state_dict` 键集合与改造前完全一致（磁盘格式零破坏）
- [ ] `is_quantized_linear` 在整个 `core/quant` 里只有基类一处实现
- [ ] 未安装 comfy-kitchen 时 `import diffsynth` 正常
- [ ] `describe_quant_method("ck_int8")` 打印出后端与默认 kwargs
- [ ] 现有 bnb / torchao / ideogram4 的量化示例都能跑
- [ ] 逐层 marker 解析已验证（可临时篡改某层 marker 的 groupsize 观察行为差异）
- [ ] 在线量化与预量化加载两个方向都验证过
- [ ] `enable_vram_management_recursively` 里已加注释说明 quantize 检查必须在 module_map 循环之前
- [ ] 所有脚本与产物在 `workspace/20260808_comfy_kitchen_backend/` 下
- [ ] 注释只加在对外暴露接口上，不写零碎行内注释（项目规范）

---

# 未来扩展与已知限制

## 其他 format（架构留口，本次不做）

`unflatten_state_dict` 按 `marker["format"]` 分派到 layout，加新 format 只需补一条映射 + 一条注册。
ComfyUI `QUANT_ALGOS` 全表与硬件门槛：

| format | layout | MIN_SM | A100 (8,0) | 备注 |
| --- | --- | --- | --- | --- |
| `int8_tensorwise` | `TensorWiseINT8Layout` | (7,5) | ✅ | **本次实现** |
| `convrot_w4a4` | `TensorCoreConvRotW4A4Layout` | (7,5) | ✅ | 实测 rel_l2 0.227，精度不可用于生产 |
| `float8_e4m3fn` / `e5m2` | `TensorCoreFP8Layout` | **(8,9)** | ⚠️ | **triton 在 sm80 编译崩**（`fp8e4nv not supported`），必须全程 `use_backend("eager")` 才可用（实测 rel_l2 0.027）。边料含标量 `weight_scale` + 可选 `input_scale` |
| `nvfp4` | `TensorCoreNVFP4Layout` | (10,0) | ❌ | block scale 本身是 fp8_e4m3，sm80 同样崩；eager 能否救**未验证** |
| `mxfp8` | `TensorCoreMXFP8Layout` | (10,0) | ❌ | `weight_scale` 需按 `float8_e8m0fnu` view |

## 已知限制

| 项 | 状态 |
| --- | --- |
| **comfy-kitchen 的 disk offload** | 接口层已就绪（Part A 的 `checkpoint_key_patterns`），但**未验证**。需注意：`onload_device="cpu"` + disk 会走到 deepcopy 分支而崩（§B8）；理论上 `onload_device="disk"` 会走 `_load_from_disk` 分支而避开 deepcopy，**这个组合需实测确认** |
| **小显存卡的 `vram_limit` 细粒度逐层上卡** | 待做，需 core 新增 `clone_to_device`（§A6 / §B8）。本次靠 `onload_device="cuda"` 规避 |
| 训练 / QLoRA | 不支持（`QuantizedTensor` 前向不可微）。推理期 LoRA hot-load 已实测可用 |
| ROCm / Windows / NPU | 无硬件，未验证（Windows 无 triton，会落 eager） |
| 自产 checkpoint 的质量 | 格式只规定"存一个 scale"，不规定怎么算。实测某第三方发布的 scale 比朴素 `absmax/127` 小 3–6%（做了裁剪搜索），用朴素配方产出的质量会略逊 |

---

# 附录：已验证事实速查

| 事实 | 值 |
| --- | --- |
| 验证环境 | A100-80GB (sm80) / torch 2.10.0+cu128 / triton 3.6.0 / driver 570.133.20 / py3.10 |
| comfy-kitchen 版本 | 0.2.26 |
| 目标 checkpoint | 31.70 GiB、1035 tensors、**stock 非 pruned**、与 `MiniMaxH3DiT` 键完全匹配（535 vs 535）、hash `fb6c399412920c6b817d5c1f43ec62cd` |
| 量化层 | 250 个 = 50 blocks × {qkv_proj, out_proj, fc1, fc2, adaln_proj.linear} |
| marker 分布 | 200× gs=256，50× gs=64（`adaln_proj.linear`，因 2688 % 256 ≠ 0） |
| `weight_scale` shape | 全部 per-channel `[out, 1]` |
| 必须禁 cuda 后端的条件 | `torch.version.cuda < 13` |
| `ck.list_backends()` 可信度 | **不可信**，会把不可用的 cuda 报成 available |
| int8 加速 | 1.19×（小 shape）～ 1.59×（`[28672,5376]` @ 8192 tokens） |
| int8 数值误差 | rel_l2 ≈ 0.013（convrot 关闭 ≈ 0.011） |
| `.to(dtype)` 是否破坏量化 | comfy-kitchen **不破坏**（无需守卫）；ideogram4 `Fp8Linear` 需要 `_apply` 守卫 |
| 是否需要自定义 forward | comfy-kitchen **不需要**（tensor subclass 自己 dispatch） |
| `copy.deepcopy(QuantizedTensor)` | ❌ 崩（wrapper 无真实 storage）；torchao 的 subclass ✅ 可以 |
| torchao 加 marker 类的影响 | 三条路线 rel_l2 全为 0.00898，`state_dict` 键不变 |
| `Fp8Linear` 改 `nn.Linear` 子类的影响 | forward / state_dict 键 / dtype 守卫全部不变；bias 由 buffer 变 Parameter（需显式 `requires_grad_(False)`） |

复现命令：

```bash
PY=/root/miniconda3/envs/debug/bin/python
cd /mnt/nas1/zhanghong/project26/main_project/DiffSynth-Studio
W=workspace/20260808_comfy_kitchen_backend

$PY $W/probe_ck_checkpoint.py                                        # checkpoint 结构（无需 GPU）
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/tmp/ck310_x $PY $W/probe_ck_backend_feasibility.py   # comfy-kitchen API
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/tmp/ck310_x $PY $W/probe_mixed_convrot.py            # 混合 groupsize
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/tmp/ck310_x $PY $W/probe_vram_compat.py               # VRAM 管理兼容性
CUDA_VISIBLE_DEVICES=1 $PY $W/probe_torchao_detection.py                                 # torchao 判据
CUDA_VISIBLE_DEVICES=1 $PY $W/probe_fp8linear_contract.py                                # Fp8Linear 契约改造
```

权威来源：
- `comfy_kitchen/tensor/{base,int8,int8_utils}.py`（layout / Params / 旋转实现）
- `comfy_kitchen/registry.py`（`use_backend`、优先级、thread-local override）
- ComfyUI `comfy/quant_ops.py`（`QUANT_ALGOS` 表、cu13 门禁）
- ComfyUI `comfy/ops.py`（`_load_quantized_weight` / `_quantized_weight_state_dict`：marker 的权威读写实现）
