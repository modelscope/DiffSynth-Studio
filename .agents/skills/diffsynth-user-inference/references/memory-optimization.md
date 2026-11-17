# 低显存推理参考

仅在普通 example 显存不足、需要低显存配置或定位 OOM 时阅读。默认行为、精度许可和选档公式以 SKILL.md 为准。先识别预量化输入与统一内存。浮点底模先原精度全 GPU，独立显存不足再仅 Text Encoder 原精度 CPU Offload；适用方案均不足才确认未知量化许可。路线 A 对 DiT 与 Text Encoder 同步配置：INT8 → INT8+EntroPack 7/6 bpp → 直接 EntroPack 5/4 bpp，优先常驻；4 bpp 仅容量不足时再 4 bpp + CPU。预量化输入保持原表示，跳过新增量化；统一内存跳过所有 CPU Offload。这里说明框架具体机制与配置；片段需嵌入匹配的官方 example，不能直接替代完整脚本。

目录：资源/统一内存 → 四阶段机制 → 量化方法 → 双组件量化选档 → CPU Offload → Disk Offload → 预量化输入与准备 → 解码/运行优化 → 源码索引。

## 1. 资源估算：区分文件、常驻权重和运行峰值

- **原生精度**指普通 example/用户指定 checkpoint 的现有状态，不强制全模型 BF16；FP32 VAE、FP16 组件、已注册预量化方法都保留。
- 首先完整识别所有组件：DiT/UNet、一个或多个 text encoder、图像编码器、视频/音频 VAE、控制模型、LoRA 等。tokenizer/processor 目录不可省略，但不应按 GPU 权重组件套 Offload。
- 用实际参数与 dtype 计算权重。BF16/FP16 每元素 2 字节，FP32 为 4；十进制 B 参数与 GiB 不同。文件体积仅是起点：FP32 文件可按 BF16 加载，packed 文件的元素形状不是原始参数量，一个文件也可能包含多个组件。
- `probe_environment.py --weight-file ...` 只读 safetensors 头；重复传入实际分片，逐组件归组。`numel_by_dtype` 是存储元素统计，`floating_2d_weight_numel` 包含 embedding，只能用于寻找候选矩阵。结合模型代码、converter、`target_modules` / `exclude_modules` 识别真正可量化的 Linear.weight，偏置与其余参数单独计数。路线 A 分别统计全部 DiT 和 Text Encoder 的 Nq、U，不将压缩比例作用于 VAE 或整个 Pipeline，也不减掉 Text Encoder 常驻开销。
- 已发布预量化文件优先使用真实 packed 数据大小、完整量化状态与格式说明；侧车 JSON、分片索引、metadata 不能代替权重，也不能遗漏附加张量。
- 常驻可行还需容纳对应任务的激活、Attention 工作区、latent/条件缓存、临时反量化权重、搬运副本、输出缓存、VAE 峰值和分配器余量。同模型不同分辨率、帧数、batch、文本长度、控制输入可能有很大差异；参数量不能推出这些峰值。
- `reserve_gib` 应覆盖这些非持久权重开销。0.5 GiB、MiniMax-H3 的 2 GiB 等官方余量只是最低参考；有同规格测量时以测量和安全余量为依据，无测量时说明不确定性。不要把开销同时计入余量又重复从 B 扣除。
- CPU 同样区分常驻、准备和暂存副本：浮点源组件、逐层转换前后对象、state_dict、预处理/输出、序列化和文件页缓存均可能占 RAM。使用探测报告的有效内存/cgroup 状态，不把主机总内存当进程额度。
- 磁盘检查原始权重、已有 packed 文件、临时文件和输出空间；同文件系统不能把两个 free 值相加。Disk Offload 的吞吐也受本地 SSD/NAS、I/O、OS 页缓存影响，不是零 RAM/零 VRAM。

### 新任务预算和 vram_limit

```python
free_bytes, total_bytes = torch.cuda.mem_get_info(compute_device)
new_task_budget_gib = free_bytes / 1024**3 - reserve_gib
vram_limit = total_bytes / 1024**3 - reserve_gib
```

前者用于判断本任务可新增占用，后者是框架比较**整卡已用量**的阈值。`AutoTorchModule.check_free_vram()` 实际比较 `(total - free) / 1024**3 < vram_limit`，其中包含其他进程。把 `free - reserve` 直接传成 `vram_limit` 会重复扣除已有占用。

阈值检查发生在进入 preparing 之前，不预先扣除下一层大小；当前层计算仍可能超限，也不主动限制所有激活分配。`None` 不限制 preparing；`0` 尽量不缓存 preparing 权重，当前计算层仍必须调入，不能解释成零显存。普通模型只传 `vram_limit` 不会自动启用 Offload。

OOM 后退出失败进程/释放引用，重新探测空闲显存后计算；不要用尚未释放的失败模型占用估计一次全新启动的配置。多卡按真实并行方案逐卡估算，不直接合并容量。

### 统一内存：先确认物理拓扑，再分支

Apple Silicon 的 MPS、明确报告集成共享内存的 CUDA/ROCm 设备，按共享物理存储评估；不能只因 API 支持 unified addressing、managed memory 或系统是 ARM 就认定统一内存。MPS 在其他硬件上也可能可用。探测器给出 `memory_architecture.kind` 与 evidence；`unknown` 需要结合目标设备规格/用户已知事实核实，不将 unknown 冒充 discrete。

统一内存仅计一次：系统/cgroup 有效余量、设备工作集/分配器限制和准备/激活峰值共同约束同一块存储。MPS `recommended_max_memory()` 是推荐工作集，不是独占可用量；还需结合 `driver_allocated_memory()` 等实际设备统计和系统余量，不能把两个数相加，或把剩余预算重复减权重。未取得统计时明确未知，不套 CUDA `mem_get_info`。

这类设备跳过原精度仅 Text Encoder CPU、原精度 CPU、量化 CPU 和预量化模型 CPU 分支；移到 CPU 不视为容量回收，也可能新增副本。后端不支持 MPS/目标设备时，CPU Offload 不会让 CUDA kernel 变得兼容。仍可检查真正按层惰性读取的 Disk，但必须确认目标设备路径、文件/转换器、I/O 和最小工作集；磁盘不能消除当前层和激活开销。

## 2. 四阶段显存管理与启用条件

| 状态 | 含义 | 典型控制方 |
| --- | --- | --- |
| offload | 暂不调用该组件时的存储 dtype/device | Pipeline 切换组件 |
| onload | 组件即将/正在被使用时的基础存储状态 | Pipeline 切换组件 |
| preparing | 预算允许时保留在计算设备上的缓存状态 | 包装层 forward |
| computation | 实际 forward 使用的临时权重状态 | 包装层 forward |

四组参数分别为 `offload_dtype/device`、`onload_dtype/device`、`preparing_dtype/device`、`computation_dtype/device`。`computation_device="cuda"` 才是 GPU 计算；将存储放 CPU 不意味着改为 CPU 计算。

实际加载链：`Pipeline.from_pretrained` → `BasePipeline.download_and_load_models` → `ModelPool` → `core/loader/model.py`。`ModelPool.need_to_enable_vram_management` 要求 offload_dtype 和 offload_device 都非空；按 example 显式填写需要的阶段，避免混用默认值。

- 查 `configs/vram_management_module_maps.py` 与 `VERSION_CHECKER_MAPS` 确认粒度。映射存在时通常包装 Linear、Norm 等；量化 Linear 使用 `AutoWrappedQuantizedModule`，其他层按原映射处理。
- 没有细粒度映射时，回退成整个模型的 `AutoWrappedModule`，`fill_vram_config` 将 onload/preparing 改成 computation 状态。此时不能靠较小 `vram_limit` 让单个大组件逐层流式运行。
- 某些根参数/缓冲会直接 materialize，不能假定所有字节都被按层惰性加载。
- `BasePipeline.load_models_to_device` 只卸载带 `vram_management_enabled` 标记的模型。只包装 text encoder 时，未包装的 DiT/VAE 保持 GPU；检查 Pipeline unit 的真实 `onload_model_names` 和调用顺序。
- preparing 在缓存允许时保留一部分层，超预算部分每次计算临时调入；降低阈值通常增加传输开销，不能保证释放其他组件或改善解码激活 OOM。

## 3. QuantizeConfig、存储 FP8 与后端选择

### 两套机制不能混为一谈

**显存管理 FP8**：通过 `offload_dtype`、`onload_dtype`、`preparing_dtype` 将受管理的普通参数存成 FP8，计算时临时转回组件要求的 BF16/FP16。是有损存储转换，不因 computation 是 BF16 而变成无损，也不等于 FP8 Tensor Core 计算。是否覆盖某类层由包装映射决定。

**QuantizeConfig**：匹配普通 `nn.Linear`，量化其权重，保留后端专用模块/张量。norm、embedding、conv、bias 和未匹配/排除的层通常不受此压缩。已量化层的 packed 存储由后端决定，不会因为四阶段 dtype 写成 BF16 就还原为 BF16。路线 A 为全部 DiT/Text Encoder 分别设置 QuantizeConfig，优先 GPU 常驻；4 bpp 容量不足且设备有独立显存时，才对该表示叠加 CPU Offload。不能用四阶段 FP8 存储替代当前默认 INT8。

```python
from diffsynth.core.quant import QuantizeConfig
quantize = QuantizeConfig(
    method="torchao_int8_w8a16",
    mode="dynamic",
    exclude_modules=sensitive_layers,  # 来自目标模型的真实层名
)
```

- `mode="dynamic"` 保留后端量化表示，按该后端 kernel/解压路径执行；不代表所有方法都显式还原全层浮点矩阵。
- `mode="dequant_once"` 一次还原普通浮点 Linear，仍有量化误差，且不再有 packed 常驻节省；不能用于满足 4-bit 常驻预算。
- 模块名匹配为完整点分名称相等，或以 `"." + pattern` 结尾；不是正则/任意子串。`target_modules` 指定范围，`exclude_modules` 排除敏感层；不使用其他模型的层名冒充已命中。
- `MixedQuantizeConfig(configs=[...])` 的匹配层集合须互斥，mode 相同。预量化混合配置的 `load_prequantized=True` 在顶层，子项不设置 True。
- `describe_quant_method(method)` 可查后端参数；构建配置可能导入并检查对应依赖，不是纯静态无依赖操作。仅安装/测试实际选中的后端。

### 非默认方法与历史质量记录

下表记录框架能力，不是自动选档顺序。默认路线不插入 FP8、NF4、通用 INT4、NF4+INT8；“4 bpp”默认指直接 EntroPack，不等同 torchao INT4/NF4。用户固定相应 checkpoint/方法或明确要求实验时按该要求处理。

原版 Qwen-Image 的本地 1024×1024、40 步、单 prompt 对照中，普通 NF4 有严重颗粒；有结构依据的 NF4+INT8 敏感层混合曾改善画面。该历史结论不意味着所有模型都应套混合配置，也不改变当前默认路线。用户明确采用混合方法时，MixedQuantizeConfig 子项覆盖集合必须互斥，原始浮点权重重载后再转换；不能通过重解释已打包 NF4 或转 BF16 假装恢复精度。

INT8、二次量化与直接 EntroPack 都要核对质量：可解码、量化命中和显存达标不等于质量合格。没有合格方法时从浮点源评估原精度卸载，统一内存跳过 CPU；不能用 CPU 卸载修复量化噪声。4 bpp 只因常驻容量不足才进入 4 bpp + CPU。历史不同量化方法、组件驻留或解码规格的峰值不能混用。

### 当前源码注册的方法

| 类别 | method | 使用要点 |
| --- | --- | --- |
| 8-bit weight-only | `torchao_int8_w8a16`, `torchao_fp8_w8a16` | 无模型专属指导时先检查兼容 W8A16，激活保留计算精度 |
| 4-bit weight-only | `bitsandbytes_nf4`, `bitsandbytes_fp4`, `torchao_int4_w4a16`, `torchao_nvfp4_w4a16` | NF4/FP4/INT4 不是同一种文件格式；保留敏感层 |
| torchao 激活量化 | `torchao_int8_w8a8`, `torchao_fp8_w8a8`, `torchao_int4_w4a8`, `torchao_mxfp8_w8a8`, `torchao_mxfp4_w4a4`, `torchao_nvfp4_w4a4` | 同时改变计算路径；不能只按存储位宽评估质量/兼容性 |
| kitchen | `comfy_kitchen_int8_w8a8`, `comfy_kitchen_fp8_w8a8` | 保留格式 marker、scale、convrot 等信息；格式需匹配 |
| DeepGEMM | `deep_gemm_fp8_w8a8`, `deep_gemm_fp4_w4a4`, `deep_gemm_fp4_w4a8` | 有设备、CUDA、编译器、形状约束，部分层回退原精度 |
| EntroPack 无损 | `entropack_lossless_df11`, `entropack_lossless_tile_ans` | 压缩率依赖数据，不保证比 4-bit 小；不是路线 B 默认新增步骤 |
| EntroPack 有损 | `entropack_lossy_quant` | 连续 bpp 控制压缩，需模型质量依据 |
| EntroPack W8A8 | `entropack_lossy_quant_fp8`, `entropack_lossy_quant_fp8_raw`, `entropack_lossy_quant_int8`, `entropack_lossy_quant_int8_raw` | coded 与 raw 不同；不能用文件名中的 bpp 推断全部计算精度 |

模型文件还可能在导入时注册专用方法，例如 Ideogram FP8；不能用上述通用方法重解释专用 checkpoint。表以所核对源码为准，版本更新时检查 `core/quant/backends/` 和目标模型代码。

### 环境与硬件过滤

- bitsandbytes：检查当前 Python/PyTorch/设备与库版本支持；框架的 bitsandbytes 后端是 NF4/FP4，不虚构 `bitsandbytes_int8` method。当前 capabilities 不支持 compile，别默认给 NF4 路径加 `torch.compile`。
- torchao：文档要求 `torchao>=0.16`，还应匹配 PyTorch 与具体格式/设备；该后端导入多个接口，包安装成功不等于选中方法可运行。INT4/MX/NV 格式硬件限制分别核对。部分版本的默认 INT4 packing 依赖 mslk；缺失时可核对设备支持的其他 packing，而非认定所有 INT4 都不可用。`group_size` 不整除输入维度的层可能保留浮点；后端 marker/“替换层数”不等于真正低比特层数，应核对实际 weight 类型、形状及 packed 存储，将跳过层计入 U。
- kitchen：文档说明 CUDA 13 及以上；实际适配器在 torch CUDA build <13 或无 CUDA 时禁用其 cuda 实现，可能走其他 dispatch，不能仅凭版本宣称全部方法必定失败或一定加速。检查目标方法的小算子和实际 dispatch。
- DeepGEMM FP8：源码检查 CUDA、SM90/SM100；文档要求 CUDA>=12.9、C++20 `<format>` 编译器，源码安装及可用 nvcc（`CUDA_HOME`/PATH），首次运行会 JIT。FP4 路径要求 SM100。FP8 的 in/out_features 均应为 128 倍数；FP4 的 K 对齐 W4A4 为 256、W4A8 为 128，输出维度为 128 倍数，不支持的层保留原精度，要加入 U。
- EntroPack：当前文档要求 Python>=3.10、CUDA PyTorch>=2.10，cuda12/cuda13 包按环境匹配；适配器还检查 CompressedLinear 等 API。FP8 W8A8 路径要求 SM8.9+，INT8 路径 SM8.0+。元数据探测不等于这些 kernel 已验证。CUDA 兼容设备（如 PPU）需匹配厂商 CuPy；通用 CUDA CuPy 能导入也可能在 device 查询或 kernel 执行时报错。先检查设备查询、代表性矩阵的压缩/前向，再运行完整目标模型；修复依赖与最终运行记录分别保留。
- NVIDIA 路线不能直接套给 ROCm/MPS/NPU；查 `core/device`、GPU_support 文档与模型实际支持。多卡须采用已支持的并行实现，不能仅改 device 为设备列表。

### EntroPack 选档细节

框架普通 lossy lattice 的 `target_bpp` 范围为 0.001–11；W8A8 家族上限为 8。Skill 的默认整数档为 **7、6、5、4**，不用框架允许范围扩展默认路线。

| 默认档 | method | 实际流程与计算 | 核对项 |
| --- | --- | --- | --- |
| 7 / 6 bpp | `entropack_lossy_quant_int8` | `CompressedINT8Linear` 在后端内先把浮点权重量化成 INT8，再对 INT8 码做 lattice 编码；前向还量化激活，属于 W8A8 | SM8.0+、选中 kernel、scale/编码状态、实际峰值与画质 |
| 5 / 4 bpp | `entropack_lossy_quant` | 从原始浮点层直接构建 `CompressedLinear`，计算时按此方法解压；没有前置 torchao INT8 | packed 大小、临时反量化及实际峰值与画质 |

`entropack_lossy_quant_int8_raw` 直接存 INT8 码，不是目标 6/7 bpp 的二次编码方法；FP8 coded 也不是这里的 INT8 路线。不先调用 `torchao_int8_w8a16` 再把打包模块交给 EntroPack，也不用 MixedQuantizeConfig 同时命中同一层。“二次量化”由该单一后端配置完成；每个新候选从原始浮点层准备，不在失败结果上累计转换。

二次量化可能利用 INT8 计算提速，但不能保证任意模型/设备更快或画质相同；INT8 W8A16 能运行不证明 W8A8 可用，反之亦然。只验证将使用的方法；7/6 kernel 不兼容时排除它们，5/4 直接方法仍单独检查。不能沿用历史直接 6/7 bpp 的质量或峰值作为二次路径的实测证据。

bpp 只作用于压缩部分；敏感层、scale、编码表、header、未压缩组件和临时工作区另计。先估算最高可行且质量合格的档位，再实际验证，不逐档完整试跑。最低 4 bpp 常驻不足时，独立显存设备评估相同 4 bpp + CPU；统一内存跳过 CPU 转兼容 Disk。4 bpp 本身后端/质量不合格时，不能靠卸载让它变合格。

无损 df11 面向 BF16；tile-ANS 支持更多原始 dtype。无损存储不保证跨计算后端输出逐位一致，也不保证压缩率。lossy 包有自描述 header；旧无 header buffer 仅允许兼容无损方法。MOSS 等官方预量化 example 需显式下载/注册侧车，框架不自动读取任意相邻 JSON。

## 4. 路线 A：双组件量化，先常驻再按需卸载 4 bpp

原精度适用方案不足且允许量化后，全部 DiT、Text Encoder 先用 `torchao_int8_w8a16`；保留各自真实敏感层，VAE 等组件保留原生 dtype。常驻候选不传 CPU/Disk 四阶段配置或 `vram_limit`。

例如已选 7 bpp 二次路径，以下变量由目标组件确定：

```python
from diffsynth.core.quant import QuantizeConfig

dit_quantize = QuantizeConfig(
    method="entropack_lossy_quant_int8", exclude_modules=dit_sensitive_layers,
    backend_config_kwargs={"target_bpp": 7.0},
)
text_quantize = QuantizeConfig(
    method="entropack_lossy_quant_int8", exclude_modules=text_sensitive_layers,
    backend_config_kwargs={"target_bpp": 7.0},
)
model_configs = [
    ModelConfig(path=dit_path, quantize=dit_quantize),
    ModelConfig(path=text_encoder_path, quantize=text_quantize),
    ModelConfig(path=vae_path, computation_dtype=vae_dtype),
]
```

6 bpp 仅将两处 target_bpp 改成 6.0；5/4 bpp 同时改 method 为 `entropack_lossy_quant`，target_bpp 分别为 5.0/4.0。默认两类组件同方法同档位，多组件逐个配置；mode 保持 dynamic，dequant_once 会还原浮点常驻，不能用来满足压缩预算。

常驻阶段检查实际量化层、packed 类型、设备；必要敏感层排除不等于跳过整组件。4 bpp 只因容量不足时，沿用其原始浮点源、直接 EntroPack 4 bpp 配置，再使用下节 CPU 四阶段配置。在线量化到 CPU 的准备与驻留预算单独核对；CPU 阶段不要求整模型先在 GPU 量化完。用户指定的预量化输入不进入本节重新选档。

## 5. CPU Offload：仅适用于独立显存设备

本节用于前置原精度仅 Text Encoder 卸载、拒绝量化/方法或质量失败后的原精度卸载，以及 4 bpp 常驻仅容量不足后的量化卸载。最初已量化的模型可按其原表示检查兼容卸载；统一内存全部跳过。

以下 dtype 仅为 BF16 示例，真实组件要求 FP16/FP32 时逐组件替换。

### 仅 text encoder 卸载，DiT/VAE 保持 GPU

```python
compute_device = "cuda:0"
text_cpu = dict(
    offload_dtype=torch.bfloat16, offload_device="cpu",
    onload_dtype=torch.bfloat16, onload_device=compute_device,
    preparing_dtype=torch.bfloat16, preparing_device=compute_device,
    computation_dtype=torch.bfloat16, computation_device=compute_device,
)
model_configs = [
    ModelConfig(path=dit_path),
    ModelConfig(path=text_encoder_path, **text_cpu),
    ModelConfig(path=vae_path, computation_dtype=vae_dtype),
]
```

这使 text encoder 闲置时在 CPU，编码时整组件上 GPU；适用于编码阶段与常驻 DiT/VAE、激活相加后仍能放下的情况。DiT 与 Text Encoder 均保持底模原生精度，不新增 quantize。组件有多个 text encoder 时按真实依赖分别处理。

如果整个 text encoder 无法同时放下，且有细粒度映射，可令 `onload_device="cpu"`、`preparing_device=compute_device`，配合 vram_limit 在编码期间逐层调入。CPU 存储、GPU 计算；RAM 需容纳相应权重。未包装的 DiT/VAE 仍常驻，所以编码峰值必须包含它们。

### 动态 CPU Offload

```python
cpu_dynamic = dict(
    offload_dtype=torch.bfloat16, offload_device="cpu",
    onload_dtype=torch.bfloat16, onload_device="cpu",
    preparing_dtype=torch.bfloat16, preparing_device=compute_device,
    computation_dtype=torch.bfloat16, computation_device=compute_device,
)
component = ModelConfig(path=component_path, **cpu_dynamic)
# Pipeline.from_pretrained(..., vram_limit=total_gib - reserve_gib)
```

从低频组件扩大到高频 DiT 层时，检查生命周期、映射、RAM 和阶段峰值。原精度分支不新增有损转换；固定预量化分支不重选表示。量化 + CPU 可能增加复制/解压和传输开销，因此只在 4 bpp 常驻不足后启用，不作为 INT8 与更低常驻档位之间的中间态。

### 直接 EntroPack 4 bpp + CPU

使用上面的 `cpu_dynamic` 字典；dtype 是未压缩层的存储/计算 dtype，已量化层的 packed 格式由后端保留：

```python
dit_quantize = QuantizeConfig(
    method="entropack_lossy_quant", exclude_modules=dit_sensitive_layers,
    backend_config_kwargs={"target_bpp": 4.0},
)
text_quantize = QuantizeConfig(
    method="entropack_lossy_quant", exclude_modules=text_sensitive_layers,
    backend_config_kwargs={"target_bpp": 4.0},
)
model_configs = [
    ModelConfig(path=dit_path, quantize=dit_quantize, **cpu_dynamic),
    ModelConfig(path=text_encoder_path, quantize=text_quantize, **cpu_dynamic),
    ModelConfig(path=vae_path, computation_dtype=vae_dtype),
]
```

按实际 VAE 生命周期和峰值决定其是否也需要按原生 dtype 卸载；不能统一量化 VAE。受管理量化层由 `AutoWrappedQuantizedModule` 包装，onload/offload 只搬运后端表示；计算阶段按后端路径使用临时 GPU 层。BF16 阶段配置不表示 packed 权重变为 BF16，改变 computation dtype 也不恢复精度。核对包装是否生效、packed/scale/metadata 是否完整随层搬运，并评估临时副本峰值。

## 6. Disk Offload：RAM 不足或统一内存常驻不足时检查

```python
disk_config = dict(
    offload_dtype="disk", offload_device="disk",
    onload_dtype="disk", onload_device="disk",
    preparing_dtype=component_dtype, preparing_device=compute_device,
    computation_dtype=component_dtype, computation_device=compute_device,
)
component = ModelConfig(path=component_safetensors, **disk_config)
```

- 真正按层惰性读取要求兼容 `.safetensors`。`DiskMap` 对 `.pth/.bin/.ckpt` 有 `torch.load(weights_only=True)` 全文件回退，不能据此声称这些格式支持低 RAM 的 Disk Offload；要将完整加载需求计入 RAM。
- 分片列表必须完整，converter 必须能在 key→key 的字符串映射上执行；仅改名/筛选通常可行，reshape、拼接、数值运算等转换不满足这种路径。不是只要后缀正确就兼容。
- `offload=disk, onload=cpu` 会在 onload 阶段把所用组件权重缓存到 CPU；与两个阶段都为 disk 的配置资源成本不同。官方 example 使用该布局时必须算 RAM。
- preparing 允许在 GPU 保留已读取层；阈值不足则计算时暂读。根参数、当前层、解码激活、OS 缓存仍需资源，最小计算层也放不下时 Disk 无法解决。
- `DIFFSYNTH_DISK_MAP_BUFFER_SIZE` 控制 DiskMap 读取元素累计后的文件句柄刷新，单位是元素数，不是精确的 RAM 字节上限，也不是自动磁盘容量配额。
- **量化 + Disk 只支持匹配的预量化 checkpoint**：4 bpp + CPU 的 RAM 不足、统一内存常驻不足或最初已量化时均核对该条件。加载器检查 `load_prequantized=True`，在线量化 + Disk 直接报错。量化包装层按层取回完整 packed 权重与量化状态，method、metadata 和排除层必须匹配。
- 原精度分支保留原始 computation dtype，固定预量化分支保留原方法与 packed 状态，不能采用 `preparing_dtype=torch.float8_*` 的隐含有损暂存。文档/旧 helper 中可能有 FP8 默认值，以实际片段为准。

## 7. 预量化输入分流、在线量化与保存成本

**最初已量化的输入先分流**：从已匹配 example、模型注册 `quant_config`、checkpoint 的 quantization metadata 和真实模块/packed 格式确认，不只按 NF4/INT8 等文件名推断。包含量化组件的固定模型包整体跳过新增量化选择，不顺带量化其余浮点组件。保留加载 packed 数据所需配置、`load_prequantized=True`、原 method、敏感层和侧车注册；“跳过量化逻辑”不是删除加载器的重建步骤。只检查保持当前表示的常驻/兼容卸载，统一内存跳过 CPU。不能调用二次量化、外传不同 QuantizeConfig 覆盖、dequant_once、转 BF16 或换浮点底模冒充保持原模型；用户明确要求转换时才另行处理。

**在线量化**先创建模型并加载原始浮点权重，再替换目标 Linear。无显存管理路径通常先把浮点组件放 CPU、逐层到 compute_device 量化，并把量化结果积累到 GPU；RAM 至少考虑当前完整浮点组件，GPU 考虑已驻留组件、已量化部分、当前浮点层、量化结果和后端临时量。不能用最终 W4 代替准备峰值。

常驻路线在准备阶段临时加载浮点组件到 CPU 是正常加载过程，不是推理 Offload。框架另支持启用 CPU 显存管理后，在 CPU 保存组件并逐层转换（4 bpp + CPU 准备路径），量化配置的 `compute_device` / `model_device` 支持这种流式准备；不意味着浮点源文件在 RAM 中也按层惰性加载。普通 loader 会把该组件所需 state_dict 物化，再量化。后端若使用 FP32 误差计算或额外 workspace，还需增加余量。

**预量化加载**先创建后端 shell，再用 checkpoint 的 packed 权重、scale、metadata 重建状态，省去在线转换；仍有加载、搬运和后端解压开销。使用官方包时保留已注册 `quant_config`：`ModelPool.resolve_quant_config` 会优先采用它，并忽略外传 quantize。不能靠再传一个 method 改变固定包的位宽。

**保存预量化包**：`utils/quant/serialization.py:save_quantized_model` 会加载原始权重、量化、取 CPU state_dict，再序列化。当前实现采用 `ModelPool.default_vram_config()`，只改 computation_device，不继承传入 ModelConfig 的 Offload 配置，也不接收推理 vram_limit。要检查整组件量化后驻 GPU和额外 CPU 序列化成本，不能因低显存推理能跑就默认保存也能跑。一次推理不默认新增大型转换任务，优先已有匹配包。

自制包需要模型识别条目与正确的 quant_config，保存时 method/排除层/混合配置保持一致，并标记 `load_prequantized=True`。按官方流程在脚本加载前 `MODEL_CONFIGS.append(entry)` 可实现进程内注册，无需修改上游框架；只写一个侧车 JSON 并不会自动注册。框架结构 hash 根据 key/shape 帮助识别模型，不是权重内容校验或 revision 证明。

## 8. 激活、解码与运行阶段优化

| 失败阶段 | 优先检查 |
| --- | --- |
| 初始化/浮点加载 | 全组件加载设备、文件格式、转换器、RAM/cgroup、是否实际开启包装 |
| 在线量化 | 原始浮点组件和单层转换峰值，是否有兼容预量化包 |
| 文本/条件编码 | 多 encoder/参考输入长度；常驻 DiT/VAE 与编码临时权重是否叠加 |
| 去噪 | 实际分辨率/帧数/batch/文本长度、Attention 工作区、缓存、DiT 映射与调入粒度 |
| VAE/视频/音频解码 | 真实支持的 spatial/temporal tiling、tile 大小/重叠、framewise 或分块解码 |
| 保存/编码 | 输出张量是否仍在 GPU、视频/音频 CPU 缓冲与编码进程 RAM |

- tiled VAE 减少解码/编码激活，可能增加时间和重叠开销；不降低 DiT 权重大小，不保证解决去噪 OOM。tile 参数单位可能是 latent、像素或帧，必须读对应实现；重叠必须符合接口。
- 对已明确的尺寸、帧数、数量、时长不能自行缩小；只有用户允许才调整。减少推理步数主要省时间，通常不会按比例降低单步峰值。
- Attention backend、`torch.compile`、缓存和并行是独立变量：可能提速，也可能新增工作区/编译峰值或改变数值结果。保留普通 example 默认，只有实际适用且符合约束时再改；不把训练的 gradient checkpoint/offload 参数当成通用推理优化。
- LoRA：未管理底模常走融合，管理/量化模型需按实际 hot-loading/融合支持处理，核对加载顺序、alpha、匹配层数与目标模块；不能把量化后不能融合的问题靠强行 `.to(dtype)` 绕过。
- 用户需要测量时覆盖加载、量化、编码、去噪、解码及保存。设备采样由外部监测进程执行；精确 CUDA 分配器峰值、模拟显存配额及必要的阶段观察放独立测试包装器，交付的模型脚本保持 example 形式。CUDA 分配器峰值只代表当前进程 PyTorch，不是整卡总量；RAM 前后快照也不是峰值。质量和资源均满足后停止调参，不为证明所有候选都能跑而继续试验。

## 9. 模型差异与源码索引

以下是需要保留的差异，不是完整模型目录；配置描述对应当前选中的 checkout，任何模型都应重新读取所选 example 与实际组件的 `ModelConfig`。示例中出现的量化设置不是所有任务的默认值；是否转换及实际命中的 Linear 仍按任务和模型核验。

| 目标/示例 | 关键区别 |
| --- | --- |
| `examples/z_image/model_inference*/Z-Image-Turbo.py` | 三组件 + tokenizer；普通脚本是默认起点；Pipeline 没有通用 `tiled` 调用参数，不从别的模型照搬 |
| `examples/moss_image/model_inference_low_vram/MOSS-Image-Base.py` | DiT/text encoder 使用 CPU offload/onload、GPU preparing/computation 的 BF16 配置及 `entropack_lossy_quant_int8_raw`；VAE 单独 `computation_dtype=torch.float32`，不可全局 BF16 化 |
| `examples/moss_image/model_inference_low_vram/MOSS-Image-EntroPack.py` | DiT/encoder 独立选择 Base/SFT 包；显式读取侧车并注册；代码所选包与顶部说明可能不同，以实际赋值/用户变体为准 |
| `examples/qwen_image/model_inference_low_vram/Qwen-Image-2512.py` | 三个 ModelConfig 共享 CPU offload/onload、GPU preparing/computation 的 BF16 配置及 `entropack_lossy_quant_int8_raw`；当前不是 Disk/FP8 暂存配方。实际量化范围由各组件 Linear 决定；原精度方案应移除新增 quantize，不能将此例当作无损卸载 |
| `examples/wanvideo/model_inference_low_vram/Wan2.1-T2V-1.3B.py` | 组件共享 BF16 CPU/GPU 四阶段配置及 `entropack_lossy_quant_int8_raw`；T5/VAE 有 pth，考虑全文件 RAM；保留 example 的 2 GiB 余量和实际 tiled/保存 FPS |
| `examples/minimax_h3/model_inference_low_vram/MiniMax-H3-NF4-FL2VA.py` | 四个模型组件 + processor，独立视频/音频 VAE；保留变体、注册配置及至少 2 GiB 余量，使用匹配 AV 保存逻辑 |
| `examples/ltx2/model_inference_low_vram/LTX-2.5-T2AV-TwoStage.py` | 六个模型组件共享 BF16 CPU/GPU 四阶段配置及 `entropack_lossy_quant_int8_raw`，保留其真实排除层；另有 tokenizer 和 stage2 LoRA。当前此例没有 FP8 e5m2 暂存；两阶段、空间/时间 tile 和音视频保存分别核对 |
| `examples/ideogram4/model_inference_low_vram/ideogram-4-nf4.py` | 使用预量化组件及 BF16 CPU/GPU 四阶段配置，另含可选 unconditional_transformer；保留实际注册和所需组件，不据文件名改成通用量化方法，也不因组件可选就自动改变用户配置 |

解码接口核对：Qwen-Image 用 `tiled/tile_size/tile_stride`；Wan 用 `tiled` 与 tuple tile_size/stride；MiniMax-H3 用 `tiled/tile_size/tile_overlap`；LTX2 区分 `tile_size_in_pixels`、`tile_overlap_in_pixels`、`tile_size_in_frames`、`tile_overlap_in_frames`。默认值、单位和路径以当前 Pipeline 签名及 VAE 实现为准。

在用户实际 DiffSynth-Studio 根目录按以下路径核对。此参考已检查当前仓库的低显存 example 配置布局及下列核心实现；依赖版本/框架更新时重查对应部分，不将文档示例数字当成通用硬件阈值。

| 知识 | 框架中的依据 |
| --- | --- |
| 四阶段、CPU/Disk、阈值语义 | `docs/zh/Pipeline_Usage/VRAM_management.md`；`docs/zh/Developer_Guide/Enabling_VRAM_management.md` |
| 方法、选层、混合、预量化、保存 | `docs/zh/Pipeline_Usage/Quantization.md`；`docs/zh/API_Reference/core/quant.md` |
| 入口参数与加载设备 | `diffsynth/core/loader/config.py`、`model.py`、`file.py`；`diffsynth/models/model_loader.py` |
| 实际包装/调度/磁盘行为 | `diffsynth/core/vram/layers.py`、`disk_map.py`、`initialization.py`；`diffsynth/diffusion/base_pipeline.py` |
| 细粒度支持与模型识别 | `diffsynth/configs/vram_management_module_maps.py`、`model_configs.py` |
| 各后端实际行为/限制 | `diffsynth/core/quant/config.py`、`base.py`、`backends/{bitsandbytes,torchao,comfy_kitchen,deep_gemm,entropack}.py` |
| 量化另存与进程内注册 | `diffsynth/utils/quant/serialization.py`；Quantization 文档“最佳实践” |
| LoRA、设备与 Attention | `docs/zh/Pipeline_Usage/Model_Inference.md`、`GPU_support.md`；`diffsynth/core/device/`；对应 Pipeline/模型 |
| 模型特有 dtype、tile 与单位 | `examples/<系列>/model_inference/`、`model_inference_low_vram/`；对应 `diffsynth/pipelines/`、VAE 模型与 `docs/zh/Model_Details/` |
