# 训练显存不足：定位与选型

仅在资源选型、OOM 或多卡配置时读取。路径相对活动 DiffSynth-Studio 根目录；先看 `diffsynth/diffusion/runner.py` 和目标 `train.py`，不要把推理卸载策略直接复制到训练。

## 先判断占用来源

训练峰值包括冻结/可训练权重、梯度、优化器状态、激活与 Attention 临时量、量化/反量化暂存、通信 bucket 和保存聚合。输入图像面积、帧数/时长、条件图/参考数量及文本长度都可能增加激活；小图能跑不代表最大规格能跑。量化仅覆盖命中的 Linear，BF16 文件大小除以四不是整机峰值估算。

保留完整异常、发生阶段、实际输入规格、rank、显存 allocated/reserved/设备总占用与 RAM/cgroup 记录。必要时在任务自己的诊断脚本中用 `reset_peak_memory_stats()`、`max_memory_allocated()`/`max_memory_reserved()` 并同步设备；外部设备占用还需驱动工具观测。探测脚本是快照，没有预留内存；CPU offload 也要计入 FP 权重、pinned 副本、checkpoint 和 DataLoader workers 的 RAM。

在 OOM 前先按组件区分状态负担：LoRA 常见的是冻结底模与激活，全量还增加目标权重、梯度和 optimizer 状态，视频/长序列还需关注 Attention workspace 与条件缓存。分别评估加载、Hook 初始化、编码、前向/反向、首次 optimizer 更新与保存的 CPU/GPU 峰值；不要将各阶段独立最大值机械相加。优先保持数值语义、少传输的最小有效组合，再比较预处理到保存的总耗时与资源成本。

## 自动配置的选型与试跑闭环

自动配置以选出并验证本次作业的一份配置为目标，不要求穷举技术或制作通用预测器。使用现有日志/测量时，先核对源码及改动、模型 revision、设备/可用容量、dtype/量化、可训练范围、最大输入与条件、进程数、有效 batch 和缓存状态；任一关键条件不匹配就降低证据可信度。

1. **过滤并建立基线**：先固定用户约束、目标参数和输入规格，以同变体官方全量/LoRA 配方为基线。核对入口支持、模型来源不重复、FP8/量化/Disk 实际匹配，以及设备、RAM/磁盘与保存格式门槛。基线已满足容量/速度要求时直接采用，不为更低显存增加复杂机制。
2. **按瓶颈产生必要替代**：激活压力优先 checkpoint/offload；冻结权重压力才评估量化/权重 offload；全量梯度或 optimizer 压力评估相应 ZeRO stage/状态 offload；重复冻结编码才评估 split。通常只需基线和一个有依据的替代。逐次改变与瓶颈有关的因素，组合技术时重验兼容性，不在无证据时随机搜索全部后端。
3. **在预算内做有界测量**：复用真实小样本，保持最大规格、目标层和有效 batch，覆盖加载/转换、pinned 设置、编码、实际更新和保存。分别记录各 rank 的 GPU 峰值、有效 RAM 峰值/余量、加载/预处理/保存耗时，以及 warmup/编译后的代表性每次优化器更新耗时。workers=0 的容量结果不替代正式 workers 的 RAM 验收；无法完成更新/保存或预算不足时标为未验证，不伪造性能结论。
4. **据总成本选择并停止**：同工作负载下，结合一次性加载/量化/预处理、预计更新数乘更新耗时、保存/验证和多卡资源成本比较；短作业尤其要计入初始化与缓存成本。测得已满足约束且没有值得验证的收益时停止搜索；OOM 先归因，再按瓶颈做最小调整，达到授权试跑预算或同一瓶颈无改善时停止重试并报告限制。
5. **固化与复核**：只生成选中的训练命令/必要配置，简述采用和排除方案的依据、测量覆盖与余量。正式启动前复核设备可用内存；环境、模型、输入、workers 或可训练范围改变时重验相关阶段。准备任务只给出候选及待验项，不将静态推算写成已测峰值。

## 设备后端与实际 dtype 门槛

| 设备/内存架构 | 配置前的过滤条件 |
| --- | --- |
| NVIDIA CUDA 独立显存 | 按目标入口与后端评估量化、Hook Offload、DDP/ZeRO；验证 pinned-memory、算子和设备间传输 |
| ROCm（也使用 `torch.cuda` API）或其他兼容设备 | 从 `torch.version.hip`、实际构建与算子判断，不仅凭设备名/`cuda` 字符串套用 NVIDIA 方法 |
| Apple Silicon MPS 等 UMA | 不默认启用当前 CUDA Hook Offload；RAM/显存共享物理容量，迁移不等于新增容量 |
| 其他 NPU/后端未知 | 使用明确支持的训练入口和厂商运行环境；缺少验证时不给出 CUDA 配方已兼容的结论 |

`OffloadTrainingManager.after_backward()` 直接调用 `torch.cuda.synchronize()`，PinnedArena 依赖 pin-memory 分配；这套实现不是设备无关的通用 offload。探测报告里的统一/独立/未知内存拓扑只是一项事实，不证明某个方案能跑。许多入口显式用 `torch.bfloat16`，SD/SDXL 等入口还有不同 dtype；核对模型、输入、算子与反向的实际 dtype。Accelerate `mixed_precision`、CUDA/BF16 能力查询或导入成功都不能代替真实小规模算子/训练验证。

| 失败阶段/现象 | 合理动作 | 容易误用的做法 |
| --- | --- | --- |
| 模型加载/在线量化时 OOM | 检查当前 device、全精度暂存与分片；核对入口的 CPU 初始化支持；考虑可识别的预量化底模或后端支持的分层量化 | checkpointing 尚未执行，不能减少加载权重峰值；CPU 初始化本身也不会减少 runner 随后整体 `.to(cuda)` 的容量 |
| 文本/VAE/条件编码时 OOM | 确认确实冻结；考虑支持的 split 预处理、冻结组件 FP8/量化或单卡 offload | Split 仍须先完成编码，不会自动消除第一阶段单样本峰值；不凭空增加通用 VAE tiling CLI |
| 去噪前向/Attention OOM | checkpointing；核对实际形状、额外条件、目标后端；必要时讨论数据规格调整 | 不能把量化或梯度累积当成激活一定能放下的保证 |
| backward OOM | 核实 checkpoint flag 确实传入目标 blocks；checkpoint offload、梯度/参数分片 | 不把一次 forward 成功当成训练可行；冻结权重仍可能需要向输入回传梯度 |
| 首次 optimizer.step OOM | 排查 trainable 范围；ZeRO/optimizer CPU offload；经验证的自定义低内存优化器 | 默认 model CPU offload 不移动可训练权重和 optimizer |
| 保存时 OOM | 检查 ZeRO gather、CPU RAM、输出空间与导出格式，预留保存峰值 | ZeRO 分片训练能跑不代表全量权重聚合一定放得下 |
| 进程被杀、主机 RAM/cgroup 达限 | 减少 workers/并行预处理，检查 pinned 副本和缓存；重新选择 offload 方案 | 继续加 CPU offload、只看主机总 RAM、不检查作业限制 |

可以结束本作业失败进程后重新测量；不清理其他用户任务。`empty_cache()` 不能释放仍被引用的权重/激活，也不能修复真实容量不足。只有证据指向碎片化时才考虑 allocator 调整。

## Gradient checkpointing 与有效 batch

`--use_gradient_checkpointing` 以反向重算换激活显存，通常是全量/LoRA 的首选。`--use_gradient_checkpointing_offload` 将保存的张量放到 CPU；在通用 `core/gradient/gradient_checkpoint.py` 中它自身会进入非 reentrant checkpoint + `save_on_cpu()` 分支，但必须确认目标模型实际调用该 helper。视频尤其可能受益，代价是 RAM/传输和计算增加。

DeepSpeed checkpoint 分支的优先级更高：当开启 checkpoint flag 且 `deepspeed.checkpointing` 已配置时，采用 DeepSpeed 接口；这时 PyTorch 的 checkpoint offload flag 不会叠加到同一次 helper 调用。没有 grad-enabled 输入的调用会直接 forward，不能只凭 CLI 判断每层都 checkpoint 了。

标准 runner microbatch=1。增加 `--gradient_accumulation_steps` 用于扩大有效 batch，不能缩小当前单样本的激活。DDP 中有效 batch 通常约为 `卡数 × 累积步数`，尾部、sampler 补齐需核对；改变卡数/累积后重新核算更新数、学习率和保存时机。

## 单 GPU：框架逐层 CPU offload

依据 `docs/zh/Training/Offload_Training.md`、`core/offload_training/{manager,offloader}.py` 与 runner：

| 配置 | 权重、梯度与 optimizer 的位置 | 适合的情况 |
| --- | --- | --- |
| 不开启 | 正常驻留 GPU | 能放下时优先避免传输成本 |
| `--enable_model_cpu_offload` | 冻结权重保留 CPU 副本按需 onload；可训练参数与 optimizer 仍在 GPU | LoRA 冻结底模占用过大；对全量 trainable 部分帮助有限 |
| 再加 `--enable_optimizer_cpu_offload` | 可训练参数也逐层搬运，反向后梯度移到 CPU，optimizer 在 CPU 更新 | 全量训练状态过大且 RAM/CPU 带宽足够 |

约束与检查：

- **显式单进程，且不使用 DeepSpeed**。该路径不 `accelerator.prepare(model)`，多进程不会自动梯度 allreduce。不要将“多个进程启动成功”视为 DDP 正确。
- optimizer offload 依赖 model offload；按框架说明使用 `--gradient_accumulation_steps 1`，不要假定 parser 会阻止无效组合。自定义 optimizer 还必须支持 CPU 参数。
- 确认目标 `train.py` 初始化时就将模型放 CPU。大多数入口使用 `device="cpu" if args.enable_model_cpu_offload ...`，仍需逐入口核对；不要因为通用 parser 有 flag 就认为所有入口已正确透传。
- 可以配合目标支持的 PyTorch checkpointing。`--cpu_offload_split_threshold` 是实验性参数，默认不启用；当前实现将输入乘 `1024*1024` 后与参数**元素数**比较。例如 100 对应约 100 Mi 个参数的阈值，并非 BF16 权重占用 100 MiB，不能按 CLI 标注的 MB 推算精确字节容量。
- 最大片段/单层及激活仍须能放入 GPU；非训练参数的 pinned 内存也可能触发 RAM 上限。量化 tensor subclass 与该 hook 管理路径的组合要单独验证，不把分别支持两项推导为组合已验证。

**准备期 RAM 也要验收**：`PinnedArenaPool.from_model()` 先按冻结参数和 buffers 规划分块，实际 pinned 大块在首次分配时延迟创建；Hook 初始化中的 `StaticParamOffloader` 再建立 CPU 副本并逐个替换原参数。期间未释放的原权重、可能的 `.cpu().contiguous()` 临时量与已分配 arena 可以同时存在，分块取整/增长和 fallback 也有开销。不能只按最终冻结权重的一份体积判断 RAM 足够，也不把峰值固定假定为两倍。分别测量模型加载、arena/Hook 设置、首轮前后向、首次 optimizer 更新与保存，加上可训练参数/状态、worker 和预处理内存；RAM/cgroup 或 pinned 分配不足时先排除该方案。

## 多卡：DDP 与 DeepSpeed ZeRO

DDP 每卡保留模型副本，主要提高吞吐，不把几张小卡自动变成一张大卡。当前 runner 为部分冻结量化 tensor subclass 设置 DDP 同步忽略列表；所有 rank 必须加载相同底模，仍要实际验证同步与后端兼容。不要据此推广为 ZeRO 与所有量化方法兼容。

| 策略 | 分片/卸载范围 | 对 OOM 的意义 |
| --- | --- | --- |
| ZeRO-1 | optimizer 状态分片 | optimizer 状态是主因时有帮助 |
| ZeRO-2 | optimizer 状态与梯度分片 | 权重本身仍需容纳；CPU optimizer offload 可进一步减 GPU 状态 |
| ZeRO-3 | 再分片参数 | 常驻模型容量不足时考虑；通信/参数聚集有峰值 |
| ZeRO-3 + CPU offload | 以配置卸载 optimizer/参数 | 可用于单 GPU 或多 GPU；RAM、传输及当前配方支持是约束 |

激活分片/CPU checkpoint 是独立配置，不会仅因 stage=3 自动全部生效。某些官方 ZeRO-2 YAML 包含 `offload_param_device: cpu`，不能因此声称 stage=2 已分片/卸载所有模型权重。

实际配置来源：

- 全量：优先目标模型的 `full/accelerate_config_zero*.yaml`。例如 Qwen-Image full 默认 shell 用 `accelerate_config_zero2offload.yaml`，另有 zero3；这些示例通常写死 8 进程和累积 1，复制到任务目录后按硬件调整。
- 更低显存的 Qwen LoRA：`examples/qwen_image/model_training/special/low_vram_training/` 下的 shell、`deepspeed_zero3_cpuoffload.yaml`、`ds_z3_cpuoffload.json`。该 YAML 默认 **1 进程**，不是只有多卡才能用的功能；shell 还使用拆分预处理。

生成配置时检查：

1. YAML 的 `distributed_type`、`num_processes`、`num_machines`、精度、`zero3_init_flag`，以及其引用 JSON 的实际路径。复制 YAML 后修正 JSON 路径；框架根目录执行仍要路径正确。
2. JSON 的 stage、offload、`gradient_accumulation_steps`、microbatch=1、`train_batch_size`/`auto` 与 CLI 一致；不只改一处累积值。
3. 加载容量不足时核对入口是否支持 `--initialize_model_on_cpu`（Qwen/Wan 等有，Z-Image 等当前没有）；这个 flag 不等同于持续训练 CPU offload。ZeRO-3 初始化还依赖 Accelerate/Transformers 的真实状态与 loader 分支。
4. ZeRO-3 CPU offload 配合 checkpoint 时采用框架适配的 DeepSpeed checkpoint 路径，参考 JSON 中保留 `activation_checkpointing`，同时打开训练 checkpoint flag。runner 的 `initialize_deepspeed_gradient_checkpointing()` 负责配置；不能只复制普通 YAML 就保证该路径生效。
5. 聚合保存所需选项（例如 `zero3_save_16bit_model` 或 JSON 的 `stage3_gather_16bit_weights_on_model_save`）、所有 rank 的保存参与及 CPU/GPU 峰值。ModelLogger 先调用 `accelerator.get_state_dict()`，再过滤可训练权重；LoRA 产物小也可能在 gather 时产生大峰值。

不要同时添加框架 `--enable_model_cpu_offload` / `--enable_optimizer_cpu_offload`；ZeRO 自己的 offload 是另一套机制。量化 loader 分支没有普通浮点分支相同的 ZeRO-3 state_dict 装载处理，不能承诺 QLoRA + ZeRO-3 自动兼容。

## 冻结权重存储与 optimizer 优化

`--fp8_models` 通过 `parse_vram_config(fp8=True)` 设置 FP8 存储、BF16 计算，只用于不直接更新的参数。LoRA 底模可以冻结后用它，全量训练的目标组件不能如此处理；全量训练中冻结的 encoder/VAE 可独立评估。它与 `--quant_options` 的专用 Linear 量化不是同一机制；不要默认在同一组件上叠加。精度误差和效果应验收。

**当前 FP8/Disk 匹配不支持本地分片列表**：`parse_model_configs()` 先将 `fp8_models` / `offload_models` 以逗号拆成字符串列表，再直接判断 `path in ...`，没有量化选项的 `normalize_quant_key()`。

| 组件表示 | 当前 FP8/Disk CLI 是否可匹配 |
| --- | --- |
| `--model_paths` 中的单文件字符串 | 不含分隔符歧义时，按完全相同字符串匹配 |
| `--model_paths` 中的嵌套分片数组 | 不匹配；传列表文本也无效，可能无警告地普通加载 |
| `--model_id_with_origin_paths` 的 `repo:pattern` | 按完全相同组件字符串匹配 |

预检可只调用真实 `parse_model_configs()` 而不下载权重，检查相应 ModelConfig 的四阶段 dtype/device 与 `clear_parameters`，再核对实际加载。需要 FP8/Disk 的分片组件应使用兼容的 ID 配方；离线时仍需正确本地缓存与 tokenizer/processor。不能表达目标组合时明确限制，不拆散组件的分片冒充多个独立模型，也不因命令里写了 flag 就认定优化生效。同一组件同时匹配 FP8 与 Disk 时当前 `parse_vram_config()` 优先 FP8，应避免含糊叠加。

`--customized_optimizer bitsandbytes.optim.Adam8bit` 是当前 parser/runner 支持的自定义 optimizer 类路径示例，压缩**可训练参数的 optimizer 状态**，不是 NF4 底模量化。先验证依赖、硬件、目标参数和数值行为；不能保证每种 CPU offload/ZeRO/packed 权重组合可用，也不要将其称为官方所有模型已验证配方。默认 AdamW 与带 `--warm_up_steps` 的 LambdaLR 不应被未经说明替换；切换资源方案后核对实际成功更新和 lr，不以数据迭代数推断 warmup 进度。

`save_steps` 与累积边界要分别核对：ModelLogger 在每次 `accumulate(model)` 内部的迭代后计数并可能保存，此时新梯度可能还未用于更新。间隔设为累积倍数可以对齐普通完整窗口，但尾部同步、跨 epoch 与分布式行为仍须实测；文件名与通用权重导出不包含待累积梯度，无法精确恢复该中间状态。

## 拆分预处理的容量边界

参照主流程的 `sft:data_process` / `sft:train`。只有冻结且可复用的计算移至缓存；如果训练 text encoder 或有条件分支需要梯度，不能把这些结果错误缓存为常量。`--offload_models` 实际产生 disk VRAM config 并 `clear_parameters=True`，适用于第二阶段不再使用的组件，不是用于在每层 CPU/GPU 之间搬运可训练权重。在线量化与 disk offload 的同组件组合在 loader 中会报错，见量化参考。

Disk 还需验收 checkpoint 格式与 `state_dict_converter`。`core/vram/disk_map.py` 的 `fetch_rename_dict()` 先把每个原 key 映射到 key 字符串，再调用 converter；转换结果必须仍指向可读取的原 key。纯重命名/筛选可以兼容，需要真实 Tensor 的 reshape、split、concat 或数值计算则不能据此成立；映射通过后仍核对读出 Tensor 的 shape/layout，不能遗漏原本必需的数值转换。真正按需读取优先采用兼容的 safetensors；当前非 safetensors 分支会 `torch.load` 整个文件，不能据 Disk 配置承诺同样的低内存加载。发现转换器或文件格式不满足本次容量条件时排除该配置，不通过写入 flag 宣称已卸载。

比较原生、缓存、量化、offload 或 ZeRO 时保持相同输入规格/可训练范围，记录预处理、加载/量化、稳定训练和保存的总成本，不只比较最小显存数字。
