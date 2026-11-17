---
name: diffsynth-user-training
description: >
  Lead preparation or execution of full-parameter and LoRA fine-tuning for supported
  DiffSynth-Studio models, including requests that also need dataset preparation.
  Use the dataset skill for data steps; dataset-only requests use that skill directly.
  Select the exact model recipe,
  diagnose training OOM, and configure checkpointing, quantized LoRA, CPU offload,
  split preprocessing or DeepSpeed where supported. Check saved weights and resume
  semantics. 用于全量/LoRA 训练、训练显存不足、多卡配置及训练产物验证；不新增模型架构或特殊训练算法。
---

# DiffSynth-Studio 用户模型训练

以用户实际运行的 DiffSynth-Studio 源码、对应模型变体的官方配方及真实硬件为依据，完成数据验收、配置、训练和产物验证。本 Skill 聚焦标准监督训练的**全量与 LoRA**；量化 LoRA、拆分预处理与显存优化属于执行配置。蒸馏、DMD2、OPD、Template 等专用方法超出这里的默认流程，不能仅更改 `--task` 套用本文。

用户只要求准备时生成可复现命令；要求执行时在已授权的算力、时间和数据范围内运行。不要把 smoke test、loss 下降或后台进程存活称为训练完成。

用户要求准备或运行标准全量/LoRA 训练时，以本 Skill 主导完整任务；涉及数据时在第3节显式读取 dataset Skill，不依赖它再次被自动检索。用户仅要求格式解释、数据检查或转换时，由 dataset Skill 直接处理，不额外启动训练。

## 1. 先理解用户需求

从对话、项目和已有文件确定下列信息，只询问真正缺失且影响方案的部分：

| 要确定的内容 | 对后续决策的影响 |
| --- | --- |
| 精确模型、变体、底模路径/revision，训练任务与预期效果 | 定位真实训练与验证入口，区分生成、编辑、视频和音频条件 |
| 全量还是 LoRA；LoRA 的组件、rank、目标层；是否从已有权重继续 | 决定可训练参数、显存开销、恢复与保存格式；“全量”通常指指定组件，例如 DiT，并非整个 Pipeline |
| 原始数据或 metadata、标注/配对、独立验证输入 | 交由数据集 Skill 推导字段及验收，不能用演示数据替代 |
| 实际 Python/源码目录、GPU 型号/每卡容量与数量、可用 RAM/磁盘 | 判断单卡、DDP、ZeRO、量化及 offload 是否可行 |
| 输入尺寸/帧数/时长、有效 batch、`dataset_repeat`、epoch/目标更新数、预算、输出位置；准备还是执行 | 明确数据曝光量、更新次数与实际作业规模；仅指定 epoch 不等于指定 repeat |
| 是否接受有损低精度、CPU 传输开销或缓存随机增强 | 决定数值与速度折衷，复用用户已经表达的偏好 |

未指定训练方式时说明适合的选择与理由；不要因 OOM 静默把全量改为 LoRA、降低分辨率/帧数、减少目标层或改变有效 batch。有损量化与无损压缩须分别说明。

## 2. 锁定当前版本的训练配方

从框架根目录查找并读取 `examples/<family>/model_training/train.py`，精确变体的 `full/` 或 `lora/` shell，以及 `validate_full/` / `validate_lora/`。不要因为目录存在就断定所有变体都支持同样的模式；例如当前 Music 提供全量配方，不能承诺现成 LoRA 闭环。检查模型文档、parser、构造参数、`task_to_loss` 和 launcher；没有兼容入口时明确缺口。

先记录 checkout 的 commit、本地改动与导入来源，再核对 `diffsynth/diffusion/{parsers,training_module,runner,logger}.py`；低显存还要沿目标 Pipeline/model 的 checkpoint 调用、`core/loader/model.py`、`models/model_loader.py` 和 `core/offload_training/` 追踪。框架路径均相对于**选中的 checkout**，不把另一份 editable install 当成本次源码；切换版本后重新核对参数和执行语义。

参考文档为 `docs/zh/Pipeline_Usage/{Model_Training,Quantization}.md` 与 `docs/zh/Training/{DeepSpeed,Offload_Training,FP8_Precision,Split_Training}.md`。文档或历史笔记与源码不一致时以实际执行代码为准：当前通用 runner 提供可选 loss logger、`--customized_optimizer` 和带 `--warm_up_steps` 的线性 warmup `LambdaLR`；不要套用旧版本的调度器或其他专用 runner 的假设。

## 3. 引用数据集 Skill 并完成训练侧交接

涉及字段说明、数据检查、转换或缓存数据验收时，优先读取同级 [diffsynth-user-dataset/SKILL.md](../diffsynth-user-dataset/SKILL.md)，按用户范围执行其中流程，不重复维护另一套数据格式规范。若同级目录缺失，先检查用户给定的 skills 根目录；仍不可用时，直接根据目标 `train.py` 的 Dataset、operator 和消费者完成必要适配与抽样验收，不仅因专项 Skill 缺失停工。

传入精确模型/任务、真实训练入口、原始数据和期望规格；取得 metadata/base path、有效样本数、字段类型、文件加载键、额外条件、验证样本与未验证项。训练侧继续核对 `UnifiedDataset → operator → get_pipeline_inputs() → Pipeline 消费者`：`data_file_keys` 和 `extra_inputs` 含义不同，首尾帧等条件可能由监督视频派生。用训练入口相同的参数与 operator 抽样调用 `dataset[i]`；静态检查通过不能代替真实消费者验收。只检查时不擅自改写数据。

## 4. 根据瓶颈选择显存方案

在实际 Python 中运行 [scripts/probe_environment.py](scripts/probe_environment.py)：

```bash
/path/to/python /path/to/diffsynth-user-training/scripts/probe_environment.py \
  --studio /path/to/DiffSynth-Studio --torch --import-diffsynth \
  --model-dir /absolute/model/path --output-dir /absolute/run/path
```

按需用 `--output` 写到任务报告目录。检查导入来源、设备查询、每卡空闲显存、RAM/cgroup 和磁盘；报告的 `target_readiness=not_assessed`、VRAM reserve 及权重头信息都不是训练容量证明。该工具不测张量算子、训练峰值、I/O 吞吐或多卡通信，必要时另外测量；不要用推理 `vram_limit` 代替训练预算，也不要把多卡容量简单相加。

区分实际导入与安装元数据：`requested_source_import_verified` 表示实际导入已匹配指定源码；`distribution_origin_match=mismatch` 会独立列入 `warnings`，提示旧 editable 元数据可能影响版本/依赖判断，不自动否定已匹配的源码。实际源码不匹配或导入失败仍须修复；`diagnostic_status=collected` 也不表示训练已验证。

先按设备后端过滤方案：当前 Hook Offload 有 `torch.cuda.synchronize()` 和 pinned-memory 假设，不默认用于 MPS/其他 NPU；ROCm 虽也使用 `torch.cuda` API，仍须验证实际实现与算子。UMA/共享物理内存不能将 RAM 与显存重复计入容量。许多入口显式设置 BF16/其他 dtype，修改 Accelerate `mixed_precision` 不会自动改写这些模型/输入 dtype；先验证目标设备的真实前向与反向路径，详见显存参考。

**遇到 OOM、选低显存方案或配置多卡前，读取 [references/memory.md](references/memory.md)**，按失败阶段诊断和配置。选择量化或预量化底模时再读 [references/quantization.md](references/quantization.md)。快速分流：

| 实际瓶颈 | 优先检查/采用 | 适用边界 |
| --- | --- | --- |
| 前向/反向激活峰值 | `--use_gradient_checkpointing`，必要时 checkpoint offload | 两种训练模式均可；增加累积不会降低当前 microbatch=1 的单样本激活 |
| LoRA 冻结权重常驻过大 | 可微量化、冻结权重 FP8、单卡 model CPU offload | 不是全量低比特参数更新；量化不消除激活和加载峰值 |
| 全量参数、梯度或 optimizer 状态过大 | 合适的 ZeRO stage/offload；单卡完整 CPU offload | model offload 单独开启时可训练参数和 optimizer 仍留 GPU |
| 冻结编码反复占用显存/时间 | 支持的 `sft:data_process → sft:train` | 缓存需有效、磁盘足够；必须计入预处理成本 |
| 多卡提速 | DDP；容量不足再评估 ZeRO | DDP 每卡复制模型；量化、ZeRO、hook offload 的组合需分别验证 |

保持用户训练目标，先排除与设备、数据规格或训练语义不兼容的方案，再按组件和**加载、offload 初始化、编码、前向/反向、首次 optimizer 更新、保存**各阶段判断 CPU/GPU 容量；各项的峰值未必同时发生，不机械叠加静态数字。优先模型专属已验证配方，选择能满足约束且端到端成本合理的最小有效优化组合：原生 LoRA + checkpointing 能运行时，不因 ZeRO-3 理论显存更低就自动引入。未知配置用覆盖真实最大输入、更新和保存的有限试跑验证。不默认安装所有 extras、升级 CUDA/PyTorch 或叠加全部开关；仍不可行时说明限制和可选调整。

自动配置按显存参考中的[选型与试跑闭环](references/memory.md#自动配置的选型与试跑闭环)执行：从一个官方基线出发，只对已识别瓶颈提出必要替代；比较相同工作负载的更新耗时、阶段峰值与一次性成本，选定一份可执行配置并简述证据。只准备时交付待验证配置，不为测量擅自启动真实训练。

## 5. 生成可复现的全量或 LoRA 命令

复制官方配方并在用户任务目录最小修改为 `train.sh`；保留官方 `train.py`，不另造通用训练器，不修改上游示例。移除演示数据下载，替换为真实数据/权重/输出路径，保留变体所需 tokenizer/processor 和条件参数。在框架根目录执行，或在脚本中显式切换到该根目录；launch 参数置于训练脚本路径前，训练参数置于其后。

- **全量**：使用目标配方的 `--trainable_models`，例如 `dit`；核对实际解冻组件，不意外设置 LoRA。低比特/FP8 存储用于不直接更新的组件，需全量更新的目标保留可训练浮点权重；`dequant_once` 不提供持续的权重压缩收益。
- **LoRA**：明确 `--lora_base_model`、模型专属 `--lora_target_modules`、`--lora_rank`；底模冻结而 A/B 分支可训练。不要从其他模型复制层名。通用注入 alpha 默认等于 rank，未提供通用 `--lora_alpha`；空目标字符串可触发源码的自动检测，但不把默认层名当成跨模型通用值。至少验证有预期参数可训练且实际注入命中。
- `--model_paths` 是 JSON；一个组件的分片须放入同一个内层数组。`--model_id_with_origin_paths` 是逗号分隔的 `repo:pattern`。本地/离线运行还需正确的 tokenizer/processor 路径；需要禁用联网时用 `DIFFSYNTH_SKIP_DOWNLOAD=True` 并预先验收本地文件。
- 两种模型来源会**追加合并**，本地路径不会覆盖模型 ID。将官方配方的某组件替换为本地权重时删除该组件原有 ID 条目；只有加载互补组件时才有意同时使用两种来源，并核对实际组件、分片与 revision，避免重复加载或选错底模。
- **本地分片限制**：当前 `--fp8_models` / `--offload_models` 仅用逗号切分后做直接成员比较，不能匹配 `--model_paths` 的嵌套列表；写入列表文本也无效。`--quant_options` 另有列表规范化，不能据此推断 FP8/Disk 也支持。核对实际 `ModelConfig.vram_config()`/`clear_parameters` 和加载行为；受限时采用兼容的 `repo:pattern` 配方（离线时仍验收本地缓存），或明确入口限制，不静默以普通精度继续。细节见显存参考。
- 标准 runner 的 DataLoader **microbatch=1**，没有通用 `--batch_size` 或 `--max_train_steps`。通常有效 batch 约为 `world_size × gradient_accumulation_steps`，结合 `N × dataset_repeat`、epoch、分片补齐和尾部累积估算更新数；每进程 batch 数与更新数分开记录。
- **推荐并显式写入 `--dataset_repeat`**：用户指定值优先；未指定时，根据真实有效样本数、数据多样性、训练方式、epoch、有效 batch、目标更新数、实测速度与预算，推荐并使用一个合适的正整数，简述依据。官方 shell 与 parser 默认值仅作参考，不能直接当作本次任务的推荐值。忽略补齐/尾部时，总更新数约为 `epoch × N × repeat / (world_size × gradient_accumulation_steps)`；可据目标更新数反推 repeat，再按实际 DataLoader 核算。小数据集可适度重复以获得足够更新，但重复不增加数据多样性，还需考虑过拟合；大数据集不应机械放大重复次数。缺少收敛或速度证据时给出合理初值和待验证依据，不声称最优，也不为取得精确值擅自增加训练轮次。用户只指定“1 epoch”并未指定 repeat；不因此自动设为 1，也不因是测试就自动缩短正式训练。记录有效样本数、选值理由、逻辑长度、补齐及实际更新次数。
- 优化器默认 AdamW；当前 `LambdaLR` 的倍率是 `min(1.0, (s + 1) / max(1, warm_up_steps))`。`--warm_up_steps` 默认 5，设 0 关闭；普通路径初始 lr 为配置值的 1/5，之后随成功更新推进，达到配置值后保持。runner 仅在 `sync_gradients` 且 optimizer 未跳步时推进 scheduler，不按每个 microstep 推进；分布式/DeepSpeed 下也核对真实更新与 lr。`--customized_optimizer` 接受完整类名，与底模量化不同；超参参考同模式配方并结合预算选择。
- `ModelLogger --save_steps` 按数据迭代计数，含梯度累积中的 microsteps；它可能在一次累积尚未更新时触发。`step-N` 的 N 不是已完成更新数，通用导出也不保存未完成的累积梯度。按实际同步边界选择并核验间隔，不能仅据文件名恢复中断位置；留空时通常每 epoch 保存。核对 `--remove_prefix_in_ckpt`；多组件权重需分别确认前缀和重新加载方法。
- 用作业自己的 Accelerate YAML/CLI 明确进程数、精度和可见设备，核对其真实 distributed type；`--num_processes 1` 不能保证消除已有 YAML 的 DeepSpeed 配置。不要改用户全局配置。保存实际 argv、源码版本及本地改动、数据/模型 revision 和框架尝试输出的 `training_args.json`。

## 6. 验证链路，再运行用户要求的训练

先做有界 smoke run：使用真实小样本 metadata、独立输出目录，通常设 `dataset_repeat=1`、`num_epochs=1`、`dataset_num_workers=0`，保留最终输入规格和可训练范围，并覆盖最大样本及关键条件。这是链路验收配置，不是正式训练默认值；单独保留 smoke 与正式配置/产物，正式启动前恢复并核对已确定的 repeat、epoch、workers 和更新预算。检查加载/量化/LoRA 命中、可训练参数、有限 loss/梯度、**真实优化器更新及非零参数变化**；保存 checkpoint 并核对 key/shape。量化和 offload 配置还需观察 CPU/GPU 峰值、传输与单步耗时，至少覆盖首次 optimizer 状态分配和一次保存。

按真实进程分片、累积步数和尾部同步行为选择足够的小样本，使 smoke 至少覆盖一次实际 optimizer update；不要只按原始样本条数推断，或为凑更新静默更改有效 batch/输入规格。若还需比较稳定速度，测量窗口应覆盖 warmup/首次编译后的代表性更新。workers=0 只验证单进程数据读取，正式启动前仍需按目标 worker 数核对 RAM、共享内存和吞吐；复核当时的 GPU/RAM 余量。

通过后在用户预算内执行目标训练，记录数据迭代与更新次数、运行时间、loss/学习率、GPU/RAM 峰值、checkpoint 与日志。OOM 按显存参考定位后再改配置；保留原作业产物，仅处理本作业进程，不循环盲目重试。

未结束时如实报告训练中/中断、进程标识和日志，不声称已完成。smoke 成功也不证明收敛或目标能力已学到。

## 7. 拆分预处理与权重继续训练

Split Training 是标准 SFT 的执行方式，只有目标 `task_to_loss`/launcher 支持时才用。第一阶段 `--task sft:data_process`、repeat=1，缓存到 `output/<process_index>/<data_id>.pth`；第二阶段 `--task sft:train`，base path 指向缓存目录，**移除 metadata 参数，显式使用正式训练的 `--dataset_repeat` 与 `--num_epochs`**，不要将预处理的 repeat=1 沿用到正式训练。缓存 Dataset 同样按缓存条数乘 repeat 计算长度；验收缓存重复/补齐，避免无意放大训练量。两阶段的可训练/LoRA 范围、输入条件、模型和数据规格要匹配；`--offload_models` 是拆分场景的 disk/清参数配置，不是逐层训练 CPU offload 的同义词。不要移除仍会被间接调用的组件。

仅加载本次生成或已明确验证可信的缓存目录。`UnifiedDataset` 会递归收集所有 `.pth`，`LoadTorchPickle` 使用 `torch.load(..., weights_only=False)`；将缓存与原始数据、模型 checkpoint、其他实验输出分开，不混入第三方或来源不明的 `.pth`。检查数量、分片补齐/重复和真实读取。caption、尺寸/裁剪、视频采样、条件、冻结权重、tokenizer、代码或可训练范围变化时重新判断缓存有效性。缓存会固定该次随机预处理，不能等同每 epoch 重新增强。

**同时核对缓存内的执行选项**：通用 runner 对缓存调用 `model({}, inputs=data)`；MOSS 等入口在 `inputs` 已提供时跳过 `get_pipeline_inputs()`，因此缓存中的 `use_gradient_checkpointing` / `use_gradient_checkpointing_offload` 可能直接进入 model_fn，第二阶段同名 CLI 不一定覆盖。例如缓存保存 offload=False 时，仅在训练命令加 `--use_gradient_checkpointing_offload` 仍可能不生效。抽样检查可信缓存的 shared/正负输入及实际参数优先级；需要更改时，按目标入口以所需 flags 重新生成到新缓存目录，或在任务级入口显式覆盖这些已确认的执行字段并记录来源，不改写原缓存或其他条件。验收真实 checkpoint helper 的分支；DeepSpeed checkpoint 的优先级仍按显存参考核对，不能只凭缓存或 CLI 的 True 声称优化生效。

继续训练时区分：`--lora_checkpoint` 在注入后加载 LoRA；`--resume_from_checkpoint` 在 LoRA 注入前加载模型 state_dict，通用用法面向兼容的单组件全量权重；`--preset_lora_path` / `--preset_lora_model` 融合预置适配器，也不是恢复 optimizer。核对 rank、目标层、前缀、实际 key 匹配，不因 `strict=False` 未报错就认为成功。通用权重导出不含 optimizer/scheduler/RNG/采样进度或累积中的梯度；这些都是**权重热启动的新 run**，重新设置并记录学习率、warmup 与更新计数，不能称为完整断点恢复，也不要覆盖原 run。

## 8. 新进程验证与交付

从同模型/变体的官方验证示例生成 `validate.py`，移除与任务无关的独立演示数据下载及固定演示输入，替换为训练前确定的独立验证数据，保留对应 prompt、图像/视频/音频条件与生成规格。用真实输出 checkpoint 在新 Python 进程加载，确认权重前缀、目标组件、key/shape 和 LoRA 补丁数。全量导出通常只含可训练参数，不是完整 Pipeline；LoRA 配置沿用官方 `pipe.load_lora(...)`。量化底模需重建训练时相同的基座与量化设置，按量化参考处理热加载。

固定真实验证输入、seed、尺寸、采样步数与底模 revision，生成可比较的训练前后产物并打开验收；区分“权重可加载”“媒体正确生成”和“效果达到目标”。不能仅凭官方演示图生成成功认定用户目标已达到。无法验证时明确未验证，不以 loss 代替质量。

交付实际数据位置/有效样本数、官方配方来源、`train.sh`、验证命令/脚本、采用的显存与多卡配置、`dataset_repeat`/epoch/逻辑长度/实际更新次数、运行状态/测量、checkpoint、加载结果与对照产物路径。区分 smoke 与正式训练记录。只准备时说明未执行；不额外生成未采用的配置矩阵或另一套训练框架。
