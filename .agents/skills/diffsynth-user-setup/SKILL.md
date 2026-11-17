---
name: diffsynth-user-setup
description: >
  Set up, repair or verify a DiffSynth-Studio user environment,
  including Python dependencies, GPU/device compatibility,
  model downloads, local storage and offline loading.
  Use for installation and environment failures, not for
  integrating a new model architecture into DiffSynth-Studio.
  环境安装、依赖与设备诊断、模型下载和离线配置。
---

# DiffSynth-Studio 安装、设备与离线环境

先识别正在运行的环境，再安装目标任务缺失的依赖。以用户现有项目和设备为起点，不要求固定 Conda 名称，也不为一个 import 错误重建整套环境。

## 1. 建立环境事实

先核对目标源码的 `pyproject.toml`。当前框架明确要求 **Python ≥3.10.1**（`requires-python`）；Python 3.9 和 3.10.0 不满足安装要求。不要把辅助工具可执行的版本当成框架支持版本，也不要因某个可选依赖而提高整个框架的最低版本。目标依赖可能有更严格要求，按任务进一步核对。

在将用于运行任务、满足上述要求的 Python 中检查解释器、包位置、torch 构建、设备、源码版本和可用存储。从本 skill 目录执行：

```bash
python scripts/probe_environment.py --studio /path/to/DiffSynth-Studio --model-dir /actual/model-storage --output-dir /actual/task-output --output /path/to/task-output/environment.json
```

工具只依赖 Python 标准库，默认只读元数据，不 import torch 或 diffsynth。需要实际验证导入时加 `--import-diffsynth`，需要设备查询时加 `--torch`；两者各在独立子进程中执行，`--timeout 30` 是每个子进程的超时。工具不主动执行张量操作或加载模型，但实际 import 会执行包的初始化代码，可能查询设备、导入扩展或获取远程内核；离线任务先按第4节检查相关环境变量。

读取字段时区分以下证据，不能只看命令退出码或 errors 是否为空：

| 字段 | 可以说明什么 |
| --- | --- |
| `python_requirement` | 从目标源码读取 requires-python；无源码时尝试发行元数据。简单的 `>=版本` 判断为 pass/fail，复杂约束或声明缺失为 not_assessed，交给 pip 核对；不是全部依赖的兼容性结论 |
| `metadata_discovery` | distribution 版本与包 spec 是否发现；两者可缺失或不一致，不代表实际导入成功 |
| `metadata_discovery` 中的来源比较 | spec 路径是否匹配指定源码/可识别的安装来源；未知不作一致结论 |
| `import_test.status` | 独立子进程导入的 pass、fail、timeout、failed 或 not_tested；同时检查实际 origin |
| `import_test` 中的来源比较 | 实际 origin 与指定源码匹配且导入成功时，`requested_source_import_verified` 为 true；安装来源不一致单独记入 warnings，不否定实际导入来源，仍不表示 Pipeline 可用 |
| `torch_probe_requested` / `torch_import_ok` / `device_query_ok` | 请求探测、成功导入和完成设备查询分别记录；未探测的成功字段为 null |
| `device_probe_status` | 区分 not_requested、import_failed、no_accelerator、partial、ok、timeout 或 failed，不能仅靠 devices 是否为空判断 |
| `validation` / `target_readiness` | 张量、后端算子和 Pipeline 默认 not_tested；诊断 collected 不表示目标已满足，目标就绪仍为 not_assessed |

指定 `--studio` 时先读 `source_validation.status`：ok 表示源码特征匹配；partial 表示可识别但不完整；source_invalid 表示路径不存在/不是目录；invalid_source_root 表示目录存在但特征不足。检查结合包入口、Pipeline 目录及 diffsynth 项目元数据，允许源码导出缺少构建文件，不把任意 Git 仓库当 Studio。无效或不完整时保留其他诊断，整体状态为 partial；先纠正路径再做源码安装。`scope` 仅表示请求范围，缺少该参数时为 python_environment_only。

再核对 `sources.studio.git` 的根目录归属、commit、dirty 与采集时点；父仓库 commit 不作为 Studio 版本。Git 状态在结果写入前采集，报告放在源码目录外。

存储分 cwd、model、output 三个角色报告，模型和产物路径必须在展开 `~` 后为绝对路径。若任务使用 `./models`，先按**实际任务 cwd** 解析，再传入绝对的 `--model-dir`/`--output-dir`；工具不按 Skill 目录猜测。未指定模型目录时仅尝试 `DIFFSYNTH_MODEL_BASE_PATH`，相对值会记 invalid_path，未提供则记 not_specified；参数只用于诊断，不改变框架配置。目标尚不存在时检查最近的现有祖先并记录 `probed_path`，不创建目录。free_bytes 是文件系统快照，未检查用户配额，不能视为下载额度保证。

RAM 分开记录 `host_visible`、可读取的 cgroup v1/v2 当前组及父级约束，以及近似可用值。`limit_status=unknown`（包括 not_detected）时近似值为 null；有限额度取 host 与各层 headroom 的较小值，成功读取且未观察到有限额度时才允许基于 host 估计。它不涵盖不可见祖先、其他调度约束或全部可回收内存，也不是预留额度。CUDA free/total 同样只是快照；记录的 CUDA_VISIBLE_DEVICES 等仅为设备选择白名单，cuda 索引是当前进程可见编号。

仓库根目录、当前 cwd、Python 导入位置分别核对。运行 `python -m pip` 使安装目标与解释器一致；实际导入成功后仍须比较 origin 与用户指定源码，处理路径遮蔽或多版本问题。包元数据存在而 spec 缺失、spec 存在而无发行版本、两者都有但导入失败，是不同的诊断分支。

导入/设备子进程的错误现场见 `import_test.worker_diagnostics` 或 `torch_worker_diagnostics`：保留阶段、退出码、POSIX 信号、超时标志及可取得的 stdout/stderr 末尾摘要，每段最多4096字符。摘要和错误文本按常见凭据格式及当前环境中的已知凭据值尽力脱敏，不自动另存完整原始日志。`metadata_discovery.import_verified` 只表示某个包成功导入，指定源码是否匹配以实际导入比较为准。

## 2. 按任务补齐依赖

读取当前存在的 `pyproject.toml`、`docs/zh/Pipeline_Usage/Setup.md` 和 `docs/zh/Pipeline_Usage/GPU_support.md`，目录调整时查实际版本的对应文档。以依赖声明及目标实现为准，不能把文档概述当成完整安装清单。先定位首个阻塞项，只安装当前任务需要的依赖。

当前基础声明为 `torch>=2.0.0`，并包含 torchvision、transformers、imageio[ffmpeg]、safetensors、einops、modelscope、ftfy、pandas、accelerate、peft 和 **`entropack>=0.1.7`**；构建使用 setuptools≥77 和 wheel。直接依赖的下界不等于完整解析环境的下界：还需检查实际 EntroPack 版本及其传递依赖，不能据 torch 声明保证最低版本可安装或可运行。检查目标组件实际访问的 API：例如 Flex Attention 明确要求 torch≥2.5，部分模型使用 float8 或 `torch.compiler.nested_compile_region`。Transformers 也应核对目标模型需要的类/API；Nexus-Gen 专项依赖固定4.49.0，不能为了其他模型盲目升级，配置文件中的 transformers_version 不等于安装约束。

| 任务 | 当前安装边界 |
| --- | --- |
| 音频/含音频组件的 Pipeline | `[audio]` 含 av、torchaudio、torchcodec、librosa；当前音频读写默认使用 torchcodec，不能只检查 torchaudio |
| 训练 | accelerate、peft 已在基础依赖；`[training]` 只增加 deepspeed，并非所有 LoRA 训练都需要它；日志后端按需选 `[logger]` |
| 量化 | `[quant]` 增加 bitsandbytes、comfy-kitchen、torchao≥0.16；EntroPack 已在基础依赖，未指定 cuda12/cuda13 extra；具体后端仍有设备/版本要求，DeepGEMM 不在其中 |
| Ascend NPU | x86 用 `[npu]`，ARM/aarch64 用 `[npu_aarch64]`；两者固定 torch2.7.1、torchvision0.22.1、torch-npu2.7.1，x86 的 torch/torchvision 为 +cpu 构建 |
| 模型专属功能 | 按实际模型选择 controlnet、infiniteyou、ses、nexusgen、yue2 等 extra；不要由模型名称猜依赖 |
| `[all]` | 当前仅汇总 audio、quant、training、logger 的依赖，不含 NPU 或上述模型专属依赖，也不含全部第三方 attention/USP 库 |

EntroPack 当前框架文档列出 CUDA 版 PyTorch≥2.10 的后端要求，但基础声明没有选择 CUDA extra；核对实际发行包依赖和目标算子，不把文档的后端要求等同于未经检查的 pip 解析结果。NPU extra 固定 torch2.7.1，**即使不选 quant/all，也必须检查基础 EntroPack 的依赖与设备兼容性**；不能因去掉量化 extra 就认定冲突消失。需要 EntroPack CUDA 内核时，按实际环境选择 `entropack[cuda12]` 或 `entropack[cuda13]`，当前 `[quant]` 本身不替用户选择。comfy-kitchen、DeepGEMM 等另有 CUDA/架构要求，读取 Quantization 文档和目标 backend 后再决定。发现当前声明无法满足目标设备时说明具体阻塞，不盲目升级 torch，也不声称通过 `--no-deps` 就完成了兼容安装。

已有正确安装则跳过。需要源码安装时，在用户选定的 Studio 根目录执行 `python -m pip install -e .`；extras 名称从当前 pyproject 取，不默认 `all`。先查 torch 与驱动/硬件兼容，再选扩展 wheel，避免为一个后端盲目升级整个环境。

安装前保留相关包版本，并核对依赖变更计划；当前 pip 支持时可用 `--dry-run` 检查解析结果。源码安装同样可能改变已有依赖，不盲目升级 PyTorch。安装后运行 `python -m pip check` 和目标包/扩展的必要导入验证，区分已有冲突与新增冲突，只修本任务相关问题。pip check 通过不代表二进制算子或 Pipeline 已验证。

只要求诊断或命令时保存具体安装步骤；已要求搭建环境时完成必要安装及验证，不逐包重复询问。安装失败保留实际错误、命令和受影响版本，先解决首个原因。

## 3. 核对设备能力

区分“驱动识别设备”“torch 能创建目标设备张量”“目标扩展支持 dtype/算子”“Pipeline 可运行”四层。CUDA/ROCm、MPS、NPU 不共享一套无条件可用的参数。

CUDA/ROCm 先核对 torch 构建 runtime、驱动和 wheel；`torch_build_cuda_version`/`torch_build_hip_version` 是 PyTorch 构建版本，不能当作当前驱动版本。nvidia-smi 正常不等于当前 Python 可用；MPS 的 is_built、available 和实际计算验证也不同。单卡查询失败时保留其他设备结果，再检查该卡的错误阶段。

通用工具不探测 NPU；先配置 CANN，再按上述架构选择 extra。x86 源码安装为 `python -m pip install -e '.[npu]' --extra-index-url https://download.pytorch.org/whl/cpu`，aarch64 为 `python -m pip install -e '.[npu_aarch64]'`。继续按当前 GPU_support/Setup 的 NPU 专节及厂商 torch_npu 流程验证；部分训练还需 initialize_model_on_cpu 或 enable_npu_patch，使用对应 NPU 样例。没有加速设备不自动排除 CPU 路线，但须按任务核实兼容性和实际速度。

对于特定量化或 attention 后端，检查实际依赖、设备架构、dtype 和小规模操作。注册表能列出名字不表示可用；不能全局替换 cuda 字符串来承诺另一设备支持。

AMD 选择 ROCm 版 torch，Apple Silicon 选 mps/cpu；迁移示例时同时检查 Pipeline.device、ModelConfig 的各阶段设备、随机数设备、显存查询和实际算子，不能保留 `torch.cuda.mem_get_info` 就认定已完成迁移。普通任务先用 PyTorch 原生 attention 验证，可在该任务进程启动前设置 `DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch`，不必安装所有加速扩展。启用 USP 才检查 yunchang/xfuser；音视频合并若调用外部 ffmpeg，还要检查 PATH 中的命令，imageio 的后端可用不能替代它。

## 4. 选择模型来源与存储

先找已有文件和缓存，再决定下载。模型组件、tokenizer/processor、VAE、索引和 sidecar 分别核对；目录存在或 glob 无匹配不算文件齐全。

建议启动前统一设置环境变量。attention 实现在导入时选择，ModelConfig 的下载源、skip_download 和存储根目录则在调用时读取。检查最终解析路径，尤其 `DIFFSYNTH_MODEL_BASE_PATH` 的覆盖行为。用户已要求实际运行时补齐必要权重；只要脚本时不顺带下载大模型。

离线任务逐个列出本地组件和辅助文件，使用真实 `ModelConfig(path=...)`，一个组件的分片保持同一组。第三方 tokenizer、processor 和指标模型也要单独检查本地加载，不把 skip_download 当成全进程禁网。

还要检查示例自身的 `snapshot_download`、`dataset_snapshot_download`、Shell 中的 `modelscope download` 和辅助权重获取代码，改用本地输入/数据；这些不受 ModelConfig 开关控制。若设置 `DIFFSYNTH_FLASH_ATTN_KERNEL_REPO_ID`，框架导入时会尝试通过 kernels.get_kernel 获取内核，同时检查对应 VERSION 和缓存；严格离线时预置内核或在该任务进程中取消不需要的远程内核配置。仅指定 attention=torch 或 SKIP_DOWNLOAD=True 不能阻止这段导入逻辑。

## 5. 分层验证并留下可复用入口

依次检查：路径/版本 → 包及目标 Pipeline、实际选中模型组件的导入 → 目标设备张量操作 → 所需后端算子 → 有权重时构建目标 Pipeline，运行最小有效输入。模型存在动态导入，顶层或 Pipeline 导入成功不代表组件依赖齐全。各层分别记录 pass、fail 或 not_tested；超时单独保留原因。每层失败只修相关问题，再复测原场景；设备查询成功不能代替张量、算子或模型运行。训练验证先调整 launcher、设备数和 accelerate 配置；某些官方示例预设 DeepSpeed/8进程，不能原样当作单卡最小验证。

按目标判断就绪：CPU 环境可能已完成自身设备验证，但仍不满足要求 CUDA 的任务。交付实际环境/安装命令、最终模型路径、各层验证结果与剩余条件，再接续原推理或训练任务；没有权重或设备时交付可运行准备，不虚构模型已跑通。不强制额外固定报告目录。

## 下载来源、路径和离线加载

权威实现：`diffsynth/core/loader/config.py` 的 ModelConfig。环境变量说明见 `docs/zh/Pipeline_Usage/Environment_Variables.md`，文档与实现有差异时检查代码。

- 远端配置为 model_id + origin_file_pattern；仅取需要的组件，包含 tokenizer/processor/VAE 等。显式 download_source 优先环境变量 DIFFSYNTH_DOWNLOAD_SOURCE，未设置时默认 modelscope，也支持 huggingface。记录仓库、文件清单及可取得的 revision/内容标识；当前 ModelConfig 没有 revision 参数，需锁定版本时使用 SDK 独立下载，再传本地 path。
- 本地配置用 `ModelConfig(path="/actual/component.safetensors")`；一个组件的多个分片用同一配置的 path 列表。不同组件不能任意合成一组。
- 当前 reset_local_model_path 中，环境变量 DIFFSYNTH_MODEL_BASE_PATH **覆盖**显式 local_model_path；两者未设置时为任务 cwd 下的 ./models。远端文件落在 `<base>/<model_id>/`，但已经设置的 path 不受这个根目录覆盖，也不触发该组件下载。记录最终解析路径，而非复述文档优先级。
- 只有 ModelConfig 未显式设置 skip_download 时，DIFFSYNTH_SKIP_DOWNLOAD=True 才决定该组件跳过自动下载；显式 skip_download=False 会覆盖环境变量，控制台提示不能代替实际配置检查。它不提供模型文件，也不是全进程网络隔离。第三方 tokenizer/processor 和评分器需要独立检查。
- 最可靠的离线方案：每个组件使用已存在的 path，并核验所需辅助文件、索引、分片和量化 sidecar。不能仅以目录存在或 glob 命中零文件判为齐全。
- 已有缓存也不保证零网络：需要下载的配置仍会调用 SDK snapshot_download(local_files_only=False)，已有文件仅加入忽略列表。离线须检查实际调用链，不能只看“未下载新文件”。

示意代码（替换实际路径并按目标 Pipeline 组合）：
```python
import os
os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
from diffsynth import ModelConfig
component = ModelConfig(path=["/models/part-00001.safetensors", "/models/part-00002.safetensors"], skip_download=True)
```

这是组件配置，不是完整推理脚本。检查目标 Pipeline.from_pretrained 所需组件及 processor/tokenizer 参数后再运行。敏感下载凭据沿用用户已有登录环境，不写入脚本或报告。

## 随附检查工具

[probe_environment.py](scripts/probe_environment.py)。上文命令从本 skill 目录执行；在用户项目中调用时改用脚本的实际路径。工具输出与日志保存到用户任务目录。
