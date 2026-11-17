# 各 Skill 具体工作流程

> 本文档列出 diffsynth-integrator 全流程中 11 个 Skill 的详细工作流程，具体到三级子步骤。

---

## 流程概览

| Step | Skill | 说明 | 人工确认点 |
|------|-------|------|-----------|
| 0 | `diffsynth-analyze-target` | 分析目标库，生成接入蓝图 + 代码阅读报告 | ✅ 确认蓝图 |
| 1 | `diffsynth-environment` | 双向环境验证：DiffSynth OK → Target OK → DiffSynth still OK | — |
| 2 | `diffsynth-model-code` | 模型代码接入 + 一致性测试 | — |
| 3 | `diffsynth-pipeline` | Pipeline 接入（每次调用处理一个功能） | — |
| 3a | `diffsynth-pipeline-lowvram` | 低显存推理脚本 + VRAM 管理配置 | — |
| 3b | `diffsynth-pipeline-training` | 训练模块接入 + 训练验证 | — |
| 4 | `diffsynth-style` | 代码风格化：统一 pipeline/模型/脚本的代码风格 | — |
| 5 | `diffsynth-testing` | 全量推理脚本测试 + 结果保存 | ✅ 确认输出质量 |
| 6 | `diffsynth-docs` | 中英文文档生成 + README 更新 | — |
| 7 | `diffsynth-pr` | 生成 PR 描述 Markdown 文件（不执行 git 操作） | ✅ Review PR 描述 |
| 8 | `diffsynth-pr-review` | 处理 PR Review 反馈，按评论针对性修复 | — |

---

## Skill 1: diffsynth-analyze-target

**目标**：分析目标模型库代码结构，生成整合蓝图报告。

### Step 0: 确定目标库路径
- 0.1 从用户输入中提取模型名称
- 0.2 在 `packages/` 下查找对应目录
- 0.3 如果找到多个匹配，向用户确认
- 0.4 记录 `target_path` 到上下文

### Step 1: 初始化执行日志目录
- 1.1 创建 `execution-logs/{timestamp}_analyze/` 目录结构
- 1.2 创建 `manifest.json` 和 `latest` 软链接

### Step 2: 制定执行计划
- 2.1 输出执行计划到 `plans/analyze-plan.md`
- 2.2 向用户展示分析范围、输出路径规划
- 2.3 等待用户确认后继续

### Step 3: 分析目标库架构
- 3.1 扫描目标库目录结构
- 3.2 识别模型组件（DiT、VAE、TextEncoder 等）
- 3.3 识别推理流程（推理脚本路径、输入输出）
- 3.4 分析依赖关系（diffusers、transformers 等外部依赖）
- 3.5 记录训练相关信息（训练脚本、数据集格式）

### Step 4: 生成整合蓝图
- 4.1 输出基本信息表（模型名称、Conda 环境名称、目标库路径）
- 4.2 输出模型组件清单表（组件名、类名、`__init__` 签名、`forward` 签名）
- 4.3 输出 Pipeline 功能规划表（功能名、输入输出、依赖组件、推理脚本路径）
- 4.4 输出依赖差异分析表（必须安装、可选安装、不需要安装的包）
- 4.5 输出模型下载命令清单
- 4.6 输出接入类型建议（new_series / version_upgrade）
- 4.7 保存到 `integration-blueprints/{model-name}-blueprint.md`

### Step 5: 生成代码阅读报告
- 5.1 对每个模型组件生成详细代码阅读报告
- 5.2 保存到 `code-reports/code-report.md`

### Step 6: 生成可行性分析
- 6.1 分析接入可行性
- 6.2 识别潜在风险和难点
- 6.3 保存到 `integration-blueprints/feasibility-analysis.md`

### Step 7: 最终验证
- 7.1 检查蓝图报告文件完整性
- 7.2 检查代码阅读报告文件完整性
- 7.3 检查执行日志和脚本完整性

**输出文件**：
- `integration-blueprints/{model-name}-blueprint.md`
- `code-reports/code-report.md`
- `integration-blueprints/feasibility-analysis.md`

---

## Skill 2: diffsynth-environment

**目标**：在同一个 conda 环境中，确保 DiffSynth-Studio 和目标库都能正常运行。

### Step 0: 读取蓝图信息
- 0.1 从蓝图报告读取 Conda 环境名称
- 0.2 从蓝图报告读取目标库路径
- 0.3 从蓝图报告读取依赖差异分析表

### Step 1: 初始化执行日志目录
- 1.1 创建 `execution-logs/{timestamp}_environment/` 目录结构
- 1.2 确保 DiffSynth-Studio 已存在（如不存在则 clone）
- 1.3 创建 `manifest.json` 和 `latest` 软链接

### Step 2: 制定执行计划
- 2.1 输出执行计划到 `plans/environment-plan.md`
- 2.2 向用户展示依赖差异、验证方式、模型下载清单
- 2.3 等待用户确认后继续

### Step 3: 确认 DiffSynth 验证方式
- 3.1 环境阶段只需 import 级别检查
- 3.2 验证能导入具体 Pipeline：`from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig`

### Step 4: 推荐模型下载（等待用户确认）
- 4.1 从蓝图报告读取模型下载清单
- 4.2 向用户展示 modelscope download 命令
- 4.3 等待用户确认下载完成

### Step 5: 创建模型软链接
- 5.1 为目标库创建 `models/` 真实目录
- 5.2 为 DiffSynth 创建 `models/` 真实目录
- 5.3 为每个模型在两个 locations 分别创建软链接 → ModelScope 缓存

### Step 6: 创建 conda 环境（通过 clone）
- 6.1 从 config.yaml 读取 clone 源环境名称
- 6.2 执行 `conda create -n {env_name} --clone {source} -y`
- 6.3 激活新环境
- 6.4 验证 PyTorch 和 CUDA 可用性

### Step 7: 安装 DiffSynth 并验证
- 7.1 在 DiffSynth-Studio 根目录执行 `pip install -e .`
- 7.2 运行 import 验证脚本
- 7.3 保存验证日志

### Step 8: 安装目标库依赖并验证
- 8.1 从蓝图报告读取额外依赖清单
- 8.2 逐步安装目标库额外依赖
- 8.3 验证目标库推理脚本能正常 import
- 8.4 验证 DiffSynth import 仍然正常（双向验证）

### Step 9: 最终验证
- 9.1 检查 DiffSynth import 正常
- 9.2 检查目标库 import 正常
- 9.3 检查报告文件完整性

**输出文件**：
- `plans/environment-plan.md`
- `skill_work_report/environment-report.md`
- `user_report/environment-report.md`

---

## Skill 3: diffsynth-model-code

**目标**：将目标模型的组件（DiT、VAE、TextEncoder 等）接入 DiffSynth-Studio。

### Step 0: 读取蓝图信息
- 0.1 从蓝图报告读取 Conda 环境名称
- 0.2 从蓝图报告读取目标库路径
- 0.3 从蓝图报告读取模型组件清单表
- 0.4 从蓝图报告读取外部依赖分析
- 0.5 从蓝图报告读取接入类型

### Step 1: 初始化执行日志目录
- 1.1 确保 DiffSynth-Studio 存在
- 1.2 创建 `execution-logs/{timestamp}_model_code/` 目录结构
- 1.3 创建 `manifest.json` 和 `latest` 软链接

### Step 2: 制定执行计划
- 2.1 输出执行计划到 `plans/model-code-plan.md`
- 2.2 向用户展示组件接入方案、注册方案、测试方案
- 2.3 等待用户确认后继续

### Step 3: 确定接入方式
- 3.1 判断接入类型（new_series / version_upgrade）
- 3.2 对每个组件制定接入方案：
  - 3.2.1 确定组件代码来源（目标库自有 / diffusers / transformers）
  - 3.2.2 确定拷贝路径
  - 3.2.3 确定重构点（依赖替换、attention/gradient checkpointing）
  - 3.2.4 确定 state_dict key 一致性策略

### Step 4: 复制并重构模型代码
- 4.1 从目标库复制模型代码文件
- 4.2 替换无法 import 的模块为 DiffSynth 核心模块或内联实现
- 4.3 替换 Attention 为 `attention_forward`
- 4.4 替换 Gradient Checkpointing 为 `gradient_checkpoint_forward`
- 4.5 保持 state_dict key 与原始 checkpoint 一致
- 4.6 不重构结构、不统一风格、不重命名变量

### Step 5: 初始化参数验证
- 5.1 在目标库中初始化模型，打印所有参数值
- 5.2 在 DiffSynth 模型中填入默认参数
- 5.3 对比验证，确认参数一致
- 5.4 创建验证脚本并保存日志

### Step 6: 创建 state_dict_converter
- 6.1 分析原始 checkpoint key 与 DiffSynth 模型 key 的差异
- 6.2 创建 key 映射转换器
- 6.3 保存到 `utils/state_dict_converters/{series}_{component}.py`

### Step 7: 注册模型到 model_configs.py
- 7.1 计算 model_hash（使用 `hash_model_file`）
- 7.2 在 `model_configs.py` 中注册 model_hash → model_class 映射
- 7.3 配置 extra_kwargs（如有）

### Step 8: 一致性测试
- 8.1 确保 Pipeline 能跑通
- 8.2 运行 E2E 端到端对比测试
- 8.3 验证输出类型、shape、dtype 一致
- 8.4 验证无 NaN/Inf
- 8.5 保存对比结果和执行留痕

### Step 9: 最终验证
- 9.1 检查所有修改文件语法
- 9.2 检查报告文件完整性
- 9.3 检查执行日志和脚本完整性

**输出文件**：
- `diffsynth/models/{series}_{component}.py`
- `utils/state_dict_converters/{series}_{component}.py`
- `diffsynth/configs/model_configs.py`（新增注册项）
- `plans/model-code-plan.md`
- `skill_work_report/model-code-report.md`
- `user_report/model-code-report.md`

---

## Skill 4: diffsynth-pipeline

**目标**：为接入的模型创建推理 Pipeline 和推理示例脚本。每次调用处理一个功能。

### Step 0: 读取蓝图信息
- 0.1 从蓝图报告读取 Conda 环境名称
- 0.2 从蓝图报告读取目标库路径
- 0.3 从蓝图报告读取 Pipeline 功能规划表（本次接入功能）
- 0.4 从蓝图报告读取 Scheduler 信息
- 0.5 从蓝图报告读取接入类型

### Step 1: 初始化执行日志目录
- 1.1 创建 `execution-logs/{timestamp}_pipeline_{feature}/` 目录结构
- 1.2 创建 `manifest.json` 和 `latest` 软链接

### Step 2: 制定执行计划
- 2.1 输出执行计划到 `plans/pipeline-{feature}-plan.md`
- 2.2 向用户展示 Pipeline 架构、Unit 链设计、对比验证方案
- 2.3 等待用户确认后继续

### Step 3: 确定本次要接入的功能
- 3.1 从蓝图报告或用户指定中获取功能信息
- 3.2 确认功能名称、输入输出、特殊处理、依赖组件
- 3.3 确认是否为首次调用（创建新 Pipeline 文件 vs 追加功能）

### Step 4: 创建或更新 Pipeline 文件
- 4.1 分析目标库推理流程 → 输出设计蓝图
  - 4.1.1 逐行拆解目标库推理脚本为四个阶段（条件准备、噪声初始化、去噪循环、解码输出）
  - 4.1.2 保存设计蓝图到 `reports/blueprint-{feature}.md`
  - 4.1.3 等待用户确认设计蓝图
- 4.2 根据设计蓝图编写 Pipeline 初稿
  - 4.2.1 编写 Pipeline 类（`__init__` + `from_pretrained` + `__call__`）
  - 4.2.2 首次调用创建完整结构，后续调用追加功能
- 4.3 Scheduler 接入
  - 4.3.1 判断接入方式（复用已有 template / 新增 set_timesteps 方法）
  - 4.3.2 运行目标库和 DiffSynth 的 Scheduler 对比 timesteps
  - 4.3.3 逐项对比验证，向目标库对齐
- 4.4 设计 Unit 链并逐个编写
  - 4.4.1 ShapeChecker（尺寸校验，必须复用 `check_resize_height_width`）
  - 4.4.2 PromptEmbedder（文本条件编码，必须使用 `seperate_cfg=True`）
  - 4.4.3 NoiseInitializer（初始噪声生成，必须调用 `pipe.generate_noise`）
  - 4.4.4 InputEmbedder（输入模态编码，区分推理/训练模式）
  - 4.4.5 功能专属 Unit（如 EditImageEmbedder、InpaintEmbedder 等）
- 4.5 设计 model_fn
  - 4.5.1 确定参数签名（与 cfg_guided_model_fn 传入参数匹配）
  - 4.5.2 编写 model_fn 函数，放在文件末尾

### Step 5: 创建推理示例脚本
- 5.1 创建标准推理脚本 `examples/{series}/model_inference/{feature}.py`
- 5.2 创建低显存版本 `examples/{series}/model_inference_low_vram/{feature}.py`

### Step 6: 一致性验证
- 6a. 阶段一：确保 Pipeline 能跑通
  - 6a.1 运行 DiffSynth 推理脚本
  - 6a.2 检查退出码，修复错误直到跑通
- 6b. 阶段二：E2E 端到端对比
  - 6b.1 在原库中运行，保存 golden reference
  - 6b.2 在 DiffSynth 中运行 Pipeline
  - 6b.3 验证输出一致性（类型、shape、dtype、无 NaN/Inf）
  - 6b.4 保存执行留痕
- 6c. 根据 E2E 结果处理
  - 6c.1 E2E 通过 → 输出报告 → 用户确认 → 完成
  - 6c.2 E2E 失败 → 修复问题 → 重新运行
- 6d. 用户检查清单
  - 6d.1 创建 `user-checks.md`
  - 6d.2 追加 6 项检查点（CHECK-001 到 CHECK-006）

### Step 7: 最终验证
- 7.1 检查三个报告文件是否存在
- 7.2 检查执行日志目录中的脚本完整性
- 7.3 如有缺失，立即补充

**输出文件**：
- `diffsynth/pipelines/{series}.py`
- `examples/{series}/model_inference/{feature}.py`
- `examples/{series}/model_inference_low_vram/{feature}.py`
- `reports/blueprint-{feature}.md`
- `plans/pipeline-{feature}-plan.md`
- `skill_work_report/pipeline-{feature}-report.md`
- `user_report/pipeline-{feature}-report.md`

---

## Skill 5: diffsynth-pipeline-lowvram

**目标**：为标准 Pipeline 生成低显存推理脚本，注册细粒度显存管理配置。

### Step 0: 读取蓝图信息
- 0.1 从蓝图报告读取 Conda 环境名称
- 0.2 从蓝图报告读取目标库路径
- 0.3 从蓝图报告读取 Pipeline 功能规划表
- 0.4 从蓝图报告读取 Pipeline 类名和模型组件列表
- 0.5 判断模型是否为逐 token 模式（决定是否注册 module_map）
- 0.6 验证前置条件（标准推理脚本存在、Pipeline 支持 vram_limit）

### Step 1: 初始化执行日志目录
- 1.1 创建 `execution-logs/{timestamp}_lowvram/` 目录结构
- 1.2 创建 `manifest.json` 和 `latest` 软链接

### Step 2: 制定执行计划
- 2.1 输出执行计划到 `plans/lowvram-plan.md`
- 2.2 包含脚本规划表（标准脚本 → 低显存脚本映射）
- 2.3 包含组件列表（哪些注册 module_map、哪些跳过）
- 2.4 等待用户确认后继续

### Step 3: 分析标准推理脚本
- 3.1 确认 Pipeline 的 `from_pretrained` 已支持 `vram_limit` 参数
- 3.2 如不支持，在 `from_pretrained` 中添加 `vram_limit` 参数
- 3.3 读取标准推理脚本，提取 ModelConfig 列表、推理参数、输出保存方式

### Step 4: 分析模型组件结构
- 4.1 通过 `print(model)` 或代码分析识别包含参数的 Layer
- 4.2 确定每个 Layer 的包裹器类型（AutoWrappedLinear / AutoWrappedModule）
- 4.3 输出每个组件的 module_map 草案

### Step 5: 注册细粒度显存管理配置
- 5.1 读取现有 `vram_management_module_maps.py`
- 5.2 为每个模型组件注册 module_map
- 5.3 逐 token 模式跳过 module_map 注册

### Step 6: 生成低显存推理脚本
- 6.1 复制标准推理脚本到 `model_inference_low_vram/`
- 6.2 为每个组件 ModelConfig 注入 `**vram_config`
- 6.3 在 `from_pretrained` 中注入 `vram_limit` 参数
- 6.4 非组件配置（tokenizer_config 等）不注入 `**vram_config`

### Step 7: 验证脚本正确性
- 7.1 静态检查：语法正确性、参数完整性
- 7.2 动态运行：确认低显存脚本能正常执行
- 7.3 保存验证日志

### Step 8: 最终验证
- 8.1 检查报告文件完整性
- 8.2 检查低显存脚本和 module_map 注册完整性

**输出文件**：
- `examples/{series}/model_inference_low_vram/{feature}.py`
- `diffsynth/configs/vram_management_module_maps.py`（新增注册项）
- `plans/lowvram-plan.md`
- `skill_work_report/lowvram-report.md`
- `user_report/lowvram-report.md`

---

## Skill 6: diffsynth-pipeline-training

**目标**：为已接入推理 Pipeline 的模型系列创建训练模块并完成训练验证。

### Step 0: 读取蓝图信息
- 0.1 从蓝图报告读取 Conda 环境名称
- 0.2 从蓝图报告读取目标库路径
- 0.3 从蓝图报告读取 Pipeline 功能规划表
- 0.4 从蓝图报告读取训练相关信息

### Step 1: 初始化执行日志目录
- 1.1 创建 `execution-logs/{timestamp}_pipeline_training/` 目录结构

### Step 2: 制定执行计划
- 2.1 输出执行计划到 `plans/pipeline-training-plan.md`
- 2.2 向用户展示训练模块架构、数据集方案、训练脚本清单
- 2.3 等待用户确认后继续

### Step 3: 分析训练需求
- 3.1 确定 SeriesName、series、ModelName
- 3.2 确定任务类型（图像生成、视频生成等）
- 3.3 确定数据集字段

### Step 4: 准备数据集
- 4.1 下载 `diffsynth_example_dataset`（全量或按需）
- 4.2 复用同类型已有数据集到 `data/diffsynth_example_dataset/{series}/{ModelName}/`
- 4.3 如无可用数据集，请用户提供

### Step 5: 创建训练代码
- 5a. 创建 `{SeriesName}TrainingModule`
  - 5a.1 继承 `DiffusionTrainingModule`
  - 5a.2 实现 `__init__`（加载 Pipeline、设置训练模式）
  - 5a.3 实现 `get_pipeline_inputs`（数据集字段映射到 Pipeline 输入，必须自定义）
  - 5a.4 实现 `forward`（标准实现，不要修改）
- 5b. 确认/修正 `InputXXXEmbedder` 的训练分支
  - 5b.1 确认推理模式（`pipe.scheduler.training=False`）不返回 `input_latents`
  - 5b.2 确认训练模式返回 `{"latents": noise, "input_latents": ...}`
  - 5b.3 如缺失，按 Qwen-Image 标准实现创建
- 5c. 创建训练共用脚本 `examples/{series}/model_training/train.py`
- 5d. 创建训练配置
  - 5d.1 全量训练 `.sh` 脚本：`examples/{series}/model_training/full/{ModelName}-{Feature}.sh`
  - 5d.2 LoRA 训练 `.sh` 脚本：`examples/{series}/model_training/lora/{ModelName}-{Feature}.sh`
- 5e. 创建验证脚本
  - 5e.1 全量验证：`examples/{series}/model_training/validate_full/{ModelName}-{Feature}.py`
  - 5e.2 LoRA 验证：`examples/{series}/model_training/validate_lora/{ModelName}-{Feature}.py`

### Step 6: 训练 Debug 测试
- 6.1 运行 LoRA 训练（单卡，少量 step）
- 6.2 观察 loss 输出，确认训练能跑通
- 6.3 修复任何报错

### Step 7: 完整训练验证
- 7.1 运行 LoRA 训练（单卡，足够 step 数确认收敛趋势）
- 7.2 观察 loss 曲线，确认收敛
- 7.3 保存训练日志和 checkpoint

### Step 8: 最终验证
- 8.1 检查报告文件完整性
- 8.2 检查训练脚本、配置、验证脚本完整性
- 8.3 检查训练日志

**输出文件**：
- `examples/{series}/model_training/train.py`
- `examples/{series}/model_training/full/{ModelName}-{Feature}.sh`
- `examples/{series}/model_training/lora/{ModelName}-{Feature}.sh`
- `examples/{series}/model_training/validate_full/{ModelName}-{Feature}.py`
- `examples/{series}/model_training/validate_lora/{ModelName}-{Feature}.py`
- `plans/pipeline-training-plan.md`
- `skill_work_report/pipeline-training-report.md`
- `user_report/pipeline-training-report.md`

---

## Skill 7: diffsynth-style

**目标**：统一新接入模型代码的代码风格，使其与 DiffSynth-Studio 现有代码一致。

### Step 0: 读取蓝图信息
- 0.1 从蓝图报告读取 Conda 环境名称
- 0.2 从蓝图报告读取接入类型

### Step 1: 初始化执行日志目录
- 1.1 创建 `execution-logs/{timestamp}_style/` 目录结构

### Step 2: 制定执行计划
- 2.1 输出执行计划到 `plans/style-plan.md`
- 2.2 向用户展示风格化范围、目标标准
- 2.3 等待用户确认后继续

### Step 3: 分析现有代码风格
- 3.1 读取 DiffSynth-Studio 现有模型的代码风格
- 3.2 提取命名规范、注释风格、代码组织方式
- 3.3 提取导入顺序、空行规范、缩进规范

### Step 4: 风格化模型代码
- 4.1 风格化 `diffsynth/models/` 下的模型文件
- 4.2 风格化 `diffsynth/pipelines/` 下的 Pipeline 文件
- 4.3 风格化 `examples/` 下的推理示例脚本
- 4.4 风格化训练脚本和配置文件

### Step 5: 风格化推理脚本
- 5.1 统一推理脚本的结构（import → from_pretrained → call → save）
- 5.2 统一推理脚本的注释风格
- 5.3 统一推理脚本的 ModelConfig 写法

### Step 6: 风格化训练脚本
- 6.1 统一训练脚本的结构
- 6.2 统一 `.sh` 配置文件的格式
- 6.3 统一验证脚本的写法

### Step 7: 静态验证
- 7.1 检查所有修改后的 Python 文件语法
- 7.2 检查代码风格一致性
- 7.3 确保功能不受影响

### Step 8: 最终验证
- 8.1 检查报告文件完整性
- 8.2 检查风格化后的代码完整性

**输出文件**：
- 风格化后的模型代码、Pipeline 文件、推理脚本、训练脚本
- `plans/style-plan.md`
- `skill_work_report/style-report.md`
- `user_report/style-report.md`

---

## Skill 8: diffsynth-testing

**目标**：运行所有推理脚本，保存输出和报错信息供人工确认。

### Step 0: 读取蓝图信息
- 0.1 从蓝图报告读取 Conda 环境名称
- 0.2 从蓝图报告读取接入类型

### Step 1: 初始化执行日志目录
- 1.1 创建 `execution-logs/{timestamp}_testing/` 目录结构
- 1.2 创建 `tests/{timestamp}/` 输出审查目录
- 1.3 创建 `manifest.json` 和 `latest` 软链接

### Step 2: 制定执行计划
- 2.1 输出执行计划到 `plans/testing-plan.md`
- 2.2 向用户展示测试范围、输出路径规划、人工确认项
- 2.3 等待用户确认后继续

### Step 3: 收集测试范围
- 3.1 方式 A：用户直接指定脚本路径
- 3.2 方式 B：整体接入流程 — 扫描 `model_inference/` 和 `model_inference_low_vram/`
- 3.3 方式 C：version_upgrade — 额外扫描已有的旧脚本
- 3.4 去重后展示给用户确认

### Step 4: 运行推理脚本
- 4.1 创建测试运行器脚本 `run_inference_tests.py`
- 4.2 逐个运行每个推理脚本
  - 4.2.1 运行前 mtime 快照
  - 4.2.2 在 DiffSynth-Studio 根目录下执行脚本（`DIFFSYNTH_SKIP_DOWNLOAD=true`）
  - 4.2.3 检测新增/修改的输出文件
  - 4.2.4 立即移动到独立子目录
  - 4.2.5 判定状态（success / error）
  - 4.2.6 保存运行日志
- 4.3 失败不阻塞，继续运行下一个脚本

### Step 5: 生成测试报告
- 5.1 从 `test_results.json` 提取汇总数据
- 5.2 生成 `test_report.md` 到输出审查目录
- 5.3 包含结果汇总表、失败脚本报错信息和修复建议

### Step 6: 等待人工确认
- 6.1 向用户展示测试结果汇总
- 6.2 引导用户检查输出文件质量
- 6.3 引导用户查看失败脚本报错信息
- 6.4 用户确认后才能进入 PR 创建步骤

### Step 7: 最终验证
- 7.1 检查报告文件完整性
- 7.2 检查执行日志目录中的脚本完整性
- 7.3 如有缺失，立即补充

**输出文件**：
- `tests/latest/test_report.md`
- `tests/latest/{script_name}/`（各脚本输出文件）
- `execution-logs/{timestamp}_testing/`（执行日志）
- `plans/testing-plan.md`
- `skill_work_report/testing-report.md`
- `user_report/testing-report.md`

---

## Skill 9: diffsynth-docs

**目标**：根据已接入的模型代码生成中英文文档，更新 README。

### Step 0: 读取蓝图信息
- 0.1 从蓝图报告读取 Conda 环境名称
- 0.2 从蓝图报告读取目标库路径
- 0.3 从蓝图报告读取接入类型（new_series / version_upgrade）
- 0.4 从蓝图报告读取 Pipeline 功能规划表

### Step 1: 初始化执行日志目录
- 1.1 创建 `execution-logs/{timestamp}_docs/` 目录结构

### Step 2: 制定执行计划
- 2.1 输出执行计划到 `plans/docs-plan.md`
- 2.2 向用户展示接入类型、参考文档、信息提取清单
- 2.3 等待用户确认后继续

### Step 3: 判断接入类型
- 3.1 从蓝图报告读取接入类型
- 3.2 无蓝图时向用户确认

### Step 4: 读取参考文档
- 4.1 读取 `references/doc-templates.md` 文档模板
- 4.2 new_series：读取 2-3 篇参考文档提取风格
- 4.3 version_upgrade：读取现有文档了解结构

### Step 5: 从代码提取信息
- 5.1 读取 Pipeline 文件提取类名、参数签名
- 5.2 读取推理脚本列表和用法
- 5.3 读取训练脚本路径和参数
- 5.4 从 `model_configs.py` 提取模型 ID

### Step 6: 向用户收集缺失信息
- 6.1 请求一句话模型描述
- 6.2 请求特殊功能说明
- 6.3 用户暂不提供的字段留空或标注 TODO

### Step 7: 生成/更新文档
- 7.1 new_series：
  - 7.1.1 按模板结构构建中文文档
  - 7.1.2 安装、通用训练参数使用固定模板
  - 7.1.3 快速开始代码从最基础的推理脚本完整复制
  - 7.1.4 模型总览表格每行对应一个推理脚本
  - 7.1.5 推理参数从 `Pipeline.__call__` 签名逐行提取
  - 7.1.6 训练参数：通用参数用固定模板 + 专有参数从代码提取
  - 7.1.7 输出中文文档到 `docs/zh/Model_Details/{Model-Name}.md`
  - 7.1.8 同步生成英文文档到 `docs/en/Model_Details/{Model-Name}.md`
- 7.2 version_upgrade：
  - 7.2.1 读取现有中英文文档
  - 7.2.2 在模型总览表格中追加新模型行
  - 7.2.3 保持已有结构和内容不变

### Step 8: 更新 docs/ 内的注册点
- 8.1 new_series：在 `docs/zh/index.rst` 和 `docs/en/index.rst` 的 Model_Details toctree 区块追加条目
- 8.2 new_series：在 `docs/zh/README.md` 和 `docs/en/README.md` 的 Section 2 模型列表末尾追加条目
- 8.3 如模型需要可选依赖：更新 `docs/{zh,en}/Pipeline_Usage/Setup.md` 的依赖组说明
- 8.4 version_upgrade：跳过（条目已存在）

### Step 9: 更新仓库根 README
- 9.1 更新历史条目（new_series 和 version_upgrade 都需要），插入在最新一条上方
- 9.2 全局模型表：new_series 在 `## All Supported Models` / `## 全部支持的模型` 表最前插入新系列块；version_upgrade 在该系列已有的连续块内追加行
- 9.3 new_series：在 `## Basic Framework` / `## 基础框架` 的分类模型列表中按类别登记（ReadTheDocs 链接）
- 9.4 同步更新 README.md（英文）和 README_zh.md（中文）

### Step 10: 静态验证
- 10.1 检查中英文文档存在且格式正确
- 10.2 检查章节标题一一对应
- 10.3 检查代码链接有效性
- 10.4 检查模型总览表格行数与推理脚本数一致
- 10.5 检查推理参数与 Pipeline `__call__` 签名一致

### Step 11: 最终验证
- 11.1 检查三个报告文件是否存在
- 11.2 检查执行日志目录中的脚本完整性
- 11.3 如有缺失，立即补充

**输出文件**：
- `docs/zh/Model_Details/{Model-Name}.md`
- `docs/en/Model_Details/{Model-Name}.md`
- `docs/zh/index.rst` + `docs/en/index.rst`（更新）
- `docs/zh/README.md` + `docs/en/README.md`（更新）
- `docs/{zh,en}/Pipeline_Usage/Setup.md`（如需可选依赖）
- `README.md` + `README_zh.md`（更新）
- `plans/docs-plan.md`
- `skill_work_report/docs-report.md`
- `user_report/docs-report.md`

---

## Skill 10: diffsynth-pr

**目标**：生成 PR 描述 Markdown 文件（不执行 git 操作）。

### Step 0: 读取蓝图信息
- 0.1 从蓝图报告读取 Conda 环境名称
- 0.2 从蓝图报告读取接入类型
- 0.3 从蓝图报告读取 Pipeline 功能规划表

### Step 1: 初始化执行日志目录
- 1.1 创建 `execution-logs/{timestamp}_pr/` 目录结构

### Step 2: 以实际代码为准，校验蓝图信息
- 2.1 扫描 `diffsynth/pipelines/` 下与本次模型相关的文件
- 2.2 扫描 `diffsynth/models/` 下与本次模型相关的文件
- 2.3 扫描 `examples/` 下与本次模型相关的推理脚本
- 2.4 读取 `model_configs.py` 中新增的注册项
- 2.5 读取 `pyproject.toml` 的实际内容
- 2.6 发现不一致时，以实际代码为准

### Step 3: 收集接入信息
- 3.1 从蓝图报告读取接入类型
- 3.2 从实际 Pipeline 文件提取功能列表、类名、参数签名
- 3.3 从 `diffsynth/models/` 和 `diffsynth/pipelines/` 提取实际存在的文件
- 3.4 从 `examples/` 提取实际存在的推理脚本
- 3.5 从 `pyproject.toml` 提取依赖变更
- 3.6 从 `model_configs.py` 提取模型信息

### Step 4: 生成 PR 描述
- 4.1 参考 `references/pr-template.md` 模板
- 4.2 填入模型名称、接入的功能列表、模型组件清单、新增依赖、注意事项
- 4.3 输出到 `pr/description.md`

### Step 5: 最终验证
- 5.1 检查三个报告文件是否存在
- 5.2 检查 PR 描述文件是否存在
- 5.3 如有缺失，立即补充

**输出文件**：
- `pr/description.md`
- `plans/pr-plan.md`
- `skill_work_report/pr-report.md`
- `user_report/pr-report.md`

---

## Skill 11: diffsynth-pr-review

**目标**：处理 PR Review 反馈，分析评论并生成修改蓝图，执行针对性修复。

### Step 0: 读取蓝图信息
- 0.1 从蓝图报告读取 Conda 环境名称
- 0.2 从蓝图报告读取 Pipeline 功能规划表
- 0.3 从蓝图报告读取训练相关信息（如有）

### Step 1: 初始化执行日志目录
- 1.1 创建 `execution-logs/{timestamp}_pr-review/` 目录结构

### Step 2: 制定执行计划
- 2.1 输出执行计划到 `plans/pr-review-plan.md`
- 2.2 向用户展示反馈来源、分类策略、修改方案、测试规划
- 2.3 等待用户确认后继续

### Step 3: 获取 PR 反馈、分析反馈与生成修改蓝图
- 3a. 获取 PR 反馈
  - 3a.1 使用 curl 请求 GitHub API 获取 PR 基本信息
  - 3a.2 获取 PR-level Reviews（Gemini、Copilot 等工具的总体评审）
  - 3a.3 获取逐行代码评审评论（inline comments）
  - 3a.4 获取 PR 正文讨论区的普通评论
  - 3a.5 获取完整 diff 变更内容
  - 3a.6 将原始反馈保存到 `pr-review/raw-feedback.md`
- 3b. 分析反馈与生成修改蓝图
  - 3b.1 分类反馈（Bug/正确性、代码质量、功能增强、风格/格式）
  - 3b.2 定位涉及的具体文件和代码行
  - 3b.3 评估影响，判断是否合理、是否需要修改
  - 3b.4 追溯原有代码意图，决定是否采纳
  - 3b.5 生成修改蓝图 `pr-review/analysis-modification-blueprint.md`
  - 3b.6 展示给用户等待确认

### Step 4: 用户确认
- 4.1 将修改蓝图展示给用户
- 4.2 展示每条反馈的分类和修改方案
- 4.3 展示 diff 预览
- 4.4 必须等用户确认后才能继续

### Step 5: 执行修改
- 5.1 按蓝图中的修改项顺序逐一执行
- 5.2 修改前先读取文件确认当前内容
- 5.3 保持代码风格与 diffsynth-style 一致
- 5.4 修改完成后保存修改摘要到 `pr-review/modification-summary.md`
- 5.5 查看变更确认（git diff --stat, git diff）

### Step 6: 执行最小测试（如需要）
- 6a. 推理最小测试（当有修改涉及推理代码时）
  - 6a.1 复用已有推理脚本
  - 6a.2 完整运行，不篡改脚本参数
  - 6a.3 采用 mtime 检测 + 立即移动策略收集输出
  - 6a.4 保存运行日志和测试结果
- 6b. 训练最小测试（当有修改涉及训练代码时）
  - 6b.1 启动 LoRA 训练脚本，后台运行并 tee 日志
  - 6b.2 观察日志确认 loss 正常输出
  - 6b.3 等待约 10 个 step 后 kill 停掉
- 6c. 测试覆盖确认
  - 6c.1 确认每个修改项被至少一个测试覆盖
  - 6c.2 说明未覆盖的修改项原因

### Step 7: 验证修改
- 7.1 检查所有修改文件是否存在语法错误
- 7.2 确认所有报告文件已生成
- 7.3 查看最终变更统计
- 7.4 如有缺失，立即补充

**输出文件**：
- `pr-review/raw-feedback.md`
- `pr-review/analysis-modification-blueprint.md`
- `pr-review/modification-summary.md`
- `plans/pr-review-plan.md`
- `skill_work_report/pr-review-report.md`
- `user_report/pr-review-report.md`

---

## 共享目录结构

所有 skill 共享以下目录结构（基于 `packages/{model-name}/`）：

```
packages/{model-name}/
├── {target-library}/          # 目标模型库
├── DiffSynth-Studio/         # DiffSynth 框架
└── .sisyphus/                # 中间报告和日志
    ├── plans/                # Plan 文件（执行前制定）
    │   ├── {skill}-plan.md
    │   └── pipeline-{feature}-plan.md
    ├── skill_work_report/    # 渐进式步骤报告
    │   ├── {skill}-report.md
    │   └── pipeline-{feature}-report.md
    ├── user_report/          # 向用户报告（必须写入文件）
    │   └── {skill}-report.md
    ├── execution-logs/       # 执行日志（scripts/, outputs/, checkpoints/）
    │   ├── latest → {timestamp}_{skill}/
    │   └── {timestamp}_{skill}/
    ├── tests/                # 测试结果 + 输出审查
    │   ├── latest → {timestamp}/
    │   └── {timestamp}/
    ├── integration-blueprints/  # 蓝图报告 + 可行性分析
    │   ├── {model-name}-blueprint.md
    │   └── feasibility-analysis.md
    ├── code-reports/         # 代码阅读报告
    ├── reports/              # Pipeline 设计蓝图
    │   └── blueprint-{feature}.md
    ├── pr/                   # PR 描述
    │   └── description.md
    └── pr-review/            # PR Review 反馈
        ├── raw-feedback.md
        ├── analysis-modification-blueprint.md
        └── modification-summary.md
```
