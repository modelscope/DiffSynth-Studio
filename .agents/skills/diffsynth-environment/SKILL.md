---
name: diffsynth-environment
description: >
  Set up and validate the shared conda environment for DiffSynth-Studio and the target model library.
  Use this skill after diffsynth-analyze-target has generated a blueprint, or when, during a model
  integration, the user wants to install the target library's dependencies, verify that both
  DiffSynth and the target library run correctly in the same environment, or check that installing
  the target library's dependencies doesn't break DiffSynth's existing functionality. This skill
  follows a strict "DiffSynth OK → Target OK → DiffSynth still OK" validation sequence.
  SCOPE: Only for integrating a new model or a new model feature into DiffSynth-Studio. Do NOT
  invoke for any other development — general environment/dependency setup, bug fixes, refactoring,
  performance tuning, general code changes, or using/training already-supported models.
---

# DiffSynth-Studio: 环境创建

在同一个 conda 环境中，确保 DiffSynth-Studio 和目标库都能正常运行。

## 配置

从 `diffsynth-integrator/config.yaml` 读取：
- `env_clone_source` — 用于 clone 的源环境名称

**路径确定**：
- 所有路径基于 `packages/{model-name}/` 结构
- `diffsynth_root`: `packages/{model-name}/DiffSynth-Studio/`
- `target_path`: `packages/{model-name}/{target-library}/`（从蓝图报告获取）
- `.sisyphus` 目录: `packages/{model-name}/.sisyphus/`

**目标库路径**和**环境名称**不在 config.yaml 中管理：
- 目标库路径：从蓝图报告或对话上下文中获取
- **Conda 环境名称**：从蓝图报告「基本信息」表格中读取（格式为 `{model_name}-diffsynth`），不自行生成
  - **单一数据源**：蓝图报告是环境名称的唯一存储位置，详见 [blueprint-contract.md](../diffsynth-integrator/references/blueprint-contract.md)

**执行日志初始化**遵循 [exec-log-init.md](../diffsynth-integrator/references/exec-log-init.md) 规范。

## 核心原则

**双向验证，互不破坏。** 先验证 DiffSynth 本身正常，再逐步安装目标库依赖并验证目标库能运行，最后确认 DiffSynth 没被破坏。

**最小化安装，不盲从目标库的版本锁定。** 很多目标库的 requirements 写得过死（如 `gradio==6.2.0`），但实际不需要那么严格。策略是：先确保 DiffSynth 正常，再分析目标库需要哪些**额外**包，逐步安装，每装一批验证一次。

**外部依赖策略**：
- ✅ **transformers**：可以安装，DiffSynth 允许依赖 transformers
- ⚠️ **diffusers**：**环境验证阶段应安装**以验证目标库运行，但**模型代码阶段需重构**为独立实现（DiffSynth 接入代码不能依赖 diffusers）
- ⚠️ **其他外部库**：根据蓝图报告中的「外部依赖组件分析」逐个评估

### 关于 diffusers 的特殊处理

**策略澄清**：
- **环境验证阶段**：安装 diffusers 以验证目标库能正常运行（记录版本到检查点）
- **模型代码阶段**：将 diffusers 组件重构为独立 PyTorch 实现，最终接入代码不依赖 diffusers
- **不卸载 diffusers**：保留在环境中用于对比调试

**处理流程**：
1. 环境阶段：`pip install diffusers=={version}`（按蓝图要求的版本）
2. 验证目标库运行：确认目标库 example 能正常执行
3. 模型代码阶段：重构 DiT/Pipeline 移除 diffusers 基类依赖
4. 最终验证：DiffSynth 接入代码不依赖 diffusers，但环境保留 diffusers 用于对比

**执行留痕，过程可追溯。** 所有测试脚本先保存再执行，所有命令输出保存到日志，关键检查点明确标记。遵循 [execution-traceability.md](../diffsynth-integrator/references/execution-traceability.md) 规范。

## 工作流程

### ⚠️ 通用执行规则（适用于下方所有 Step）

**每条 Step 开始前 — 重读本步骤描述，确认关键约束：**
开始执行任何 Step 时，必须先重新阅读当前 Step 的描述内容。这是为了防止在执行过程中遗忘流程、规则或报告要求。阅读时重点关注：
- 核心原则和约束条件
- 当前 Step 的具体要求
- `## 输出` 章节中各报告的格式和路径

**每条 Step 结束后 — 更新渐进式报告：**
每个 Step 执行完成后，必须更新渐进式报告文件。报告路径：`packages/{model-name}/.sisyphus/skill_work_report/environment-report.md`

**更新方式：先读取现有报告，再追加新内容，最后写回文件。** 不要仅凭记忆追加，必须先读取文件确认当前内容。

追加的记录格式：
```bash
cat >> packages/{model-name}/.sisyphus/skill_work_report/environment-report.md << EOF

### Step {N}: {步骤名称}
- **状态**: ✅ 完成 / ❌ 失败
- **完成时间**: \$(date -Iseconds)
- **做了什么**: {简要描述}
- **关键结果**: {1-2 句话说明结果}
- **输出文件**: \`{文件路径}\`
EOF
```

**不要跳过报告更新** — 即使某个 Step 被跳过或失败，也必须记录到报告中。报告是执行过程的唯一可追溯记录。

### 0. 读取蓝图信息

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**每个 skill 执行的第一步，强制要求。** 从蓝图报告中读取 Python 运行环境信息和本 skill 必要的信息。

```bash
# 从 CLAUDE.md 或 config.yaml 获取模型名称
MODEL_NAME="{model-name}"
BLUEPRINT_PATH="packages/${MODEL_NAME}/.sisyphus/integration-blueprints/${MODEL_NAME}-blueprint.md"
```

**本 skill 必须从蓝图报告中读取的信息：**

| 蓝图信息 | 用途 |
|---------|------|
| 基本信息表中的 `Conda 环境名称` | 环境创建和激活 |
| 基本信息表中的 `目标库路径` | 环境验证范围 |
| 依赖差异分析表 | 必须安装/可选安装/不需要安装的包清单 |
| 可选依赖表（如有） | 根据平台/功能需要额外安装的包 |

如果蓝图报告不存在，向用户说明原因并中止。

> 📝 完成后：更新渐进式报告 → `skill_work_report/environment-report.md`

### 1. 初始化执行日志目录

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**环境配置完成后，** 创建执行日志目录结构：

```bash
# 从蓝图或上下文获取模型名称
MODEL_NAME="{model-name}"

# 检查并确保 DiffSynth-Studio 存在（如果不存在则 clone）
if [ ! -d "packages/${MODEL_NAME}/DiffSynth-Studio" ]; then
    git clone https://github.com/modelscope/DiffSynth-Studio.git packages/${MODEL_NAME}/DiffSynth-Studio/
fi

# 创建执行日志目录
export EXEC_LOG_DIR="packages/${MODEL_NAME}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_environment"
mkdir -p ${EXEC_LOG_DIR}/{scripts,outputs,checkpoints}

# 创建执行清单
cat > ${EXEC_LOG_DIR}/manifest.json << EOF
{
  "skill_name": "diffsynth-environment",
  "model_name": "${MODEL_NAME}",
  "timestamp": "$(date -Iseconds)",
  "execution_id": "env_$(date +%Y%m%d_%H%M%S)",
  "steps": [],
  "user_checks": []
}
EOF

# 更新 latest 软链接
ln -sfn ${EXEC_LOG_DIR} packages/${MODEL_NAME}/.sisyphus/execution-logs/latest
```

> 📝 完成后：更新渐进式报告 → `skill_work_report/environment-report.md`

### 2. 制定执行计划

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**在开始环境配置前，先制定完整的执行计划，输出到 `packages/{model-name}/.sisyphus/plans/environment-plan.md`。** 基于蓝图信息，明确依赖差异、验证方式、模型下载清单和 conda 环境创建方案。

Plan 文件采用统一的步骤章节格式，每个步骤包含「目标、执行内容、产出物、注意事项」，详见 Plan 模板章节。

**执行计划需要写入 Plan 文件，并向用户展示**，确认后再开始创建环境。

> 📝 完成后：更新渐进式报告 → `skill_work_report/environment-report.md`

### 3. 确认 DiffSynth 验证方式

> 📖 开始前：重读本步骤描述，确认流程与报告路径

环境创建阶段的 DiffSynth 验证只需要 **import 级别检查**，确认安装路径正确即可，不需要运行具体模型推理。验证内容为：

```python
import diffsynth
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
print(f"DiffSynth installed at: {diffsynth.__file__}")
print("Import check passed.")
```

**注意**：这与模型代码阶段（Step 2）的一致性测试不同——那些测试使用完整的推理脚本来验证接入前后的行为一致性。环境阶段只需确认"能 import"。

> 📝 完成后：更新渐进式报告 → `skill_work_report/environment-report.md`

### 4. 推荐模型下载（⏸️ 等待用户确认）

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**从蓝图报告读取模型清单**：在蓝图报告第 6 节「模型下载命令清单」中获取所有必需模型的 model_id。

**向用户展示下载命令**：

```bash
# {model-name} 模型下载清单
# 请在终端中执行以下命令，等待全部下载完成后再回复"已下载完成"

modelscope download --model {org_1}/{model_1} --local_dir ~/.cache/modelscope/hub/models/{org_1}/{model_1}
modelscope download --model {org_2}/{model_2} --local_dir ~/.cache/modelscope/hub/models/{org_2}/{model_2}
...
```

**⏸️ 等待用户确认**。向用户展示下载清单后，暂停执行并询问用户是否已完成下载：

> 请执行以上下载命令，等待全部完成后回复"已下载完成"，我将继续创建模型软链接。

- 如果用户回复已下载完成，继续 Step 5（创建模型软链接）
- 如果用户表示部分模型已存在，用 Step 10 的预检查脚本确认实际状态
- 如果用户需要帮助（如下载失败），协助排查问题

> 📝 完成后：更新渐进式报告 → `skill_work_report/environment-report.md`

### 5. 创建模型软链接

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**在环境创建之前**，先创建统一的模型软链接结构。

**核心原则**：`models/` 是真实目录（不做软链接），内部每个模型 repo 单独软链接到 ModelScope 缓存 `~/.cache/modelscope/hub/models/`。

从蓝图报告获取所需模型的 `{org}/{model-name}` 列表：

```bash
# 目标库：创建真实 models 目录
mkdir -p {target_path}/models

# DiffSynth：创建真实 models 目录
mkdir -p {diffsynth_root}/models

# 为每个模型在两个 locations 分别创建软链接 → ModelScope 缓存
for model_rel in jd-opensource/JoyAI-Image-Edit; do
    # 目标库
    mkdir -p {target_path}/models/$(dirname ${model_rel})
    ln -sfn ~/.cache/modelscope/hub/models/${model_rel} {target_path}/models/${model_rel}
    
    # DiffSynth
    mkdir -p {diffsynth_root}/models/$(dirname ${model_rel})
    ln -sfn ~/.cache/modelscope/hub/models/${model_rel} {diffsynth_root}/models/${model_rel}
done
```

#### 结构示意（以 JoyAI-Image 为例）

```
~/.cache/modelscope/hub/models/                  # ModelScope 缓存（模型真实文件）
└── jd-opensource/
    └── JoyAI-Image-Edit/                        # 真实文件：transformer/, vae/, ...

{target_path}/models/                            # packages/joyai-image/JoyAI-Image/models/
└── jd-opensource/                               # （真实目录）
    └── JoyAI-Image-Edit/  → ~/.cache/modelscope/hub/models/jd-opensource/JoyAI-Image-Edit

{diffsynth_root}/models/                         # packages/DiffSynth-Studio/models/
└── jd-opensource/                               # （真实目录）
    └── JoyAI-Image-Edit/  → ~/.cache/modelscope/hub/models/jd-opensource/JoyAI-Image-Edit
```

**关键要点**：
- `models/` 是**真实目录**，不是软链接
- 每个模型 repo（如 `jd-opensource/JoyAI-Image-Edit/`）是**独立软链接**，直接指向 `~/.cache/modelscope/hub/models/` 下的对应路径
- 目标库和 DiffSynth 各自拥有独立的 `models/` 目录，但内部模型 repo 软链接指向同一个 ModelScope 缓存路径

> 📝 完成后：更新渐进式报告 → `skill_work_report/environment-report.md`

### 6. 创建 conda 环境（通过 clone）

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**不手动创建环境。** 使用 `conda clone` 从已有环境复制，避免重复安装 PyTorch 等重型依赖。

**环境名称从蓝图报告读取**：在蓝图报告「基本信息」表格中查找 `Conda 环境名称` 字段。

```bash
conda create -n {conda_env_name} --clone {env_clone_source} -y
conda activate {conda_env_name}
```

**注意**：
- clone 源环境已包含 PyTorch 和 DiffSynth 基础依赖，无需额外安装
- 如果 clone 源环境不存在，先创建基础环境：`conda create -n {env_clone_source} python=3.10 -y` 并安装 PyTorch
- clone 完成后验证：`python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"`

> 📝 完成后：更新渐进式报告 → `skill_work_report/environment-report.md`

### 7. 安装 DiffSynth 并验证

> 📖 开始前：重读本步骤描述，确认流程与报告路径

```bash
cd {diffsynth_root}
pip install -e . 2>&1 | tee ${EXEC_LOG_DIR}/outputs/step_02_pip_install_diffsynth.log
```

**保存并运行 DiffSynth import 验证**：

```bash
# 保存验证脚本到执行日志目录
cat > ${EXEC_LOG_DIR}/scripts/test_diffsynth_import.py << 'PYEOF'
import sys
try:
    import diffsynth
    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
    print(f"DiffSynth installed at: {diffsynth.__file__}")
    print("Import check passed.")
    sys.exit(0)
except Exception as e:
    print(f"Import check failed: {e}")
    sys.exit(1)
PYEOF

# 执行验证
cat > ${EXEC_LOG_DIR}/scripts/run_diffsynth_test.sh << SCRIPT
#!/bin/bash
cd {diffsynth_root}
echo "[$(date -Iseconds)] 开始 DiffSynth import 验证测试" > "${EXEC_LOG_DIR}/outputs/step_02_diffsynth_test.log"
python ${EXEC_LOG_DIR}/scripts/test_diffsynth_import.py 2>&1 | tee -a "${EXEC_LOG_DIR}/outputs/step_02_diffsynth_test.log"
exit_code=\${PIPESTATUS[0]}
echo "[$(date -Iseconds)] 测试完成，exit code: \${exit_code}" >> "${EXEC_LOG_DIR}/outputs/step_02_diffsynth_test.log"
exit \${exit_code}
SCRIPT

chmod +x ${EXEC_LOG_DIR}/scripts/run_diffsynth_test.sh
bash ${EXEC_LOG_DIR}/scripts/run_diffsynth_test.sh
```

**通过标准**：`from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig` 无报错。

记录当前环境状态：
```bash
pip list > ${EXEC_LOG_DIR}/checkpoints/pip_list_diffsynth_ok.txt
```

**🔍 USER_CHECK: DiffSynth 验证结果**

```bash
cat >> ${EXEC_LOG_DIR}/user-checks.md << EOF

## [CHECK-003] DiffSynth import 验证
- **严重程度**: critical
- **描述**: 确认 DiffSynth 安装路径正确，模块可正常导入
- **查看日志**: \`cat ${EXEC_LOG_DIR}/outputs/step_02_diffsynth_test.log\`
- **通过标准**: 
  - 无 ImportError
  - \`diffsynth.pipelines\` 可正常导入
- **状态**: ⬜ 待确认
EOF
```

如果失败，先解决 DiffSynth 自身的环境问题，再继续。

> 📝 完成后：更新渐进式报告 → `skill_work_report/environment-report.md`

### 8. 分析目标库依赖

> 📖 开始前：重读本步骤描述，确认流程与报告路径

读取目标库的依赖声明（`target_path` 从蓝图报告获取）：

```bash
cat {target_path}/pyproject.toml
cat {target_path}/requirements.txt
```

对比当前环境（`pip list`），分析出三类包：

| 类别 | 说明 | 处理方式 |
|------|------|---------|
| **已有包** | 当前环境已安装，版本满足目标库要求 | 跳过，不重装 |
| **额外包** | 目标库独有、当前环境没有 | 需要安装 |
| **冲突包** | 目标库要求的版本与当前环境不兼容 | 谨慎处理，见下方 |

**冲突包处理原则**：
- **不要降级**当前环境的包（可能破坏 DiffSynth）
- 如果目标库要求 `gradio==6.2.0` 而当前环境没有 gradio → 安装
- 如果目标库要求 `transformers>=4.51.0,<4.58.0` 而当前环境已有 `transformers 4.52.0` → 跳过
- 如果目标库要求 `diffusers>=0.37.0` 而 DiffSynth 没有此依赖 → 安装（DiffSynth 不依赖 diffusers，不会冲突）
- 如果必须升级某个包才能满足目标库 → 先记录原版本，升级后验证 DiffSynth 是否仍正常

**平台特定依赖**：只安装当前平台需要的。识别 `sys_platform`、`platform_machine` 标记，过滤掉不相关的包（如 macOS 的 `mlx` 在 Linux 上不需要）。

> 📝 完成后：更新渐进式报告 → `skill_work_report/environment-report.md`

### 9. 逐步安装目标库依赖

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**不要一次性 `pip install -r requirements.txt`。** 按批次安装，每批验证：

```bash
# 第 1 批：核心推理依赖（不含训练、UI、API server）
pip install {core_package_1} {core_package_2} ...

# 验证 DiffSynth 未被破坏（import 级别）
cd {diffsynth_root}
python -c "from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig; print('DiffSynth import OK')"

# 第 2 批：特殊依赖（如 flash-attn、平台特定包）
pip install {special_package} ...

# 再次验证
cd {diffsynth_root}
python -c "from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig; print('DiffSynth import OK')"
```

**分批建议**：
- **第 1 批**：transformers、accelerate、einops、safetensors 等（如果 DiffSynth 已有则跳过）
- **第 2 批**：diffusers、vector-quantize-pytorch、soundfile 等目标库独有包
- **第 3 批**：flash-attn、torchao 等平台特定/编译密集型包
- **跳过**：gradio、fastapi、uvicorn 等 UI/API 包（除非目标库验证脚本需要）

每装完一批，记录环境变化：
```bash
pip list > ${EXEC_LOG_DIR}/checkpoints/pip_list_after_batch_N.txt
diff ${EXEC_LOG_DIR}/checkpoints/pip_list_diffsynth_ok.txt ${EXEC_LOG_DIR}/checkpoints/pip_list_after_batch_N.txt
```

> 📝 完成后：更新渐进式报告 → `skill_work_report/environment-report.md`

### 10. 模型预检查

> 📖 开始前：重读本步骤描述，确认流程与报告路径

在运行目标库验证脚本之前，先检查所需模型是否已准备就绪。

**从蓝图报告获取模型清单**：读取蓝图报告第 6 节「模型下载命令清单」。

**检查并创建软链接**：

```bash
cat > ${EXEC_LOG_DIR}/scripts/check_and_link_models.sh << SCRIPT
#!/bin/bash
# 模型预检查 + 软链接创建
LOG_FILE="${EXEC_LOG_DIR}/outputs/model_check.log"
echo "[$(date -Iseconds)] 开始模型预检查" > "${LOG_FILE}"

missing_models=()
for model_id in {model_id_1} {model_id_2} ...; do
    cache_path="${HOME}/.cache/modelscope/hub/models/${model_id}"
    if [ ! -d "${cache_path}" ]; then
        missing_models+=("${model_id}")
        echo "❌ 模型未找到: ${model_id}" | tee -a "${LOG_FILE}"
    else
        echo "✅ 模型已存在: ${model_id}" | tee -a "${LOG_FILE}"
        du -sh "${cache_path}" >> "${LOG_FILE}" 2>/dev/null || true
        
        # 在目标库和 DiffSynth 下分别创建软链接
        mkdir -p "{target_path}/models/$(dirname ${model_id})"
        ln -sf "${cache_path}" "{target_path}/models/${model_id}"
        mkdir -p "{diffsynth_root}/models/$(dirname ${model_id})"
        ln -sf "${cache_path}" "{diffsynth_root}/models/${model_id}"
    fi
done

if [ ${#missing_models[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  以下模型尚未下载："
    for model_id in "${missing_models[@]}"; do
        echo "   modelscope download --model ${model_id} --local_dir ~/.cache/modelscope/hub/models/${model_id}"
    done
    exit 1
fi
echo "[$(date -Iseconds)] 模型预检查通过，软链接已创建" >> "${LOG_FILE}"
SCRIPT

chmod +x ${EXEC_LOG_DIR}/scripts/check_and_link_models.sh
bash ${EXEC_LOG_DIR}/scripts/check_and_link_models.sh 2>&1 | tee ${EXEC_LOG_DIR}/outputs/model_check.log
```

**如果模型缺失**：预检查会列出具体下载命令，引导用户先完成下载。

> 📝 完成后：更新渐进式报告 → `skill_work_report/environment-report.md`

### 11. 验证目标库能运行

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**前提**：模型预检查通过，所有模型软链接已创建。

**⚠️ 核心原则：完整验证目标模型效果。** 这一步不是简单的 import 测试或冒烟测试，而是**在共享环境中完整运行目标库的推理流程**，验证目标库的模型能正常加载、推理、并生成有效输出。这是后续接入 DiffSynth 后对比效果的 golden baseline。

**验证要求**：
1. **完整推理流程**：运行目标库的完整推理脚本（如 `inference.py`、`demo.py`），加载目标模型的全部权重，跑完从输入到输出的完整 pipeline
2. **必须生成有效输出**：不仅仅是脚本不报错，必须产生目标模型的真实效果（图像/视频/音频等），人工可检查输出质量
3. **所有输出保存到执行日志目录**：禁止保存到 /tmp，修改原始脚本的输出路径或环境变量
4. **结果可追溯**：保存推理输入参数、输出文件、完整执行日志、运行耗时

**运行目标库验证脚本**：

从蓝图报告获取目标库验证脚本路径（一般为 `{target_path}/{target_validation_script}.py`）。

```bash
# 定义输出目录（必须在 EXEC_LOG_DIR 内）
TARGET_OUTPUT_DIR="${EXEC_LOG_DIR}/outputs/target_validation"
mkdir -p "${TARGET_OUTPUT_DIR}"

# 复制原始验证脚本（保留原始版本）
cp {target_path}/{target_validation_script}.py ${EXEC_LOG_DIR}/scripts/test_target_validation_original.py

# 创建修改版验证脚本（修改输出路径为 EXEC_LOG_DIR）
# 根据目标库脚本实际情况，修改输出路径变量/参数
# 原则：保持原始推理逻辑不变，只改输出路径
cp {target_path}/{target_validation_script}.py ${EXEC_LOG_DIR}/scripts/test_target_validation.py
# 用 sed 修改脚本中的输出路径（根据实际脚本调整）
# 例如：将输出路径替换为环境变量 OUTPUT_DIR
sed -i "s|output_path.*=.*'|output_path = '${TARGET_OUTPUT_DIR}/output.png'#|" ${EXEC_LOG_DIR}/scripts/test_target_validation.py
```

**如果目标库验证脚本是命令行形式**（如 `python main.py --input xxx --output yyy`）：

```bash
cat > ${EXEC_LOG_DIR}/scripts/run_target_test.sh << BASH_SCRIPT
#!/bin/bash
set -e
echo "[$(date -Iseconds)] 开始目标库完整推理验证" | tee "${EXEC_LOG_DIR}/outputs/target_test.log"
echo "输出目录: ${TARGET_OUTPUT_DIR}" | tee -a "${EXEC_LOG_DIR}/outputs/target_test.log"

cd {target_path}

# 运行目标库的完整推理脚本（修改输出路径）
python {target_validation_script} --output "${TARGET_OUTPUT_DIR}/output.png" \
    2>&1 | tee -a "${EXEC_LOG_DIR}/outputs/target_test.log"
exit_code=\${PIPESTATUS[0]}

echo "[$(date -Iseconds)] 推理完成，exit code: \${exit_code}" | tee -a "${EXEC_LOG_DIR}/outputs/target_test.log"
[ -f "${TARGET_OUTPUT_DIR}/output.png" ] && cp "${TARGET_OUTPUT_DIR}/output.png" "${EXEC_LOG_DIR}/outputs/target_output.png"
exit \${exit_code}
BASH_SCRIPT

chmod +x ${EXEC_LOG_DIR}/scripts/run_target_test.sh
bash ${EXEC_LOG_DIR}/scripts/run_target_test.sh
```

**如果目标库验证脚本是 Python 模块形式**（需要 import 后调用）：

```bash
cat > ${EXEC_LOG_DIR}/scripts/test_target_validation.py << 'PYTHON_SCRIPT'
import sys, os, json, time
from datetime import datetime

OUTPUT_DIR = os.environ.get('TARGET_VALIDATION_OUTPUT_DIR')
LOG_FILE = os.path.join(OUTPUT_DIR, 'execution.log')
RESULT_FILE = os.path.join(OUTPUT_DIR, 'result.json')

def log(message):
    line = f"[{datetime.now().isoformat()}] {message}"
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

log("=" * 60)
log("目标库完整推理验证开始")
log(f"输出目录: {OUTPUT_DIR}")

# --- 加载目标库 ---
sys.path.insert(0, '{target_path}')
import torch
log(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")

# --- 加载模型权重 ---
# TODO: 根据目标库实际模型加载方式修改
log("加载目标模型权重...")
start_time = time.time()

# from target_library import Pipeline
# model = Pipeline.from_pretrained("{model_path}")

# --- 完整推理 ---
log("开始推理...")
# TODO: 根据目标库实际推理流程修改
# result = model(prompt="...")
# result.save(os.path.join(OUTPUT_DIR, 'output.png'))

elapsed = time.time() - start_time
log(f"推理完成，耗时: {elapsed:.2f}秒")

# --- 验证输出 ---
output_path = os.path.join(OUTPUT_DIR, 'output.png')
result = {"success": False, "checks": {}}

if os.path.exists(output_path):
    result["checks"]["file_exists"] = True
    result["checks"]["file_size"] = os.path.getsize(output_path)
    from PIL import Image
    with Image.open(output_path) as img:
        result["checks"]["format"] = img.format
        result["checks"]["size"] = img.size
        result["checks"]["mode"] = img.mode
        result["success"] = img.size[0] > 0 and img.size[1] > 0

log(f"验证结果: {'✅ 通过' if result['success'] else '❌ 失败'}")
with open(RESULT_FILE, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

sys.exit(0 if result["success"] else 1)
PYTHON_SCRIPT
```

**🔍 USER_CHECK: 目标库验证**

```bash
cat >> ${EXEC_LOG_DIR}/user-checks.md << EOF

## [CHECK-004] 目标库完整推理验证
- **严重程度**: critical
- **描述**: 在共享环境中运行目标库的完整推理流程，验证模型能正常加载、推理、并生成有效输出
- **重要性**: 这是后续 DiffSynth 接入后的 golden baseline，必须确认目标库本身效果正确
- **查看日志**: \`cat ${EXEC_LOG_DIR}/outputs/target_test.log\`
- **查看结果**: \`cat ${EXEC_LOG_DIR}/outputs/target_result.json\`
- **查看输出**: \`ls -lh ${EXEC_LOG_DIR}/outputs/target_output.*\`
- **通过标准**:
  - 目标模型权重成功加载（无权重缺失/shape mismatch 错误）
  - 完整推理流程成功执行（不中途报错）
  - 输出文件存在且格式有效（图像/视频/音频等）
  - **人工检查输出内容质量**：确认目标模型效果符合预期
- **状态**: ⬜ 待确认
EOF
```

**失败处理**：
- Import Error → 缺依赖，回到 Step 9 安装缺失包
- 模型权重加载失败 → 检查模型文件是否完整，检查软链接是否正确
- Runtime Error → 记录到 `${EXEC_LOG_DIR}/checkpoints/target_test_error.txt`，排查具体原因
- 输出不存在 → 检查脚本输出路径配置
- 输出文件为零/损坏 → 推理中途失败或模型未正确初始化
- **输出质量不达标** → 确认推理参数是否正确，模型版本是否匹配

**输出文件说明**：
- `${EXEC_LOG_DIR}/outputs/target_test.log` — 完整执行日志
- `${EXEC_LOG_DIR}/outputs/target_output.*` — 目标库生成的输出文件（图像/视频/音频等）
- `${EXEC_LOG_DIR}/outputs/target_result.json` — 结构化验证结果
- `${EXEC_LOG_DIR}/scripts/test_target_validation_original.py` — 原始验证脚本备份

> 📝 完成后：更新渐进式报告 → `skill_work_report/environment-report.md`

### 12. 验证 DiffSynth 仍然正常

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**这是关键一步。** 安装完所有目标库依赖后，必须确认 DiffSynth 没被破坏。

```bash
# 保存最终环境状态
pip list > ${EXEC_LOG_DIR}/checkpoints/pip_list_final.txt
diff ${EXEC_LOG_DIR}/checkpoints/pip_list_diffsynth_ok.txt ${EXEC_LOG_DIR}/checkpoints/pip_list_final.txt \
    > ${EXEC_LOG_DIR}/checkpoints/pip_list_diff.txt 2>&1 || true

# 运行最终 import 验证
cat > ${EXEC_LOG_DIR}/scripts/test_diffsynth_import_final.py << 'PYEOF'
import sys
try:
    import diffsynth
    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
    print(f"DiffSynth installed at: {diffsynth.__file__}")
    print("Import check passed.")
    sys.exit(0)
except Exception as e:
    print(f"Import check failed: {e}")
    sys.exit(1)
PYEOF

cat > ${EXEC_LOG_DIR}/scripts/run_final_test.sh << SCRIPT
#!/bin/bash
cd {diffsynth_root}
echo "[$(date -Iseconds)] 开始最终 DiffSynth import 验证" > "${EXEC_LOG_DIR}/outputs/final_test.log"
python ${EXEC_LOG_DIR}/scripts/test_diffsynth_import_final.py 2>&1 | tee -a "${EXEC_LOG_DIR}/outputs/final_test.log"
exit_code=\${PIPESTATUS[0]}
echo "[$(date -Iseconds)] 测试完成，exit code: \${exit_code}" >> "${EXEC_LOG_DIR}/outputs/final_test.log"
exit \${exit_code}
SCRIPT

chmod +x ${EXEC_LOG_DIR}/scripts/run_final_test.sh
bash ${EXEC_LOG_DIR}/scripts/run_final_test.sh
```

**通过标准**：`from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig` 无报错，与 Step 7 结果一致。

**🔍 USER_CHECK: 最终验证**

```bash
cat >> ${EXEC_LOG_DIR}/user-checks.md << EOF

## [CHECK-005] 最终 DiffSynth import 验证（关键）
- **严重程度**: critical
- **描述**: 安装目标库依赖后，DiffSynth 是否仍然可正常导入
- **查看日志**: \`cat ${EXEC_LOG_DIR}/outputs/final_test.log\`
- **查看环境变化**: \`cat ${EXEC_LOG_DIR}/checkpoints/pip_list_diff.txt\`
- **通过标准**: `from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig` 无报错，与 Step 7 一致
- **状态**: ⬜ 待确认

## [CHECK-006] 依赖冲突评估（如存在）
- **严重程度**: warning
- **描述**: 检查环境变化中是否有潜在冲突
- **查看详情**: \`cat ${EXEC_LOG_DIR}/checkpoints/pip_list_diff.txt\`
- **状态**: ⬜ 待确认
EOF
```

**失败处理**：查看 `pip_list_diff.txt` 定位冲突包 → 尝试回滚 → 重新验证 → 无法解决则记录到 `${EXEC_LOG_DIR}/checkpoints/conflict_resolution.txt`。

> 📝 完成后：更新渐进式报告 → `skill_work_report/environment-report.md`

## 常见报错排查指南

### Import 错误

| 报错信息 | 可能原因 | 排查方法 |
|---------|---------|---------|
| `AttributeError: module 'transformers' has no attribute 'XxxForCausalLM'` | transformers 版本过低，不支持目标模型的 model_type | 1. 查看报错中的 model_type（如 `Ministral3`）<br>2. 搜索 HuggingFace 上该 model_type 首次支持的 transformers 版本<br>3. 升级到合适版本：`pip install transformers>=x.y.z` |
| `ModuleNotFoundError: No module named 'diffusers'` | diffusers 未安装 | `pip install diffusers` |
| `ImportError: cannot import name 'xxx' from 'xxx'` | 包版本不兼容 | 查看目标库 requirements 中指定版本 |
| `No module named 'flash_attn'` | flash-attn 未安装或 CUDA 版本不匹配 | 确认 CUDA 版本后安装：`pip install flash-attn --no-build-isolation` |

### 模型加载错误

| 报错信息 | 可能原因 | 排查方法 |
|---------|---------|---------|
| `ValueError: Unrecognized model in ...` | transformers 版本过低，无法识别该 model_type | 同上，升级 transformers |
| `Error(s) in loading state_dict for ...: size mismatch for ...` | 模型权重与模型结构不匹配 | 1. 确认模型权重文件未损坏<br>2. 确认使用的是正确的模型架构<br>3. 检查目标库是否有自定义的权重加载逻辑 |
| `Can't load weights for ...` | 权重文件格式不对 | 确认权重是 safetensors 还是 pytorch_model.bin 格式 |
| `RuntimeError: expected dtype Float but got Half` | 模型精度不匹配（FP16/FP32/BF16） | 检查模型加载时是否正确指定了 `torch_dtype` |

### 显存/Runtime 错误

| 报错信息 | 可能原因 | 排查方法 |
|---------|---------|---------|
| `CUDA out of memory` | 显存不足 | 1. 检查 GPU 显存：`nvidia-smi`<br>2. 尝试 `--use_gradient_checkpointing`（如果是训练脚本）<br>3. 降低 batch_size 或分辨率（仅测试阶段） |
| `RuntimeError: "xformers" not available` | xformers 未安装 | 可选安装：`pip install xformers` |
| `cuDNN error` | CUDA/cuDNN 版本不匹配 | 检查 PyTorch CUDA 版本与系统 CUDA 是否兼容 |

### 排查流程

遇到上述错误时，按以下流程排查：

```bash
# 1. 确认当前环境已安装包和版本
pip list | grep -i {package_name}

# 2. 查看目标库 requirements 中要求的版本
cat {target_path}/requirements.txt | grep {package_name}
cat {target_path}/pyproject.toml | grep {package_name}

# 3. 查看模型 config.json 中声明的 model_type / transformers 版本要求
cat ~/.cache/modelscope/hub/models/{org}/{model}/text_encoder/config.json | grep -i "model_type"
cat ~/.cache/modelscope/hub/models/{org}/{model}/config.json | grep -i "architectures"

# 4. 搜索该 model_type 对应的最低 transformers 版本要求
#    在 HuggingFace transformers 仓库搜索：https://github.com/huggingface/transformers
#    或使用 WebSearch 搜索 "{model_type} transformers version"

# 5. 升级/降级到合适版本
pip install {package_name}=={version}
```

**注意事项**：
- 升级 transformers/diffusers 可能影响 DiffSynth 已有功能，每次修改版本后都要重新运行 DiffSynth 验证
- 不同目标库可能要求不同版本的 transformers，优先满足当前目标库要求，再确认 DiffSynth 是否兼容

## 蓝图需求

本 skill 需要从蓝图报告中获取：
- **依赖差异分析**：哪些包需要新增、哪些有版本冲突
- **DiffSynth 验证脚本**：用于环境创建前后验证 DiffSynth 是否正常
- **目标库验证脚本**：用于验证目标库在共享环境中能否运行
- **目标库验证脚本所需模型**：需要下载哪些模型才能运行验证脚本
- **模型下载命令清单**（蓝图报告第 6 节）：包含所有必需模型的 model_id 和下载命令

## 目录结构

本 skill 读取和写入以下路径：
- **蓝图报告**: `packages/{model-name}/.sisyphus/integration-blueprints/{model-name}-blueprint.md`
- **执行日志**: `packages/{model-name}/.sisyphus/execution-logs/$(date)_environment/`
- **Plan**: `packages/{model-name}/.sisyphus/plans/environment-plan.md`
- **步骤报告**: `packages/{model-name}/.sisyphus/skill_work_report/environment-report.md`
- **用户报告**: `packages/{model-name}/.sisyphus/user_report/environment-report.md`

## 参考

- 依赖对比实例：[references/dependency-examples.md](references/dependency-examples.md)

### 13. 最终验证

> 📖 开始前：重读本步骤描述，确认流程与报告路径

在所有环境步骤完成后，执行最终验证：

```bash
# 1. 检查三个报告文件是否存在
for f in \
  "packages/{model-name}/.sisyphus/plans/environment-plan.md" \
  "packages/{model-name}/.sisyphus/skill_work_report/environment-report.md" \
  "packages/{model-name}/.sisyphus/user_report/environment-report.md"; do
  if [ ! -f "$f" ]; then
    echo "WARNING: 缺失报告文件: $f"
  fi
done

# 2. 检查所有执行的测试脚本是否存在于执行日志目录
EXEC_LOG_DIR="packages/{model-name}/.sisyphus/execution-logs/latest"
for script in "$EXEC_LOG_DIR/scripts/"*; do
  if [ -f "$script" ]; then
    echo "OK: $script 已保存"
  else
    echo "WARNING: 测试脚本缺失: $script"
  fi
done
```

如有缺失，立即补充。

> 📝 完成后：更新渐进式报告 → `skill_work_report/environment-report.md`

## 输出

### 执行日志

所有执行过程保存到：
- **执行日志目录**: `packages/{model-name}/.sisyphus/execution-logs/latest/`
- **脚本目录**: `scripts/` - 保存的所有测试脚本
- **输出目录**: `outputs/` - 命令执行日志
- **检查点目录**: `checkpoints/` - 中间结果和环境状态（`pip_list_*.txt`）
- **用户检查清单**: `user-checks.md` - 需要人工确认的项目

### Plan

在制定执行计划步骤，将详细执行计划输出到 `packages/{model-name}/.sisyphus/plans/environment-plan.md`。Plan 文件采用统一的步骤章节格式，每个步骤包含「目标、执行内容、产出物、注意事项」。模板如下：

```markdown
# Environment 执行 Plan

## 基本信息
| 字段 | 值 |
|------|-----|
| 模型名称 | {model-name} |
| Skill | diffsynth-environment |
| 执行时间 | {timestamp} |
| Conda 环境名称 | {model-name}-diffsynth |

## 执行步骤规划

以下按顺序列出所有执行步骤。每个步骤包含：目标、具体执行内容、产出物、注意事项。

### Step 0: 读取蓝图信息

**目标**：从蓝图报告中获取环境配置所需的上下文信息。

**执行内容**：
- 读取 `packages/{model-name}/.sisyphus/integration-blueprints/{model-name}-blueprint.md`
- 提取：依赖差异分析、DiffSynth 验证脚本路径、目标库验证脚本路径、验证脚本所需模型清单
- 如果蓝图报告不存在，向用户说明原因并中止

**产出物**：确认蓝图信息可用

---

### Step 1: 初始化执行日志目录

**目标**：创建执行日志目录结构。

**执行内容**：
- 创建 `packages/{model-name}/.sisyphus/execution-logs/{timestamp}_environment/` 目录及子目录
- 创建 `manifest.json` 和 `latest` 软链接

**产出物**：执行日志目录

---

### Step 2: 制定执行计划

**目标**：输出本 Plan 文件，向用户展示环境配置规划并确认。

**执行内容**：
- 将本 Plan 内容输出到 `packages/{model-name}/.sisyphus/plans/environment-plan.md`
- 向用户展示依赖分析、验证方案、模型下载清单
- 等待用户确认后继续

**产出物**：
- `packages/{model-name}/.sisyphus/plans/environment-plan.md`

---

### Step 3: 确认 DiffSynth 验证方式

**目标**：确定 DiffSynth 环境创建前后的验证方式。

**执行内容**：
- 确认 DiffSynth import 验证方式（导入基本模块）
- 确认是否有已有的验证脚本

**产出物**：验证方式确认

---

### Step 4: 推荐模型下载

**目标**：根据蓝图报告列出需要下载的模型清单。

**执行内容**：
- 从蓝图报告提取模型下载命令清单
- 展示给用户，确认预检查方式

**产出物**：模型下载清单

---

### Step 5: 创建模型软链接

**目标**：在 `{target_path}/models/` 和 `{diffsynth_root}/models/` 下创建软链接，指向 ModelScope 缓存。

**执行内容**：
- 创建目录结构
- 为每个模型创建软链接到 `~/.cache/modelscope/hub/models/{model_id}`

**产出物**：模型软链接

---

### Step 6: 创建 conda 环境

**目标**：clone 源环境并安装新增依赖。

**执行内容**：
- 从源环境 clone 创建新的 conda 环境
- 按批次安装依赖（核心 → 特殊 → 平台特定）
- 记录安装日志

**产出物**：conda 环境

**注意事项**：
- 分批安装，便于定位冲突

---

### Step 7: 安装 DiffSynth 并验证

**目标**：在新环境中安装 DiffSynth 并确认可导入。

**执行内容**：
- `pip install -e .` 安装 DiffSynth
- 在新环境中导入 `Flux2ImagePipeline` 验证
- 记录此时的 pip list 作为「DiffSynth OK」基线

**产出物**：验证结果 + pip list 基线

---

### Step 8: 分析目标库依赖

**目标**：对比目标库与 DiffSynth 的依赖，得出需要安装的清单。

**执行内容**：
- 读取目标库的依赖声明
- 与 DiffSynth 核心依赖及可选依赖组对比
- 输出分批安装清单，标注潜在版本冲突

**产出物**：依赖差异分析

---

### Step 9: 逐步安装目标库依赖

**目标**：分批安装目标库依赖，每批后验证 DiffSynth 未被破坏。

**执行内容**：
- 按批安装，每批后重新运行 DiffSynth 导入验证
- 记录每批前后的 pip list 差异

**产出物**：安装记录 + 每批的环境快照

---

### Step 10: 模型预检查

**目标**：确认所需模型已在缓存中且软链接有效。

**执行内容**：
- 检查 `~/.cache/modelscope/hub/models/` 下模型是否完整
- 检查软链接指向是否正确

**产出物**：模型状态报告

---

### Step 11: 验证目标库能运行

**目标**：验证目标库的推理脚本在共享环境中能跑通。

**执行内容**：
- 在共享环境中运行目标库验证脚本
- 确认输出文件存在且质量正常

**产出物**：验证结果 + 输出文件

---

### Step 12: 验证 DiffSynth 仍然正常

**目标**：确保所有操作完成后 DiffSynth 仍然正常。

**执行内容**：
- 重复 Step 10 的验证

**产出物**：验证结果

---

### Step 13: 最终验证

**目标**：确认所有报告文件和执行脚本完整性。

**执行内容**：
- 检查三个报告文件是否存在：Plan 文件、skill_work_report、user_report
- 检查执行日志目录中的脚本完整性
- 如有缺失，立即补充

**产出物**：验证通过确认

---

## 环境规划

### 依赖分析
- **源环境**: {env_clone_source}
- **新增依赖**: {target 库特有的包}
- **分批安装**: 核心依赖 → 特殊依赖 → 平台特定包
- **潜在冲突**: 需要关注的版本冲突

### 验证方案
- **DiffSynth 验证**: 导入 `diffsynth.pipelines.flux2_image.Flux2ImagePipeline`
- **目标库验证**: 运行目标库推理脚本
- **模型下载**: 预检查模型是否已在缓存中
```

### 渐进式步骤报告

**每个步骤完成后立即追加记录**。格式详见 [step-report.md](../diffsynth-integrator/references/step-report.md)。

报告路径：`packages/{model-name}/.sisyphus/skill_work_report/environment-report.md`

**步骤划分**（与上方「工作流程」章节的 Step 0-12 一一对应）：

| Step | 名称 | 对应 Workflow |
|------|------|---------------|
| 1 | 读取蓝图信息 | Step 0 |
| 2 | 初始化执行日志目录 | Step 1 |
| 3 | 制定执行计划 | Step 2 |
| 4 | 确认 DiffSynth 验证方式 | Step 3 |
| 5 | 推荐模型下载 | Step 4 |
| 6 | 创建模型软链接 | Step 5 |
| 7 | 创建 conda 环境 | Step 6 |
| 8 | 安装 DiffSynth 并验证 | Step 7 |
| 9 | 分析目标库依赖 | Step 8 |
| 10 | 逐步安装目标库依赖 | Step 9 |
| 11 | 模型预检查 | Step 10 |
| 12 | 验证目标库能运行 | Step 11 |
| 13 | 验证 DiffSynth 仍然正常 | Step 12 |

每完成一个步骤，**先读取现有报告文件，确认当前内容**，然后执行以下命令追加记录：
```bash
cat >> packages/{model-name}/.sisyphus/skill_work_report/environment-report.md << EOF

### Step {N}: {步骤名称}
- **状态**: ✅ 完成 / ❌ 失败
- **完成时间**: \$(date -Iseconds)
- **做了什么**: {简要描述}
- **关键结果**: {1-2 句话说明结果}
- **输出文件**: \`{文件路径}\`
EOF
```

### 向用户报告

环境准备完成后，向用户报告 **必须写入文件**：

```bash
cat > packages/{model-name}/.sisyphus/user_report/environment-report.md << 'OUTER_EOF'
## ✅ 环境准备完成

执行日志: `packages/{model-name}/.sisyphus/execution-logs/latest/`

### 📋 人工检查清单

请查看并确认以下检查项：
`cat packages/{model-name}/.sisyphus/execution-logs/latest/user-checks.md`

关键检查项：
1. [CHECK-005] 最终 DiffSynth 验证 - **必须确认**
2. [CHECK-006] 依赖冲突评估（如存在）

### 📁 生成的文件

- 测试脚本: `scripts/`
- 执行日志: `outputs/`
- 环境快照: `checkpoints/pip_list_*.txt`

### 🚀 下一步

确认所有检查项后，可以继续：
- `diffsynth-model-code` - 接入模型代码
OUTER_EOF
```
