---
name: diffsynth-testing
description: >
  Run all inference scripts for an integrated model in DiffSynth-Studio and save outputs for manual
  review. Scripts run from the DiffSynth-Studio root with DIFFSYNTH_SKIP_DOWNLOAD=true by default.
  Use this skill after all
  integration steps are done (post diffsynth-style), or whenever the user wants to validate that
  inference scripts run correctly before creating a PR. This skill only tests inference scripts — it
  does NOT test training code, does NOT run consistency tests, and does NOT modify any script code.
  Use this skill when the user wants to verify a model/feature integration before PR creation —
  e.g. "verify the integration", or asks whether the newly added inference scripts can execute
  successfully.
  SCOPE: Only for integrating a new model or a new model feature into DiffSynth-Studio. Do NOT
  invoke for any other development or testing — running existing scripts, bug fixes, refactoring,
  performance tuning, general code changes, or using/training already-supported models.
---

# DiffSynth-Studio: 推理脚本测试

运行所有推理脚本，保存输出和报错信息供人工确认。**不修改脚本代码，不做一致性测试，不做训练相关测试。**

## 配置

从 `diffsynth-integrator/config.yaml` 读取配置。

**路径确定**：
- 所有路径基于 `packages/{model-name}/` 结构
- `diffsynth_root`: `packages/{model-name}/DiffSynth-Studio/`
- `.sisyphus` 目录: `packages/{model-name}/.sisyphus/`

**Conda 环境名称**从蓝图报告中获取：
- **Conda 环境名称**：蓝图报告「基本信息」表格中的 `Conda 环境名称` 字段（如 `ace-step-diffsynth`）。**本 skill 执行的所有 Python 命令都必须使用该环境，使用 `conda run -n {conda_env_name} python ...` 形式。**

## 核心原则

**测试目标是完整跑一遍脚本。** 每个脚本都要完整执行其全部行为，不做任何代码修改或逻辑裁剪。

**⚠️ 所有测试脚本必须串行执行，不可并行，不可后台运行。** 所有推理脚本逐个运行，前一个脚本完全结束后才能开始下一个。原因：多个推理脚本同时运行会竞争 GPU 显存，导致 OOM 或结果不可复现。

| 规则 | 说明 |
|------|------|
| **串行执行** | 每个脚本必须等前一个完全结束后才能开始，不允许同时启动多个脚本进程 |
| **禁止后台运行** | 不可使用 `&`、`nohup`、`disown` 等将脚本放到后台执行 |
| **禁止并行/并行池** | 不可使用 `xargs -P`、`GNU parallel`、多进程等方式同时运行多个脚本 |
| **运行器脚本内部串行** | 运行器（`run_inference_tests.py`）内部通过 for 循环逐个执行，不可改为并发模式 |

**脚本不可修改。** 测试时必须原样运行脚本，**不得更改脚本的任何代码**。脚本的完整行为就是测试对象。

**禁止修改 DiffSynth-Studio 中已有的代码。** 测试过程中不得修改 DiffSynth-Studio 仓库内的任何代码文件（`diffsynth/` 下的所有内容），包括但不限于模型代码、管线代码、配置、工具函数等。如有需要修复的 bug，应报告并记录，而非自行修改。

**禁止超出工作目录范围的操作。** 测试过程中所有操作（文件创建、移动、读取、写入）必须限制在本工作目录（`packages/{model-name}/`）范围内：
- **不能将文件移出本工作目录**（包括移动到 `/tmp`、`~`、或其他任意目录）
- **不能修改工作目录外的任何文件**
- **不能读取工作目录外的文件**（模型软链接 `~/.cache/modelscope/hub/models/` 除外，这是测试必需的读取路径）

**输出集中存放供审查，运行后立即移动。** 采用 mtime 检测 + 立即移动策略，不做任何文件删除：

1. **mtime 快照**：每个脚本运行前，递归扫描 DS_ROOT 记录所有输出文件的 mtime
2. **检测变化**：脚本运行后，递归扫描，捕获新增文件或 mtime 更新的文件（即使文件名与之前相同，如 `output.jpg`）
3. **立即移动**：将检测到的文件立即移动到 `outputs/generated/{script_name}/` 独立目录，不会与后续脚本互相覆盖

**失败不阻塞。** 一个脚本报错不影响其他脚本的运行，全部运行完成后统一报告。报错时保存错误信息和推荐修复方案。

**默认跳过模型下载。** 所有测试脚本默认设置环境变量 `DIFFSYNTH_SKIP_DOWNLOAD=true`，直接使用本地已有的模型文件，不尝试联网下载。不要等报错后再设置。

**执行留痕，过程可追溯。** 所有测试脚本先保存再执行，所有命令输出保存到日志。遵循 [execution-traceability.md](../diffsynth-integrator/references/execution-traceability.md) 规范。

**不测试训练相关内容。** 训练脚本的验证由 `diffsynth-pipeline-training` skill 负责。

**不做一致性测试。** forward 一致性验证不在本 skill 范围内。

## 工作流程

### ⚠️ 通用执行规则（适用于下方所有 Step）

**每条 Step 开始前 — 重读本步骤描述，确认关键约束：**
开始执行任何 Step 时，必须先重新阅读当前 Step 的描述内容。这是为了防止在执行过程中遗忘流程、规则或报告要求。阅读时重点关注：
- 核心原则和约束条件
- 当前 Step 的具体要求
- `## 输出` 章节中各报告的格式和路径

**每条 Step 结束后 — 更新渐进式报告：**
每个 Step 执行完成后，必须更新渐进式报告文件。报告路径：`packages/{model-name}/.sisyphus/skill_work_report/testing-report.md`

**更新方式：先读取现有报告，再追加新内容，最后写回文件。** 不要仅凭记忆追加，必须先读取文件确认当前内容。

追加的记录格式：
```bash
cat >> packages/{model-name}/.sisyphus/skill_work_report/testing-report.md << EOF

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
| 基本信息表中的 `Conda 环境名称` | 测试环境 |
| 基本信息表中的 `接入类型` | 判断是否需要查找旧脚本（version_upgrade 时需要） |

如果蓝图报告不存在，向用户说明原因并中止。

> 📝 完成后：更新渐进式报告 → `skill_work_report/testing-report.md`

### 1. 初始化执行日志目录

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**读取蓝图信息后，** 创建执行日志目录结构：

```bash
export EXEC_LOG_DIR="packages/{model-name}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_testing"
export OUTPUT_REVIEW_DIR="packages/{model-name}/.sisyphus/tests/$(date +%Y%m%d_%H%M%S)"
mkdir -p ${EXEC_LOG_DIR}/{scripts,outputs}
mkdir -p ${OUTPUT_REVIEW_DIR}

cat > ${EXEC_LOG_DIR}/manifest.json << EOF
{
  "skill_name": "diffsynth-testing",
  "model_name": "{model-name}",
  "timestamp": "$(date -Iseconds)",
  "execution_id": "test_$(date +%Y%m%d_%H%M%S)",
  "steps": [],
  "test_scripts": [],
  "user_checks": []
}
EOF

ln -sfn ${EXEC_LOG_DIR} packages/{model-name}/.sisyphus/execution-logs/latest
ln -sfn ${OUTPUT_REVIEW_DIR} packages/{model-name}/.sisyphus/tests/latest
```

- `EXEC_LOG_DIR` — 执行日志、脚本、运行输出
- `OUTPUT_REVIEW_DIR` — 脚本产生的输出文件（图片、视频等），供用户审查

> 📝 完成后：更新渐进式报告 → `skill_work_report/testing-report.md`

### 2. 制定执行计划

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**在开始运行测试前，先制定完整的执行计划，输出到 `packages/{model-name}/.sisyphus/plans/testing-plan.md`。** 基于蓝图信息和收集到的脚本列表，明确测试范围、执行顺序、输出路径规划和人工确认项。

Plan 文件采用统一的步骤章节格式，每个步骤包含「目标、执行内容、产出物、注意事项」，详见 Plan 模板章节。

**执行计划需要写入 Plan 文件，并向用户展示**，确认后再开始运行测试。

> 📝 完成后：更新渐进式报告 → `skill_work_report/testing-report.md`

### 3. 收集测试范围

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**重要：以实际收集到的脚本为准。** 由于之前执行了 `diffsynth-style`，脚本的命名或路径可能与蓝图中的记录有差异。因此**不要依赖蓝图中的脚本清单**，而是通过扫描实际文件系统来确定。

**确定测试范围的三种方式：**

#### 方式 A：用户直接指定

如果用户在调用 skill 时明确指定了脚本路径或列表，直接使用用户指定的脚本，跳过其他扫描逻辑。

```bash
# 示例：用户说"只跑这几个脚本"
TEST_SCRIPTS=(
    "examples/ernie_image/model_inference/Ernie-Image-T2I.py"
    "examples/ernie_image/model_inference/Ernie-Image-I2I.py"
)
```

#### 方式 B：整体接入流程 — 扫描新接入的脚本

如果用户未指定脚本，说明是在整体接入流程中。扫描以下目录，确定所有推理脚本：

| 类型 | 路径 | 说明 |
|------|------|------|
| 推理脚本 | `examples/{series}/model_inference/*.py` | 每个功能一个 |
| 低显存推理 | `examples/{series}/model_inference_low_vram/*.py` | 可选 |

**只测试推理脚本**，不扫描 `model_training/` 下的任何内容。

#### 方式 C：version_upgrade 接入类型 — 同时查找旧脚本

如果蓝图报告中的接入类型是 `version_upgrade`（修改了已有模型），说明本次修改可能影响已有的旧脚本。除了扫描新接入的脚本外，还需要：

1. 从蓝图报告中读取已有的 Pipeline 功能规划表，了解之前已接入的功能
2. 扫描所有 `examples/` 下与该 series 相关的已有推理脚本（不仅是新创建的）
3. 将**旧脚本**也纳入测试范围，确保修改没有破坏已有功能

```bash
# version_upgrade 时需要额外检查的目录
# 扫描该 series 下所有已存在的推理脚本，不限于本次新增的
find examples/{series}/model_inference/ -name "*.py"
find examples/{series}/model_inference_low_vram/ -name "*.py" 2>/dev/null
```

**汇总后**，将所有待测试的脚本列出，去重后展示给用户确认。

如果没有找到任何脚本，跳过测试并告知用户。

> 📝 完成后：更新渐进式报告 → `skill_work_report/testing-report.md`

### 4. 运行推理脚本

> 📖 开始前：重读本步骤描述，确认流程与报告路径

**核心原则：不修改脚本代码。** 每个脚本必须原样运行，测试其完整行为。

**运行前检查**：
- 检查 `{target_path}/models/` 软链接是否存在
- 检查需要的模型是否已下载（通过蓝图报告中的模型列表）

**运行方式**：逐个串行运行每个推理脚本，保存输出到审查目录。

**⚠️ 串行执行规则**：所有脚本必须按顺序逐个运行，前一个脚本完全结束后才能开始下一个。**不可并行、不可后台运行（`&`/`nohup`/`disown`）、不可使用多进程池**。运行器内部通过 for 循环逐一执行，不可改为并发模式。

```bash
# 创建测试运行器脚本
cat > ${EXEC_LOG_DIR}/scripts/run_inference_tests.py << 'SCRIPT'
#!/usr/bin/env python3
"""
推理脚本测试运行器 — v4
原则：不修改任何脚本代码，不做任何文件删除
输出捕获：mtime 快照 + 检测变化 + 立即移动
边界：所有文件操作限制在 DS_ROOT（DiffSynth-Studio 根目录）内，不超出工作目录
"""

import os
import sys
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

LOG_FILE = os.environ.get("LOG_FILE", "test_suite.log")
RESULT_FILE = os.environ.get("RESULT_FILE", "test_results.json")
GENERATED_OUTPUT_DIR = os.environ.get("GENERATED_OUTPUT_DIR", "outputs/generated")
DS_ROOT = os.environ.get("DS_ROOT", ".")

OUTPUT_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.mp4', '.webm', '.wav', '.mp3', '.pt', '.safetensors', '.npy'}

# 保护目录：任何情况下都不扫描、不触碰的目录
PROTECTED_DIRS = {'.github', '.git', '__pycache__', '.cache', '.sisyphus', 'node_modules'}


def log(msg):
    ts = datetime.now().isoformat()
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def _should_skip(path, root_dir):
    """判断路径是否在保护目录中"""
    try:
        rel = Path(path).relative_to(root_dir)
        return rel.parts[0] in PROTECTED_DIRS
    except ValueError:
        return False


def snapshot_output_files(root_dir):
    """递归扫描，记录所有输出文件的 mtime。跳过保护目录。"""
    result = {}
    for f in Path(root_dir).glob("**/*"):
        if _should_skip(f, root_dir):
            continue
        if f.is_file() and f.suffix.lower() in OUTPUT_EXTENSIONS:
            result[str(f)] = f.stat().st_mtime
    return result


def collect_new_or_modified_files(root_dir, before_snapshot):
    """检测新增或 mtime 更新的输出文件。跳过保护目录。"""
    result = []
    for f in Path(root_dir).glob("**/*"):
        if _should_skip(f, root_dir):
            continue
        if not f.is_file() or f.suffix.lower() not in OUTPUT_EXTENSIONS:
            continue
        fpath = str(f)
        if fpath not in before_snapshot:
            result.append(fpath)
        elif f.stat().st_mtime > before_snapshot[fpath]:
            result.append(fpath)
    return result


def make_unique_script_name(rel_script_path):
    """生成唯一的脚本标识符，避免不同目录下同名脚本冲突。
    例: examples/ernie_image/model_inference/Ernie-Image-T2I.py -> inference_Ernie-Image-T2I
         examples/ernie_image/model_inference_low_vram/Ernie-Image-T2I.py -> inference_low_vram_Ernie-Image-T2I
    """
    parts = Path(rel_script_path).parts
    stem = Path(rel_script_path).stem
    # 查找包含 inference 或 training 的目录名作为前缀
    prefix = ""
    for p in parts:
        if "inference" in p.lower() or "training" in p.lower():
            prefix = p + "_"
    if prefix:
        return prefix + stem
    return stem


def move_outputs_to_review(outputs, script_name, output_dir):
    """将输出文件移动到独立子目录"""
    target_dir = Path(output_dir) / script_name
    target_dir.mkdir(parents=True, exist_ok=True)
    moved = []
    for src_path in outputs:
        src = Path(src_path)
        if src.exists() and src.is_file():
            dst = target_dir / src.name
            shutil.move(str(src), str(dst))
            moved.append(str(dst))
    return moved


def run_script(script_path, ds_root, env_override=None):
    """运行脚本，在 DiffSynth-Studio 根目录下执行"""
    env = os.environ.copy()
    env["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    if env_override:
        env.update(env_override)
    start_time = datetime.now()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=ds_root,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    duration = (datetime.now() - start_time).total_seconds()
    return result, duration


results = []

scripts_json = os.environ.get("TEST_SCRIPTS_JSON", "[]")
test_scripts = json.loads(scripts_json)

for script_path in test_scripts:
    rel_path = script_path if not os.path.isabs(script_path) else str(Path(script_path).relative_to(DS_ROOT))
    script_name = make_unique_script_name(rel_path)
    abs_path = script_path if os.path.isabs(script_path) else os.path.join(DS_ROOT, script_path)

    log(f"\n{'='*60}")
    log(f"运行测试: {rel_path}")
    log(f"工作目录: {DS_ROOT}")
    log(f"脚本标识: {script_name}")
    log(f"{'='*60}")

    # 运行前 mtime 快照
    before_snapshot = snapshot_output_files(DS_ROOT)

    # 运行脚本
    result, duration = run_script(abs_path, DS_ROOT)

    # 检测新增/修改的输出文件
    output_files = collect_new_or_modified_files(DS_ROOT, before_snapshot)

    # 立即移动到独立子目录
    review_files = move_outputs_to_review(output_files, script_name, GENERATED_OUTPUT_DIR)
    if review_files:
        log(f"已移动输出文件: {review_files}")

    # 判定状态
    status = "success" if result.returncode == 0 else "error"

    test_result = {
        "script": rel_path,
        "script_name": script_name,
        "status": status,
        "exit_code": result.returncode,
        "duration": duration,
        "retries": 0,
        "retry_reason": None,
        "output_files": output_files,
        "review_files": review_files,
        "error_info": None,
        "suggested_fix": None,
    }

    if result.returncode != 0:
        error_log_path = f"{GENERATED_OUTPUT_DIR}/{script_name}_error.log"
        with open(error_log_path, "w") as f:
            f.write(f"EXIT CODE: {result.returncode}\n\n")
            f.write(f"STDOUT (last 2000 chars):\n{result.stdout[-2000:]}\n\n")
            f.write(f"STDERR (last 2000 chars):\n{result.stderr[-2000:]}\n")
        test_result["error_info"] = {
            "exit_code": result.returncode,
            "stderr_tail": result.stderr[-2000:],
            "error_log": error_log_path,
        }

        stderr_lower = result.stderr.lower()
        suggestions = []
        if "module not found" in stderr_lower or "importerror" in stderr_lower:
            suggestions.append("缺少依赖模块，检查 conda 环境是否正确")
        if "cuda out of memory" in stderr_lower or "oom" in stderr_lower:
            suggestions.append("显存不足，尝试使用 model_inference_low_vram 脚本")
        if "file not found" in stderr_lower or "no such file" in stderr_lower:
            suggestions.append("文件路径不存在，检查模型路径和数据集路径")
        if "safetensors" in stderr_lower and "header" in stderr_lower:
            suggestions.append("safetensors 文件损坏或格式不正确，重新下载模型")
        if not suggestions:
            suggestions.append("检查报错堆栈，定位到具体文件和行号，查看代码逻辑")
        test_result["suggested_fix"] = suggestions

    results.append(test_result)

    status_icon = "✅" if status == "success" else "❌"
    log(f"{status_icon} {rel_path} (exit={result.returncode}, duration={duration:.1f}s)")

    # 保存详细日志
    log_path = f"{GENERATED_OUTPUT_DIR}/{script_name}_output.log"
    with open(log_path, "w") as f:
        f.write(f"EXIT CODE: {result.returncode}\n")
        f.write(f"DURATION: {duration:.1f}s\n\n")
        f.write(f"STDOUT:\n{result.stdout}\n\n")
        f.write(f"STDERR:\n{result.stderr}\n")

# 保存汇总结果
summary = {
    "timestamp": datetime.now().isoformat(),
    "total": len(results),
    "success": sum(1 for r in results if r["status"] == "success"),
    "error": sum(1 for r in results if r["status"] == "error"),
    "results": results,
}
with open(RESULT_FILE, "w") as f:
    json.dump(summary, f, indent=2, default=str)

log(f"\n测试完成: {len(results)} 个脚本, {summary['success']} 成功, {summary['error']} 失败")
SCRIPT

chmod +x ${EXEC_LOG_DIR}/scripts/run_inference_tests.py
```

**执行测试套件**：

**关键注意**：
- `script_name` 使用 `make_unique_script_name()` 生成，包含父目录前缀，避免同名脚本冲突
- 保护目录（`.github/`, `.git/`, `__pycache__/` 等）永远不会被扫描或触碰
- mtime 快照直接采集，无需预清理

```bash
# 将脚本列表转为 JSON 传入，DS_ROOT 指向 DiffSynth-Studio 根目录（使用绝对路径）
TEST_SCRIPTS_JSON='["script1.py", "script2.py"]'  # 替换为实际列表
DS_ROOT="$(cd packages/{model-name}/DiffSynth-Studio && pwd)"

LOG_FILE=${EXEC_LOG_DIR}/outputs/test_suite.log \
RESULT_FILE=${EXEC_LOG_DIR}/outputs/test_results.json \
GENERATED_OUTPUT_DIR="${EXEC_LOG_DIR}/outputs/generated" \
TEST_SCRIPTS_JSON="${TEST_SCRIPTS_JSON}" \
DS_ROOT="${DS_ROOT}" \
python ${EXEC_LOG_DIR}/scripts/run_inference_tests.py 2>&1 | tee ${EXEC_LOG_DIR}/outputs/run.log
```

**注意**：所有测试脚本必须在 `DiffSynth-Studio` 根目录下执行。运行器采用 mtime 检测 + 立即移动策略，**不做任何文件删除**：

1. **mtime 快照**：每个脚本运行前记录所有输出文件的 mtime（递归扫描）
2. **检测变化**：运行后递归扫描，捕获新增或 mtime 更新的文件
3. **立即移动**：检测到的文件立即移动到独立子目录

多个脚本使用相同输出文件名（如 `image.save("output.jpg")`）时，第一个脚本的输出被移动后，后续脚本会创建新文件，被 mtime 检测正确捕获。

> 📝 完成后：更新渐进式报告 → `skill_work_report/testing-report.md`

### 5. 生成测试报告

> 📖 开始前：重读本步骤描述，确认流程与报告路径

生成测试报告到 `packages/{model-name}/.sisyphus/tests/latest/test_report.md` 和 `${EXEC_LOG_DIR}/test_report.md`。

```bash
cat > ${OUTPUT_REVIEW_DIR}/test_report.md << EOF
# 推理脚本测试报告

**执行时间**: $(date -Iseconds)
**模型**: {model-name}

## 结果汇总

| 状态 | 数量 |
|------|------|
| 成功 | $(jq '.success' ${EXEC_LOG_DIR}/outputs/test_results.json) |
| 失败 | $(jq '.error' ${EXEC_LOG_DIR}/outputs/test_results.json) |
| 总计 | $(jq '.total' ${EXEC_LOG_DIR}/outputs/test_results.json) |

## 脚本详情

EOF

# 为每个脚本添加详情
jq -r '.results[] | "| \(.script_name) | \(.status) | \(.duration)s | \(.exit_code) | \(.review_files | join(", ")) |"' \
  ${EXEC_LOG_DIR}/outputs/test_results.json >> ${OUTPUT_REVIEW_DIR}/test_report.md

# 添加失败脚本的报错信息
ERROR_COUNT=$(jq '[.results[] | select(.status == "error")] | length' ${EXEC_LOG_DIR}/outputs/test_results.json)
if [ "$ERROR_COUNT" -gt 0 ]; then
    cat >> ${OUTPUT_REVIEW_DIR}/test_report.md << EOF

## 失败脚本详情

### 报错信息和修复建议

EOF
    jq -r '.results[] | select(.status == "error") | "#### \(.script_name)\n\n**退出码**: \(.exit_code)\n\n**报错信息**:\n```\(.error_info.stderr_tail)\n```\n\n**建议修复方案**:\n\(.suggested_fix | join("\n- "))\n\n**完整日志**: `\(.error_info.error_log)`\n"' \
      ${EXEC_LOG_DIR}/outputs/test_results.json >> ${OUTPUT_REVIEW_DIR}/test_report.md
fi
```

> 📝 完成后：更新渐进式报告 → `skill_work_report/testing-report.md`

### 6. 等待人工确认

> 📖 开始前：重读本步骤描述，确认流程与报告路径

向用户报告：

```markdown
## ✅ 推理脚本测试完成

执行日志: `packages/{model-name}/.sisyphus/execution-logs/latest/`
测试报告: `packages/{model-name}/.sisyphus/tests/latest/test_report.md`
输出审查: `packages/{model-name}/.sisyphus/tests/latest/`

### 📊 测试结果汇总

| 状态 | 数量 |
|------|------|
| 成功 | X |
| 失败 | Y |
| 总计 | X+Y |

### 📁 审查目录

每个脚本的输出文件保存在 `packages/{model-name}/.sisyphus/tests/latest/{script_name}/` 下。
失败脚本的报错信息保存在 `packages/{model-name}/.sisyphus/tests/latest/{script_name}_error.log` 下。

### 🔍 需要你确认

- [ ] 检查成功脚本的输出文件（图片/视频等）质量是否符合预期
- [ ] 如果有失败脚本，查看报错日志和修复建议，确认是否需要修复后重跑
```

**用户确认所有输出质量后**，才能进入 PR 创建步骤。

> 📝 完成后：更新渐进式报告 → `skill_work_report/testing-report.md`

### 7. 最终验证

> 📖 开始前：重读本步骤描述，确认流程与报告路径

在所有测试步骤完成后，执行最终验证：

```bash
# 1. 检查报告文件是否存在
for f in \
  "packages/{model-name}/.sisyphus/plans/testing-plan.md" \
  "packages/{model-name}/.sisyphus/skill_work_report/testing-report.md" \
  "packages/{model-name}/.sisyphus/user_report/testing-report.md"; do
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

> 📝 完成后：更新渐进式报告 → `skill_work_report/testing-report.md`

## 目录结构

本 skill 读取和写入以下路径：
- **蓝图报告**: `packages/{model-name}/.sisyphus/integration-blueprints/{model-name}-blueprint.md`（用于获取环境信息）
- **测试结果**: `packages/{model-name}/.sisyphus/tests/$(date)_testing/`
- **输出审查**: `packages/{model-name}/.sisyphus/tests/$(date)_testing/`（与测试结果同目录）
- **执行日志**: `packages/{model-name}/.sisyphus/execution-logs/$(date)_testing/`

## 输出

### 执行日志

所有执行过程保存到：
- **执行日志目录**: `packages/{model-name}/.sisyphus/execution-logs/latest/`
- **脚本目录**: `scripts/` - 保存的测试运行器
- **输出目录**: `outputs/` - 运行日志和结果汇总

### 输出审查

每个脚本的输出文件按脚本名分目录存放（采用 mtime 检测 + 立即移动策略，**不做任何文件删除**）：
- `packages/{model-name}/.sisyphus/tests/latest/{script_name}/` — 成功脚本的输出文件
- `packages/{model-name}/.sisyphus/tests/latest/{script_name}_output.log` — 脚本完整 stdout/stderr
- `packages/{model-name}/.sisyphus/tests/latest/{script_name}_error.log` — 失败脚本的报错信息（如有）

执行日志目录中也同样保留一份：
- `${EXEC_LOG_DIR}/outputs/generated/{script_name}/` — 移动过来的输出文件

### Plan

执行完成后，生成 Plan 文件到 `packages/{model-name}/.sisyphus/plans/testing-plan.md`。Plan 文件采用统一的步骤章节格式，每个步骤包含「目标、执行内容、产出物、注意事项」。模板如下：

```markdown
# Testing 执行 Plan

## 基本信息
| 字段 | 值 |
|------|-----|
| 模型名称 | {model-name} |
| Skill | diffsynth-testing |
| 执行时间 | {timestamp} |
| 接入类型 | {new_series / version_upgrade} |

## 执行步骤规划

以下按顺序列出所有执行步骤。每个步骤包含：目标、具体执行内容、产出物、注意事项。

### Step 0: 读取蓝图信息

**目标**：从蓝图报告中获取测试所需的上下文信息。

**执行内容**：
- 读取 `packages/{model-name}/.sisyphus/integration-blueprints/{model-name}-blueprint.md`
- 提取：Conda 环境名称、接入类型
- 如果接入类型为 version_upgrade，需要额外关注已有脚本的回归测试
- 如果蓝图报告不存在，向用户说明原因并中止

**产出物**：确认蓝图信息可用

**注意事项**：
- Conda 环境名称必须从蓝图报告的「基本信息」表格中获取，不自行推断

---

### Step 1: 初始化执行日志目录

**目标**：创建标准化的执行日志和测试输出目录结构。

**执行内容**：
- 创建 `packages/{model-name}/.sisyphus/execution-logs/{timestamp}_testing/` 目录及子目录（scripts/, outputs/）
- 创建 `packages/{model-name}/.sisyphus/tests/{timestamp}/` 作为输出审查目录
- 创建 `manifest.json` 记录执行元信息
- 创建 `latest` 软链接

**产出物**：
- 执行日志目录、输出审查目录
- `manifest.json`、`latest` 软链接

**注意事项**：
- `EXEC_LOG_DIR` 和 `OUTPUT_REVIEW_DIR` 使用带时间戳的路径，确保每次测试独立

---

### Step 2: 制定执行计划

**目标**：输出本 Plan 文件，向用户展示完整测试规划并确认。

**执行内容**：
- 将本 Plan 内容输出到 `packages/{model-name}/.sisyphus/plans/testing-plan.md`
- 向用户展示测试范围、输出路径规划、人工确认项
- 等待用户确认后继续

**产出物**：
- `packages/{model-name}/.sisyphus/plans/testing-plan.md`

**注意事项**：
- 测试范围以实际文件系统中的脚本为准，不依赖蓝图中的脚本清单

---

### Step 3: 收集测试范围

**目标**：确定需要运行的推理脚本列表。

**执行内容**：
- 如果用户指定了脚本路径，直接使用
- 否则扫描 `examples/{series}/model_inference/*.py` 和 `model_inference_low_vram/*.py`
- 如果是 version_upgrade，额外扫描已有的旧脚本
- 去重后展示给用户确认

**产出物**：确认后的测试脚本列表

**注意事项**：
- diffsynth-style 可能已重命名脚本，必须以实际扫描结果为准
- 未找到任何脚本时跳过测试

---

### Step 4: 运行推理脚本

**目标**：逐个运行测试脚本，收集输出和报错。

**执行内容**：
- 在 DiffSynth-Studio 根目录下执行每个脚本（`DIFFSYNTH_SKIP_DOWNLOAD=true`）
- 采用 mtime 快照 + 立即移动策略收集输出文件
- 失败不阻塞，继续运行下一个脚本
- 保存每个脚本的 stdout/stderr 日志

**产出物**：
- 每个脚本的运行日志（`outputs/`）
- 输出文件（`outputs/generated/{script_name}/`）
- 失败脚本的报错日志

**注意事项**：
- 不修改任何脚本代码，原样运行
- 不修改 DiffSynth-Studio 中已有的代码

---

### Step 5: 生成测试报告

**目标**：汇总测试结果，生成可读的测试报告。

**执行内容**：
- 从 `test_results.json` 提取汇总数据
- 生成 `test_report.md` 到输出审查目录
- 包含：结果汇总表、失败脚本的报错信息和修复建议

**产出物**：
- `packages/{model-name}/.sisyphus/tests/latest/test_report.md`

---

### Step 6: 等待人工确认

**目标**：向用户展示测试结果，等待质量确认。

**执行内容**：
- 展示测试结果汇总（成功/失败数量）
- 引导用户检查输出文件质量
- 引导用户查看失败脚本的报错信息

**产出物**：用户确认结果

**注意事项**：
- 用户确认所有输出质量后，才能进入 PR 创建步骤

---

### Step 7: 最终验证

**目标**：确认所有报告文件和执行脚本完整性。

**执行内容**：
- 检查三个报告文件是否存在：Plan 文件、skill_work_report、user_report
- 检查执行日志目录中的脚本完整性
- 如有缺失，立即补充

**产出物**：验证通过确认

---

## 测试规划

### 测试范围
- 以实际扫描到的推理脚本为准
- version_upgrade 时同时包含旧脚本（回归测试）
- 不测试训练相关内容，不做一致性测试

### 输出路径规划
每个脚本运行后，其产生的输出文件通过 mtime 检测 + 立即移动到对应的独立子目录：

| 原始脚本 | 输出目录 |
|----------|---------|
| {script_1} | outputs/generated/{script_1_name}/ |
| {script_2} | outputs/generated/{script_2_name}/ |

**关键机制**：
- 不做任何文件删除
- mtime 检测确保即使文件名重复也能正确捕获
- 立即移动避免多个脚本互相覆盖

### 失败处理策略
- 一个脚本报错不影响其他脚本
- 保存错误日志和建议修复方案
- 全部运行完成后统一报告
```

### 渐进式步骤报告

**每个步骤完成后立即追加记录**。格式详见 [step-report.md](../diffsynth-integrator/references/step-report.md)。

报告路径：`packages/{model-name}/.sisyphus/skill_work_report/testing-report.md`

**步骤划分**（与上方「工作流程」章节的 Step 0-7 一一对应）：

| Step | 名称 | 对应 Workflow |
|------|------|---------------|
| 0 | 读取蓝图信息 | Step 0 |
| 1 | 初始化执行日志目录 | Step 1 |
| 2 | 制定执行计划 | Step 2 |
| 3 | 收集测试范围 | Step 3 |
| 4 | 运行推理脚本 | Step 4 |
| 5 | 生成测试报告 | Step 5 |
| 6 | 等待人工确认 | Step 6 |
| 7 | 最终验证 | Step 7 |

每完成一个步骤，**先读取现有报告文件，确认当前内容**，然后执行以下命令追加记录：
```bash
cat >> packages/{model-name}/.sisyphus/skill_work_report/testing-report.md << EOF

### Step {N}: {步骤名称}
- **状态**: ✅ 完成 / ❌ 失败
- **完成时间**: \$(date -Iseconds)
- **做了什么**: {简要描述}
- **关键结果**: {1-2 句话说明结果}
- **输出文件**: \`{文件路径}\`
EOF
```

### 向用户报告

测试完成后，向用户报告 **必须写入文件**：

```bash
cat > packages/{model-name}/.sisyphus/user_report/testing-report.md << 'OUTER_EOF'
## ✅ 推理脚本测试完成

执行日志: `packages/{model-name}/.sisyphus/execution-logs/latest/`
测试报告: `packages/{model-name}/.sisyphus/tests/latest/test_report.md`
输出审查: `packages/{model-name}/.sisyphus/tests/latest/`

### 📊 测试结果

| 状态 | 数量 |
|------|------|
| 成功 | X |
| 失败 | Y |
| 总计 | X+Y |

### 📁 输出文件

- 成功脚本输出: `.sisyphus/tests/latest/{script_name}/`
- 运行日志: `.sisyphus/tests/latest/{script_name}_output.log`
- 失败报错: `.sisyphus/tests/latest/{script_name}_error.log`（如有）

### 🚀 下一步

确认所有输出质量后，可以继续：
- `diffsynth-pr` - 创建 PR
OUTER_EOF
```
