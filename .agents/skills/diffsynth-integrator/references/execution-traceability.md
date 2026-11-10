# DiffSynth Skill 执行留痕规范

所有 diffsynth-* skill 必须遵循统一的执行留痕机制，确保过程可追溯、结果可审查、问题可定位。

## 核心原则

1. **代码先存后执行**：所有测试/验证代码必须先写入文件，再执行
2. **执行必有日志**：每个命令的执行结果、输出、错误都保存到日志
3. **关键节点标记**：明确标记需要人工检查的地方
4. **结果可复现**：保存足够信息让用户可以独立复现执行

## 目录结构

```
packages/{model-name}/.sisyphus/
├── plans/                              # Plan 文件（执行前制定）
│   └── {skill}-plan.md
├── skill_work_report/                  # 渐进式步骤报告
│   └── {skill}-report.md
├── user_report/                        # 向用户报告
│   └── {skill}-report.md
├── integration-blueprints/             # 蓝图报告
│   ├── {model-name}-blueprint.md
│   └── feasibility-analysis.md
├── code-reports/                       # 代码阅读报告
│   └── code-report.md
├── reports/                            # Pipeline 设计蓝图
│   └── blueprint-{feature}.md
├── execution-logs/                     # 执行日志
│   ├── {timestamp}_{skill-suffix}/
│   │   ├── manifest.json               # 执行元数据
│   │   ├── commands.log                # 执行的命令记录
│   │   ├── outputs/                    # 命令输出
│   │   ├── scripts/                    # 保存的测试脚本
│   │   ├── checkpoints/                # 中间检查点
│   │   └── user-checks.md              # 需要用户检查的项目
│   └── latest -> {timestamp}_{skill-suffix}/
├── tests/                              # 测试结果 + 输出审查
│   ├── {timestamp}/
│   │   └── test_report.md
│   └── latest -> {timestamp}/
├── pr/                                 # PR 描述
│   └── description.md
└── pr-review/                          # PR Review 反馈
    ├── raw-feedback.md
    ├── analysis-modification-blueprint.md
    └── modification-summary.md
```

## 执行清单（manifest.json）

每个 skill 执行时生成：

```json
{
  "skill_name": "diffsynth-environment",
  "model_name": "ace-step",
  "timestamp": "2025-04-07T14:30:00+08:00",
  "execution_id": "env_20250407_143000",
  "steps": [
    {
      "step_id": 1,
      "name": "创建 conda 环境",
      "status": "completed",
      "command": "conda create -n ace-step-diffsynth --clone base -y",
      "log_file": "outputs/step_01_conda_create.log",
      "duration_ms": 45000,
      "exit_code": 0
    },
    {
      "step_id": 2,
      "name": "模型预检查",
      "status": "completed",
      "command": "bash scripts/check_models.sh",
      "script_file": "scripts/check_models.sh",
      "log_file": "outputs/step_02_model_check.log",
      "duration_ms": 5000,
      "exit_code": 0,
      "user_check_required": false
    }
  ],
  "user_checks": [
    {
      "check_id": "check_01",
      "title": "依赖冲突检查",
      "description": "发现 transformers 版本冲突，请确认是否接受",
      "severity": "warning",
      "file": "checkpoints/dependency_conflict.txt",
      "status": "pending"
    }
  ]
}
```

## 测试脚本保存规范

### 1. 脚本保存位置

所有测试脚本保存到 `execution-logs/{model-name}/{timestamp}_{skill-name}/scripts/`：

```
scripts/
├── check_models.sh              # 模型预检查脚本
├── test_import.py               # 导入测试
├── test_forward.py              # forward 一致性测试
├── test_e2e.py                  # 端到端测试
└── README.md                    # 脚本说明
```

### 2. 脚本头部模板

每个脚本必须包含以下头部：

```python
#!/usr/bin/env python3
"""
测试脚本: {描述}
生成时间: {timestamp}
所属 Skill: {skill_name}
执行命令: python {script_name}

用途:
  - {详细说明}

预期输出:
  - {预期结果}

需要检查:
  - {需要人工确认的地方}
"""

import sys
import json
from datetime import datetime

# 执行日志配置
LOG_FILE = "../outputs/{script_name}_{timestamp}.log"
RESULT_FILE = "../checkpoints/{script_name}_result.json"

def log(message):
    timestamp = datetime.now().isoformat()
    line = f"[{timestamp}] {message}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def save_result(result):
    with open(RESULT_FILE, "w") as f:
        json.dump(result, f, indent=2)

# 主逻辑
if __name__ == "__main__":
    log("开始执行测试...")
    # ... 测试逻辑 ...
    result = {"status": "passed", "details": {...}}
    save_result(result)
    log("测试完成")
```

## 执行日志规范

### 1. 命令日志（commands.log）

记录每个执行的命令：

```bash
# [2025-04-07 14:30:00] Step 1: 创建 conda 环境
conda create -n ace-step-diffsynth --clone base -y
# Exit code: 0
# Duration: 45.2s

# [2025-04-07 14:30:50] Step 2: 激活环境
conda activate ace-step-diffsynth
# Exit code: 0

# [2025-04-07 14:31:00] Step 3: 模型预检查
bash /path/to/scripts/check_models.sh
# Exit code: 0
# Output: outputs/step_03_model_check.log
```

### 2. 输出日志

每个命令的标准输出和错误保存到 `outputs/step_{N}_{name}.log`：

```
[2025-04-07 14:31:00] 开始模型预检查
[2025-04-07 14:31:01] 检查模型: ACE-Step/ACE-Step-1.5-Text2Music
[2025-04-07 14:31:01] ✅ 模型已存在: ~/.cache/modelscope/hub/ACE-Step/ACE-Step-1.5-Text2Music
[2025-04-07 14:31:02] 检查完成，所有模型已就绪
```

## 人工检查点标记

### 1. 检查点文件（user-checks.md）

生成 Markdown 格式的检查清单：

```markdown
# 人工检查清单

执行: diffsynth-environment
时间: 2025-04-07 14:35:00
模型: ace-step

## ⚠️ 需要确认的项目

### [CHECK-001] 依赖版本冲突
- **严重程度**: warning
- **描述**: transformers 版本要求 4.51.0，当前环境为 4.52.0
- **影响**: 可能不影响功能，但建议确认
- **查看详情**: `cat checkpoints/dependency_conflict.txt`
- **建议操作**: 接受当前版本或降级到 4.51.0
- **状态**: ⬜ 待确认

### [CHECK-002] 模型下载完整性
- **严重程度**: info
- **描述**: 已自动创建软链接，请验证模型文件可访问
- **验证命令**:
  ```bash
  ls -la ~/.cache/modelscope/hub/ACE-Step/ACE-Step-1.5-Text2Music/
  ```
- **预期结果**: 应看到 transformer/、vae/、text_encoder/ 等目录
- **状态**: ⬜ 待确认

## ✅ 已确认的项目

（无）

## 如何确认

1. 查看每个检查项的详情
2. 确认无误后，在对应项标记 `[x]`
3. 全部确认后，继续下一步
```

### 2. 检查点标记规范

在代码和日志中明确标记：

```python
# 🔍 USER_CHECK: 请确认以下输出是否符合预期
print("模型加载完成，参数数量:", num_params)
# 预期: 约 1.5B 参数
# 如果差异较大，请检查模型文件完整性

# 🔍 USER_CHECK: 依赖冲突
if conflict_detected:
    print("⚠️  发现依赖冲突，详情见 checkpoints/conflict_report.txt")
    print("   请确认是否接受当前版本")
```

## Skill 特定规范

### diffsynth-environment

**必须保存的脚本**:
- `scripts/check_models.sh` - 模型预检查
- `scripts/test_diffsynth_import.py` - DiffSynth 导入测试
- `scripts/test_target_import.py` - 目标库导入测试

**必须保存的日志**:
- 环境创建日志
- 依赖安装日志（分批）
- 模型预检查日志
- 验证脚本运行日志

**检查点**:
- [ ] 模型是否已正确下载
- [ ] 软链接是否创建成功
- [ ] 依赖冲突是否可接受
- [ ] DiffSynth 验证脚本是否通过
- [ ] 目标库验证脚本是否通过

### diffsynth-model-code

**必须保存的脚本**:
- `scripts/test_forward_original.py` - 原始模型 forward 测试
- `scripts/test_forward_diffsynth.py` - DiffSynth 模型 forward 测试
- `scripts/compare_forward.py` - forward 输出对比

**必须保存的日志**:
- 模型文件创建/修改日志
- converter 编写日志
- forward 测试执行日志
- 一致性对比结果

**检查点**:
- [ ] 模型文件结构是否正确
- [ ] state_dict_converter 是否正确映射 key
- [ ] model_hash 是否计算正确
- [ ] forward 输出是否与原始模型一致（数值对比）
- [ ] 注册条目是否正确

### diffsynth-pipeline

**必须保存的脚本**:
- `scripts/test_pipeline.py` - Pipeline 测试
- `scripts/test_e2e_original.py` - 原库端到端测试
- `scripts/test_e2e_diffsynth.py` - DiffSynth 端到端测试
- `scripts/compare_outputs.py` - 输出对比

**必须保存的日志**:
- Pipeline 文件创建/更新日志
- 推理示例脚本运行日志
- 端到端测试执行日志
- 输出文件对比结果

**检查点**:
- [ ] Pipeline 是否能正常初始化
- [ ] 推理示例是否能正常运行
- [ ] 输出文件是否正确生成
- [ ] 与原库输出是否一致（图像/音频对比）
- [ ] 低显存版本是否正常工作

### diffsynth-testing

**必须保存的脚本**:
- `scripts/run_all_tests.py` - 测试套件运行器
- `scripts/validate_outputs.py` - 输出验证

**必须保存的日志**:
- 每个推理脚本的执行日志
- 输出文件列表和元数据
- 一致性测试结果汇总

**检查点**:
- [ ] 所有推理脚本是否通过
- [ ] 输出文件质量是否可接受（需人工查看）
- [ ] 训练验证脚本是否通过（如有）
- [ ] 一致性测试是否在误差范围内

## 快速查看命令

```bash
# 查看最新执行日志
cd packages/{model-name}/.sisyphus/execution-logs/latest

# 查看用户检查清单
cat user-checks.md

# 查看执行清单
jq . manifest.json

# 查看所有检查点
ls -la checkpoints/

# 查看所有脚本
ls -la scripts/

# 查看所有输出
ls -la outputs/
```
