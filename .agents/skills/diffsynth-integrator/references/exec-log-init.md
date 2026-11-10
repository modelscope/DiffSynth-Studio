# 执行日志初始化规范

> **共享模板**：所有 diffsynth-* skill 的执行日志初始化均遵循此规范。

## 核心原则

每个 skill 执行的第一步，必须创建统一的执行日志目录结构，确保过程可追溯、结果可审查。

## 目录结构

```
packages/{model-name}/.sisyphus/execution-logs/
├── {timestamp}_{skill_suffix}/
│   ├── manifest.json            # 执行元数据
│   ├── scripts/                 # 保存的所有测试/验证脚本
│   ├── outputs/                 # 命令执行日志
│   ├── checkpoints/             # 中间检查点和环境状态
│   └── user-checks.md           # 需要人工确认的项目
└── latest -> {timestamp}_{skill_suffix}/  # 软链接到最新
```

## 初始化脚本模板

**每个 skill 执行时，将 `{skill_suffix}` 替换为对应的后缀名**：

| Skill | 后缀名 |
|-------|--------|
| diffsynth-analyze-target | `analyze` |
| diffsynth-environment | `environment` |
| diffsynth-model-code | `model_code` |
| diffsynth-pipeline | `pipeline_{feature}` |
| diffsynth-pipeline-lowvram | `lowvram` |
| diffsynth-pipeline-training | `pipeline_training` |
| diffsynth-style | `style` |
| diffsynth-testing | `testing` |
| diffsynth-docs | `docs` |
| diffsynth-pr | `pr` |
| diffsynth-pr-review | `pr_review` |

```bash
# 创建执行日志目录
export EXEC_LOG_DIR="packages/{model-name}/.sisyphus/execution-logs/$(date +%Y%m%d_%H%M%S)_{skill_suffix}"
mkdir -p ${EXEC_LOG_DIR}/{scripts,outputs,checkpoints}

# 创建执行清单
cat > ${EXEC_LOG_DIR}/manifest.json << EOF
{
  "skill_name": "{skill_name}",
  "model_name": "{model-name}",
  "timestamp": "$(date -Iseconds)",
  "execution_id": "{execution_prefix}_$(date +%Y%m%d_%H%M%S)",
  "steps": [],
  "user_checks": []
}
EOF

# 更新 latest 软链接
ln -sfn ${EXEC_LOG_DIR} packages/{model-name}/.sisyphus/execution-logs/latest
```

**变量替换表**：

| 变量 | analyze | environment | model_code | pipeline | lowvram | training | style | testing | docs | pr | pr_review |
|------|---------|-------------|------------|----------|---------|----------|-------|---------|------|-----|-----------|
| `{skill_name}` | diffsynth-analyze-target | diffsynth-environment | diffsynth-model-code | diffsynth-pipeline | diffsynth-pipeline-lowvram | diffsynth-pipeline-training | diffsynth-style | diffsynth-testing | diffsynth-docs | diffsynth-pr | diffsynth-pr-review |
| `{execution_prefix}` | analyze | env | model | pipeline | lowvram | training | style | test | docs | pr | pr_review |

## manifest.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `skill_name` | string | 当前执行的 skill 名称 |
| `model_name` | string | 模型名称（蛇形命名） |
| `timestamp` | string | ISO 8601 格式时间戳 |
| `execution_id` | string | 唯一执行标识符 |
| `steps` | array | 执行步骤记录（见下方格式） |
| `user_checks` | array | 需要用户确认的检查点 |

### steps 数组格式

```json
{
  "step_id": 1,
  "name": "步骤名称",
  "status": "completed|failed|skipped",
  "command": "执行的命令",
  "log_file": "outputs/step_01_xxx.log",
  "duration_ms": 45000,
  "exit_code": 0,
  "user_check_required": false
}
```

### user_checks 数组格式

```json
{
  "check_id": "check_01",
  "title": "检查点标题",
  "description": "检查点描述",
  "severity": "critical|warning|info",
  "file": "checkpoints/xxx.txt",
  "status": "pending|confirmed|rejected"
}
```

## USER_CHECK 生成规范

**关键规则**：使用 `cat >>`（追加）而非 `cat >`（覆盖），确保多个检查点不会互相覆盖。

```bash
cat >> ${EXEC_LOG_DIR}/user-checks.md << EOF
# 人工检查清单 - {skill_name}

## [CHECK-{编号}] {检查点标题}
- **严重程度**: {critical|warning|info}
- **描述**: {检查点描述}
- **查看方式**: \`{查看命令或文件路径}\`
- **通过标准**:
  - [ ] {标准 1}
  - [ ] {标准 2}
- **状态**: ⬜ 待确认
EOF
```

**首次生成时**（skill 的第一个检查点），先创建文件头：

```bash
cat > ${EXEC_LOG_DIR}/user-checks.md << EOF
# 人工检查清单 - {skill_name}

执行时间: $(date -Iseconds)
模型: {model-name}

---
EOF
```

**后续检查点**统一使用 `cat >>` 追加。

## 快速查看命令

```bash
# 查看最新执行日志目录
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

---

*本文档是共享参考，修改时需同步更新所有依赖此文档的 skill。*
