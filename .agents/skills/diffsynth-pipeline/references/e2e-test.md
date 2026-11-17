# 端到端一致性测试（E2E）

**定位**：Pipeline 接入的第一道验证。通过相同输入分别运行原库和 DiffSynth Pipeline，对比最终输出。

## 核心原则

**Pipeline 是一个完整的推理链路。** 数据流如下：

```
prompt + seed + cfg_scale + ...
    ↓
scheduler.set_timesteps(num_inference_steps)
    ↓
三字典准备: inputs_posi / inputs_nega / inputs_shared
    ↓
Unit 链执行（条件编码 → 噪声初始化 → 条件注入 → ...）
    ↓
Denoise loop: cfg_guided_model_fn(model_fn) + scheduler.step（循环 N 次）
    ↓
VAE/decoder 解码
    ↓
最终输出（图像/音频/视频/多模态）
```

E2E 测试不关心中间过程，只验证**整条链路的首尾是否一致**。

## 判定口径按接入类型分流

**开始测试前，先从蓝图报告读取 `接入类型`**，它决定数值对比是门槛还是仅记录：

| 接入类型 | 结构门槛（跑通 / 类型 / shape / dtype / 无 NaN-Inf） | 数值一致（`allclose(atol=1e-5)`） | 最终验收 |
|---------|:--:|:--:|:--|
| `new_series` | 必须通过 | **门槛**：不一致即失败，逐 Unit 定位修复 | 用户确认输出质量 |
| `version_upgrade` | 必须通过 | **非门槛**：照常运行、记录 `max_diff` / MSE，不判定失败 | **用户确认生成结果符合预期即通过** |

**为什么 version_upgrade 不以数值一致为门槛**：version_upgrade 复用同系列已有的 Pipeline 文件，该文件正在服务其他已验证通过的变体。为了让数值对齐而大幅改写共用逻辑，会把已经跑通的变体一起拖入未知状态，代价远大于收益。新变体的组件级正确性已由 `diffsynth-model-code` 的模型一致性测试保证，Pipeline 层只需确认结果符合预期。

**⚠️ version_upgrade 下严禁为了缩小 `max_diff` 而修改已有 Unit、`model_fn`、scheduler template 或 denoise loop。**

## 前提条件

- 目标库有可运行的推理示例脚本（或可以编写一个）
- 有可用的模型权重文件
- 如果目标库没有对应的推理脚本，可以跳过此测试，仅验证 DiffSynth Pipeline 能正常运行

---

## 严格约束（所有 Pipeline 都必须遵守）

### 1. 参数一致性

| 参数 | 说明 |
|------|------|
| `seed` | 必须相同（控制随机数生成器） |
| `num_inference_steps` | 必须相同（决定 scheduler timesteps 数量和值） |
| `cfg_scale` | 必须相同（影响 CFG 加权） |
| `prompt` / `negative_prompt` | 必须相同（文本输入） |
| `height` / `width` | 必须相同（输出尺寸） |

### 2. 调用接口

所有 Pipeline 都通过 `__call__` 方法暴露推理接口：

```python
output = pipe(
    prompt="same prompt as original",
    negative_prompt=None,
    cfg_scale=7.5,
    height=768,
    width=1280,
    seed=42,
    num_inference_steps=30,
)
```

**不需要**也不应该直接调用内部方法（`unit_runner`、`model_fn`、`step` 等）。

### 3. 输出类型适配

根据 Pipeline 的输出类型，采用不同的数值化方式：

| 输出类型 | 数值化方式 | 对比形状 |
|---------|-----------|---------|
| **图像** | `np.array(PIL.Image)` | `(H, W, C)` uint8 |
| **音频** | `np.ndarray` | `(channels, samples)` 或 `(samples,)` float32 |
| **视频帧** | `np.stack([np.array(f) for f in frames])` | `(T, H, W, C)` uint8 |
| **多模态** | 每种模态分别数值化 | 分别对比 |

---

## 测试流程

### Step 1: 在原库中运行，保存 golden reference

```bash
cd {target_path}
python {target_inference_script.py} \
    --seed 42 \
    --output packages/{model-name}/.sisyphus/tests/golden_output_{feature}/
```

**必须保存两样东西**：
1. 最终输出文件（按类型保存为 npy 或原始格式）
2. 推理时使用的完整参数，保存为 JSON：

```python
# 原库推理脚本末尾追加
import json
params = {
    "prompt": "...",
    "seed": 42,
    "num_inference_steps": 30,
    "cfg_scale": 7.5,
    "height": 768,
    "width": 1280,
}
with json.dump(params, open("params.json", "w"))
```

**如果原库没有推理脚本**：根据其 API 编写一个简单脚本，确保参数与 DiffSynth Pipeline 一致。

### Step 2: 在 DiffSynth 中运行 Pipeline

```python
import torch
import numpy as np
import json
from diffsynth.pipelines.{series} import {SeriesName}Pipeline, ModelConfig

# 1. 加载 golden reference 参数
with open("packages/{model-name}/.sisyphus/tests/golden_output_{feature}/params.json") as f:
    params = json.load(f)

# 2. 运行 DiffSynth Pipeline（使用相同参数）
pipe = {SeriesName}Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="...", origin_file_pattern="..."),
        # ... 所有组件
    ],
)
output = pipe(
    prompt=params["prompt"],
    seed=params["seed"],
    num_inference_steps=params["num_inference_steps"],
    cfg_scale=params.get("cfg_scale", 7.5),
    height=params.get("height", 1024),
    width=params.get("width", 1024),
)

# 3. 根据输出类型保存
# 单一输出：
np.save("packages/{model-name}/.sisyphus/tests/diffsynth_output_{feature}.npy", output)

# 多模态输出：
# np.save(".../video_frames.npy", np.stack([np.array(f) for f in output[0]]))
# np.save(".../audio.npy", output[1])
```

### Step 3: 验证输出一致性

**检查 1-4（结构门槛）：两种接入类型都必须通过。**

```python
import numpy as np

# 加载输出
golden = np.load("packages/{model-name}/.sisyphus/tests/golden_output_{feature}/output.npy")
diffsynth = np.load("packages/{model-name}/.sisyphus/tests/diffsynth_output_{feature}.npy")

# 1. 存在性检查
assert golden is not None and golden.size > 0, "原库输出为空"
assert diffsynth is not None and diffsynth.size > 0, "DiffSynth 输出为空"

# 2. 类型检查（dtype 一致）
assert golden.dtype == diffsynth.dtype, f"dtype 不一致: {golden.dtype} vs {diffsynth.dtype}"

# 3. 形状检查（shape 一致）
assert golden.shape == diffsynth.shape, f"shape 不一致: {golden.shape} vs {diffsynth.shape}"

# 4. 数值有效性（无 NaN / Inf）
assert not np.isnan(diffsynth).any(), "DiffSynth 输出包含 NaN"
assert not np.isinf(diffsynth).any(), "DiffSynth 输出包含 Inf"

# 数值指标（两种类型都计算，判定口径不同）
diff = golden.astype(np.float64) - diffsynth.astype(np.float64)
max_diff = float(np.abs(diff).max())
mse = float((diff ** 2).mean())
aligned = bool(np.allclose(golden, diffsynth, atol=1e-5))
```

**检查 5（数值判定）：按接入类型分流。**

#### new_series —— 数值必须完全一致

```python
assert aligned, f"数值不一致: max_diff={max_diff}"

print("✅ E2E 输出一致性验证通过")
print(f"  输出形状: {diffsynth.shape}")
print(f"  输出 dtype: {diffsynth.dtype}")
print(f"  max_diff: {max_diff}")
```

DiffSynth Pipeline 虽然经过重组（Unit 链、scheduler 替换等），但逻辑等价的实现应给出相同数值。**出现差异说明某个环节与目标库不等价**，需逐 Unit 定位，不要当作正常累积误差接受。按以下顺序排查：随机种子与噪声生成方式 → scheduler 的 sigmas/timesteps → 某个 Unit 的前处理（resize/normalize/VAE 编码）→ attention 后端差异（用 `DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch` 排除）。

`max_diff` 非 0 但小于 `atol` 时仍算通过；一旦断言失败，按上文的排查顺序定位不等价的环节，不要放宽阈值。

#### version_upgrade —— 数值只记录，不断言、不改 Pipeline

```python
print(f"ℹ️ version_upgrade: max_diff={max_diff}, mse={mse}, allclose(atol=1e-5)={aligned}")
print("   数值差异仅记录，不作为通过/失败判据；验收以生成结果是否符合预期为准")
```

**不得在此处断言数值一致，也不得回头修改 Pipeline 来缩小 `max_diff`。**

写入 `e2e_comparison_result.json` 时，`status` 由结构门槛决定，数值字段作为附加信息记录：

```json
{
  "integration_type": "version_upgrade",
  "status": "PASSED",
  "shape_match": true,
  "dtype_match": true,
  "no_nan_inf": true,
  "max_diff": 0.0312,
  "mse": 0.00041,
  "allclose_1e-5": false,
  "note": "数值差异仅记录；验收以生成结果是否符合预期为准"
}
```

`max_diff` 较大时的正确处理顺序：

1. 确认组件级一致性已在 `diffsynth-model-code` 阶段验证通过（新变体的模型 forward 与原库一致）
2. 记录差异与可能来源（原库版本差异、scheduler 特异参数、采样策略变化、attention 后端等）
3. 把输出文件交用户确认生成质量

**不是**回头改 Pipeline。

### Step 4: 保存执行留痕

```bash
EXEC_LOG_DIR="packages/{model-name}/.sisyphus/execution-logs/pipeline-{feature}"
mkdir -p ${EXEC_LOG_DIR}/{scripts,outputs,checkpoints}

# 保存所有测试脚本
cp {target_path}/{original_example_script}.py ${EXEC_LOG_DIR}/scripts/test_e2e_original.py
cp {diffsynth_root}/examples/{series}/model_inference/{feature}.py ${EXEC_LOG_DIR}/scripts/test_e2e_diffsynth.py

# 执行并记录日志
cd {target_path}
python ${EXEC_LOG_DIR}/scripts/test_e2e_original.py 2>&1 | tee ${EXEC_LOG_DIR}/outputs/e2e_original.log

cd {diffsynth_root}
python ${EXEC_LOG_DIR}/scripts/test_e2e_diffsynth.py 2>&1 | tee ${EXEC_LOG_DIR}/outputs/e2e_diffsynth.log

python ${EXEC_LOG_DIR}/scripts/compare_outputs.py 2>&1 | tee ${EXEC_LOG_DIR}/outputs/e2e_comparison.log
```

---

## 通过标准

**结构门槛 —— `new_series` 与 `version_upgrade` 都必须通过**：

| 检查项 | 要求 | 判定 |
|--------|------|------|
| 原库输出存在且非空 | `golden.size > 0` | ❌ 失败则原库无输出 |
| DiffSynth 输出存在且非空 | `diffsynth.size > 0` | ❌ 失败则 Pipeline 未产出结果 |
| 输出类型一致 | 原库输出什么类型，DiffSynth 也输出什么类型 | ❌ 不一致则失败 |
| 输出形状一致 | `golden.shape == diffsynth.shape` | ❌ 不一致则失败 |
| dtype 一致 | `golden.dtype == diffsynth.dtype` | ❌ 不一致则失败 |
| 无 NaN/Inf | `diffsynth` 中无数值异常 | ❌ 存在则失败 |

**数值门槛 —— 按接入类型分流**：

| 接入类型 | 要求 | 判定 |
|---------|------|------|
| `new_series` | 数值完全一致（数值相同或 `allclose(atol=1e-5)`） | ❌ 不一致则失败，逐 Unit 定位修复 |
| `version_upgrade` | 无 —— `max_diff` / MSE 仅记录 | ✅ 不因数值差异判失败 |

结构门槛（以及 `new_series` 的数值门槛）通过后，由用户检查输出文件（图像/音频/视频）确认质量。**`version_upgrade` 以用户确认「生成结果符合预期」为最终验收标准。**

---

## E2E 通过后的流程

E2E 通过后，**不需要再进行逐步测试**。Agent 直接执行代码 review：

1. 检查 Pipeline 文件结构是否符合模板
2. 检查每个 Unit 是否有目标库对应参考
3. 检查 Unit 链设计是否合理
4. 检查推理示例脚本是否正确
5. （`version_upgrade`）检查改动是否为最小增量：只新增了该变体所需的参数 / Unit，未改写已有 Unit、`model_fn`、scheduler template、denoise loop，也未改变已有变体的参数默认值与行为
6. 确认无遗留问题后，标记该功能接入完成

## E2E 失败后的流程

E2E 失败意味着结构性问题（输出为空、类型/形状不匹配、dtype 不一致、NaN/Inf）。检查 Unit 对应关系、参数传递、数据类型等，修复后重新运行 E2E 测试，直到通过。

**判定失败的范围按接入类型不同**：

| 接入类型 | 算失败 | 不算失败 |
|---------|--------|---------|
| `new_series` | 结构门槛任一项不通过；数值 `allclose(atol=1e-5)` 不成立 | — |
| `version_upgrade` | 结构门槛任一项不通过 | `max_diff` 超过 `atol` —— 记录后交用户确认生成质量 |

`version_upgrade` 下唯一的修复方向是结构性问题和新变体自身接线的错误（参数没透传、Unit 顺序错、组件加载错）。**数值差异本身不是待修的问题，不得通过改写共用 Pipeline 逻辑来"修复"它。**

## 跳过条件

- 目标库没有可用的推理脚本，且无法编写
- 目标库的推理流程与 DiffSynth 差异过大（如完全不同的 scheduler、不同的采样策略）
- 跳过时在输出中说明原因
