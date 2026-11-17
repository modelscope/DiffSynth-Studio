# 蓝图报告格式

蓝图报告是下游 skill 的数据源。所有信息必须具体、可操作，不模糊描述。

输出到 `packages/{model-name}/.sisyphus/integration-blueprints/{model-name}-blueprint.md`。

## 固定结构

以下为蓝图报告的标准模板，生成时应严格按照此格式组织。

---

# {Model Name} 接入蓝图

## 基本信息

| 字段 | 值 |
|------|---|
| 模型名称 | {name} |
| 目标库路径 | {target_path} |
| **整体接入类型** | new_series / version_upgrade |
| 判断依据 | {说明：是否已有类似架构的模型系列？新模型的 `__init__`/`forward` 与已有组件是否相同？} |
| 模型系列名（series） | {snake_case_name}，用于文件命名（如 z_image、qwen_image） |
| Pipeline 类名 | {SeriesName}Pipeline |
| **Conda 环境名称** | `{model_name}-diffsynth`（如 `ace-step-diffsynth`） |
| Clone 源环境 | `{env_clone_source}`（从 config.yaml 读取，如 `diffsynth`） |

## 1. 依赖分析（供 diffsynth-environment 使用）

### 1.0 LLM/推理后端分析（如果适用）

如果目标库使用了外部推理引擎（如 vLLM、TGI 等），必须分析：

| 字段 | 值 |
|------|-----|
| 推理引擎名称 | 如 nano-vllm、vLLM |
| 后端选项 | vLLM / PyTorch / MLX |
| 默认后端 | 如 vLLM |
| Fallback 机制 | 如 vLLM 不可用时自动降级到 PyTorch |
| DiffSynth 默认后端 | 如 PyTorch（transformers 直接加载） |
| 可选后端 | 如 nano-vllm（标记为可选依赖） |
| 本地包路径 | 如 `acestep/third_parts/nano-vllm/` |

**结论**：明确说明哪些后端是必须的，哪些是可选的。

### 1.1 依赖差异

| 包名 | 目标库版本 | DiffSynth 版本 | 处理方式 | 备注 |
|------|-----------|---------------|---------|------|
| {package} | {version} | {version} | 安装/跳过/冲突 | 必须/可选/不需要 |

### 1.2 依赖分类

**必须安装**：

| 包名 | 版本 | 用途 |
|------|------|------|
| ... | ... | ... |

**可选安装**：

| 包名 | 版本 | 用途 | 安装条件 |
|------|------|------|---------|
| ... | ... | ... | 如 CUDA 平台 + 需要 triton |

**不需要安装**：

| 包名 | 原因 |
|------|------|
| ... | 如 UI 组件、API 服务、训练框架 |

### 1.3 平台特定依赖

- Linux x86_64 需要：...
- macOS 需要：...
- Windows 需要：...

### 1.4 验证脚本

- **DiffSynth 验证脚本**：`{diffsynth_root}/examples/{series}/model_inference/{script}.py`
- **目标库验证脚本**：`{target_path}/{example_script}.py`
- **目标库验证脚本所需模型**：
  - 模型名称：...
  - 下载方式：`{download_command}` 或 `从 {url} 下载`

## 2. 模型组件清单（供 diffsynth-model-code 使用）

### 2.0 模型耦合分析（关键）

**目标库的模型代码往往耦合了多种职责**。必须分析耦合情况，为后续拆分提供依据。

**耦合点清单**：

| 耦合类型 | 文件 | 类/方法 | 拆分建议 |
|---------|------|---------|---------|
| 训练+推理耦合 | `{path}` | `{Class}.forward()` | 分离 forward 和 compute_loss |
| 条件+DiT 耦合 | `{path}` | `{Class}.forward()` | 提取 TextEncoder 为独立模型 |
| VAE+DiT 耦合 | `{path}` | `{Class}.forward()` | 将 VAE 移到 Pipeline 中 |
| 采样+模型耦合 | `{path}` | `{Class}.generate()` | 将采样逻辑移到 Pipeline |

**拆分后的模型组件清单**：

| 目标库组件 | DiffSynth 模型 | 文件 | 说明 |
|-----------|---------------|------|------|
| `{OriginalClass}` | `{Series}{Component}` | `{series}_{component}.py` | 职责说明 |

### 2.0.1 外部依赖组件分析（关键）

**DiffSynth 依赖原则**：
- 模型代码不可 import diffusers：所有 diffusers 组件必须重构为 DiffSynth 风格的独立实现（但 diffusers 包可以安装到 conda 环境供目标库使用）
- 可以依赖 transformers：但需编写接入包装类（参考 `qwen_image_text_encoder.py`）

| 组件 | 代码来源 | 处理方式 | 工作量评估 | 状态 |
|------|---------|---------|-----------|------|
| DiT | 目标库自有源码 | 直接接入 | 低 | 就绪 |
| VAE | diffusers.AutoencoderKL | 重构 diffusers 源码 | 高（需复制 ~2000 行） | 需用户确认 |
| TextEncoder | transformers.Qwen2_5_VLModel | 编写接入包装类 | 中（~200 行包装代码） | 就绪 |

**需要重构的 diffusers 组件**：
- 组件名称：如 `AutoencoderKL`
- diffusers 源码路径：如 `src/diffusers/models/autoencoders/autoencoder_kl.py`
- 需要复制的文件列表：
  - `autoencoder_kl.py` → `diffsynth/models/{series}_vae.py`
  - 相关依赖文件（如 `activations.py`、`attention.py` 等）
- 许可证：Apache 2.0（需保留原始许可证头，但不要添加代码来源注释如 "copied from"）
- 重构要点：
  - 移除 diffusers 内部依赖（如 `diffusers.models.modeling_utils.ModelMixin`）
  - 改为继承 `torch.nn.Module`
  - 适配 DiffSynth 的权重加载方式（`state_dict_converter`）

**需要包装的 transformers 组件**：
- 组件名称：如 `Qwen2_5_VLModel`
- transformers 版本要求：如 `>=4.54.0`
- 包装类实现要点（参考 `qwen_image_text_encoder.py`）：
  - 在 `__init__` 中 lazy import transformers 类
  - 手动构建 config dict（或从预训练配置加载）
  - 包装为 `torch.nn.Module` 子类
  - 实现符合 DiffSynth 需求的 `forward` 方法

### 2.1 组件概览

| 组件 | 目标库类名 | 目标库文件 | **组件接入方式** | 代码来源 | 需要 converter |
|------|-----------|-----------|-----------------|---------|---------------|
| DiT | {ClassName} | {path} | 自有源码 / diffusers 重构 / transformers 包装 / 完全复用 / 结构变化 | 目标库自有 | 是/否 |
| VAE | {ClassName} | {path} | 自有源码 / diffusers 重构 / transformers 包装 / 完全复用 / 结构变化 | diffusers 需重构 | 是 |
| TextEncoder | {ClassName} | {path} | 自有源码 / diffusers 重构 / transformers 包装 / 完全复用 / 结构变化 | transformers 需包装 | 否 |

**说明**：
- **组件接入方式**：该组件的处理方式（与 model-code skill 保持一致）
  - `自有源码` — 复制目标库代码，仅替换 attention 和 gradient checkpointing
  - `diffusers 重构` — 从 diffusers 源码重构为独立实现
  - `transformers 包装` — 编写包装类接入 transformers 模型
  - `完全复用` — 复用 DiffSynth 已有组件（`__init__`/`forward` 签名相同）
  - `结构变化` — 修改现有模型文件（version_upgrade 时签名有变化）
- **代码来源**：该组件的代码来自哪里（目标库自有 / diffusers / transformers）
- **整体接入类型**（见基本信息）：针对整个模型系列的判断（new_series / version_upgrade）
  - 如果整体是 `new_series` → 组件接入方式通常是 `自有源码`/`diffusers 重构`/`transformers 包装`
  - 如果整体是 `version_upgrade` → 组件接入方式可能是 `完全复用` 或 `结构变化`

### 2.2 每个组件的详细信息

**{Component Name}**：
- **代码来源**：目标库自有 / diffusers 需重构 / transformers 需包装
- `__init__` 签名：`def __init__(self, ...)`
- `forward` 签名：`def forward(self, ...)`
- 权重 key 模式：`{prefix}.layers.{i}.weight` 示例
- 特殊依赖：flash-attn / xformers / 等
- 与 DiffSynth 已有组件的差异：...（version_upgrade 时填写）
- **外部依赖处理**（如适用）：
  - diffusers 组件：源文件路径 → 目标文件路径，需要修改的点
  - transformers 组件：包装类实现要点，config 构建方式

### 2.2.1 一致性测试规划

**所有组件都必须执行两种测试，永不跳过**（详见 [model-code SKILL.md](../../diffsynth-model-code/SKILL.md) 的「8. 一致性验证」，含 8a ModelPool 加载测试与 8b 统一测试脚本）。

| 场景 | Forward 一致性测试 | ModelConfig + ModelPool 加载测试 | 说明 |
|------|-------------------|-------------------|------|
| 新建模型文件（new_series） | 必须 | 必须 | 全新代码，需验证 forward 输出和完整注册链 |
| 修改现有模型文件（version_upgrade） | 必须 | 必须 | 代码有变更，需验证 forward 输出和完整注册链 |
| 仅修改 `__init__` 参数（version_upgrade，forward 无变更） | 必须 | 必须 | 需验证参数传递和输出一致性 |
| 完全复用已有文件（仅注册新 hash） | 必须 | 必须 | 验证不同 checkpoint 的加载和 forward 输出 |

**输出格式**：

| 组件 | 组件接入方式 | Forward 测试 | 加载测试 | 说明 |
|------|-------------|-------------|---------|------|
| DiT | 自有源码 | 必须 | 必须 | 全新代码，需验证 forward 输出和完整注册链 |
| VAE | 完全复用 | 必须 | 必须 | 复用已有类，验证不同 checkpoint 的加载和 forward |
| TextEncoder | 结构变化 | 必须 | 必须 | 修改现有代码，验证修改后的 forward 逻辑 |

### 2.3 权重文件映射

| 目标库权重文件 | 对应组件 | DiffSynth model_name |
|---------------|---------|---------------------|
| `transformer/*.safetensors` | DiT | `{series}_dit` |
| `vae/diffusion_pytorch_model.safetensors` | VAE | `{series}_vae` |

## 3. Pipeline 功能规划（供 diffsynth-pipeline 使用）

按从简单到难排序。每个功能对应一次 diffsynth-pipeline skill 调用。

### 功能 1：{功能名称}（优先级 1 — 最简单）

| 字段 | 值 |
|------|---|
| 功能描述 | 如 Text-to-Image |
| 可行性评估 | 高度可行 / 中等可行 / 不建议 |
| 原库 example | `{target_path}/examples/xxx.py` |
| 输入 | prompt, seed, num_steps, ... |
| 输出 | 图像 / 视频 / 音频 |
| 需要的模型组件 | DiT, VAE, TextEncoder |
| 需要的模型下载 | 模型 ID + origin_file_pattern |
| 依赖前置功能 | 无 |
| 训练支持 | 全量 / LoRA / 无 |
| 训练类型（如有） | Text-to-X / Image-to-X / 微调 / 从头训练 |
| 训练数据集格式（如有） | 如 jsonl 配对数据、单模态目录结构 |
| 训练特殊输入（如有） | 如 InputAudioEmbedder、InputImageEmbedder 需求 |
| 特殊处理 | 如 ControlNet 输入、多模态编码 |

### 功能 2：{功能名称}（优先级 2）

...

### 功能 N：{功能名称}（优先级 N — 最复杂）

...

## 3.1 Scheduler 信息（供 diffsynth-pipeline 使用）

**必须在蓝图中明确目标模型的调度策略。** 这是 Pipeline 接入的关键信息。

| 字段 | 值 |
|------|---|
| 调度策略类型 | Flow Matching / DDPM / DDIM / Euler / 其他 |
| 目标库 Scheduler 类名 | `{ClassName}` |
| 目标库 Scheduler 文件路径 | `{path}` |
| 是否兼容已有 FlowMatchScheduler template | 是（template 名称：`{name}`）/ 否（需新增方法） |
| 调度公式 | 如果是 Flow Matching，列出 sigma 计算公式；否则简述采样策略 |

**模型特异参数**（如果有）：

| 参数名 | 类型 | 默认值 | 用途 | 推理时是否暴露 |
|--------|------|--------|------|--------------|
| `dynamic_shift_len` | int | 4096 | 序列长度，影响 sigma_shift | 是 |
| `terminal` | float | 0.1 | terminal shift | 是 |
| `special_case` | str | None | 调度方案标记 | 是 |

**特殊调度方案**（如有 distilled、turbo 等变体）：

| 方案名称 | `special_case` 值 | 固定 sigmas | Step 数 | 说明 |
|---------|------------------|------------|--------|------|
| 默认 | `None` | 动态计算 | 用户指定 | 标准推理 |
| Distilled Stage 2 | `"stage2"` | `[0.909, 0.725, 0.422]` | 3 | 蒸馏模型 |

**训练时 timestep 策略**（如果有）：
- 是否有特殊的 timestep 权重计算？如 LTX-2 的 `set_training_weight`
- 训练时 `set_timesteps` 是否传入额外参数？

## 4. 文档信息（供 diffsynth-docs 使用）

| 字段 | 值 |
|------|---|
| **整体接入类型** | new_series / version_upgrade（继承自基本信息） |
| **功能列表** | 见第 3 节 Pipeline 功能规划 |
| **Pipeline 类名** | 见基本信息 |
| **与已有版本的差异** | （仅 version_upgrade 时填写，new_series 时留空） |

**整体接入类型说明**：
- `new_series`：全新模型家族，需要创建完整的文档结构
- `version_upgrade`：已有系列的新版本，只需添加变体说明到现有文档

## 4.1 功能-脚本映射（供 diffsynth-environment 和 diffsynth-testing 使用）

为每个功能创建独立的可运行验证脚本映射。

### 功能与脚本映射总表

| 功能 | 目标库Example | 目标库验证脚本 | DiffSynth推理脚本 | 优先级 |
|------|--------------|---------------|------------------|--------|
| {功能1} | `{target_path}/examples/...` | `test_env_{feature1}.py` | `examples/{series}/model_inference/{feature1}.py` | P1 |
| {功能2} | `{target_path}/examples/...` | `test_env_{feature2}.py` | `examples/{series}/model_inference/{feature2}.py` | P2 |

> `{series}` 是 `examples/` 下的目录名，不一定等于模型系列名。例如 Wan 系列的目录是 `examples/wanvideo/`，`examples/wan/` 并不存在。以实际目录名为准。

### 每个功能的验证脚本说明

#### 功能 1: `{feature_name}`

**目标库验证脚本**：`test_env_{feature}.py`
- **运行方式**：`cd {target_path} && python test_env_{feature}.py`
- **模型要求**：需要哪些模型（model_id）
- **输入要求**：是否需要额外输入文件（如参考音频、源音频）
- **对应原库入口**：`{target_path}/{inference_script}.py:{function_name}()` with `{task_type="..."}`
- **关键参数**：生成该功能需要的核心参数
- **特殊说明**：如跳过LM、需要TimbreEncoder等

#### 功能 2: `{feature_name}`

...

### 功能依赖关系

```
{Feature1} (P1) ──┬── {Feature2} (P2)
                  ├── {Feature3} (P3)
                  └── {Feature4} (P4)
```

### 推理一致性测试流程

1. **目标库生成基准**：运行 `test_env_{feature}.py`，输出到 `./output/test_{feature}/`
2. **DiffSynth 生成对比**：运行 `examples/{series}/model_inference/{feature}.py`，输出到 `./output/test_{feature}/`
3. **对比输出**：使用 `torch.allclose()` 对比音频/图像，允许一定误差

### 4.2 低显存推理信息（供 diffsynth-pipeline-lowvram 使用）

**基于第 3 节 Pipeline 功能规划和第 2 节模型组件清单，分析低显存优化需求。**

| 字段 | 值 |
|------|---|
| Pipeline 类名 | 见基本信息 |
| 可低显存优化的功能 | 见第 3 节功能列表（标记哪些功能适合低显存优化） |
| 需要 module_map 注册的组件 | 见第 2 节组件列表（DiT/VAE/TextEncoder 等，按组件大小排序） |
| 目标库显存管理参考 | `{target_path}/` 下的显存优化实现（如 `offload`、`cpu_offload`、`sequential_cpu_offload`） |
| 预估最低显存需求 | 如 ~6GB（DiT offload + VAE sequential） |

**组件显存占用分析**（如适用）：

| 组件 | 参数量 | 预估显存（FP16） | 是否支持 offload | 优化建议 |
|------|--------|-----------------|-----------------|---------|
| DiT | {N}M / {N}B | ~X GB | 是/否 | 如 sequential CPU offload |
| VAE | {N}M | ~X GB | 是/否 | 如 sequential loading |
| TextEncoder | {N}M | ~X GB | 是/否 | 通常较小，无需 offload |

### 4.3 训练信息（供 diffsynth-pipeline-training 使用）

**基于第 3 节 Pipeline 功能规划，分析训练接入需求。**

| 字段 | 值 |
|------|---|
| 支持全量训练 | 是/否 |
| 支持 LoRA 训练 | 是/否 |
| 训练入口脚本 | `{target_path}/train_*.py`（如存在） |
| 训练数据集格式 | 如 `jsonl` 配对数据、`--train_data_dir` 目录结构 |
| 数据集配置要求 | 如需要 caption 文件、需要配对音频等 |
| Input Embedder 需求 | 如 `InputAudioEmbedder`、`InputImageEmbedder`（见第 2 节组件分析） |
| 训练时 Scheduler 特殊要求 | 见第 3.1 节「训练时 timestep 策略」 |
| 梯度检查点支持 | 是/否，目标库是否已有 |
| 混合精度训练 | 如 `fp16`、`bf16`、`fp8` |
| 预估训练显存需求 | 如 ~24GB（LoRA, batch=1, 512x512） |

**训练功能详情**（仅当第 3 节中某功能标注了训练支持时填写）：

#### 训练功能 1: `{feature_name}`

- **训练类型**：Text-to-X 微调 / Image-to-X 微调 / LoRA / 全量
- **需要的模型组件**：DiT, VAE（仅推理）, TextEncoder
- **数据集准备**：说明数据格式、目录结构、预处理脚本
- **训练参数**：learning rate, epochs, batch size, gradient accumulation 等
- **验证方式**：训练后运行推理脚本验证

## 5. 模型下载规划

**模型软链接结构**：
- `{target_path}/models/` 和 `{diffsynth_root}/models/` 是真实目录
- 每个模型 repo 分别软链接到 `~/.cache/modelscope/hub/models/{org}/{model}/`

| 模型 | model_id | 来源 | 软链接路径 | 对应组件 | 必需性 |
|------|---------|------|-----------|---------|--------|
| DiT 权重 | {modelscope_id} | ModelScope | `{target_path}/models/{org}/{model}/transformer/` | DiT | 必需 |
| VAE 权重 | {modelscope_id} | ModelScope | `{target_path}/models/{org}/{model}/vae/` | VAE | 必需 |
| TextEncoder | {modelscope_id} | ModelScope | `{target_path}/models/{org}/{model}/text_encoder/` | TextEncoder | 必需 |
| Tokenizer | {modelscope_id} | ModelScope | `{target_path}/models/{org}/{model}/tokenizer/` | — | 必需 |

## 6. 模型下载命令清单

**在环境准备之前执行**：以下命令用于下载模型到 ModelScope cache。下载可能耗时较长，建议提前执行。

### 6.1 必需模型下载命令

```bash
#!/bin/bash
# {model-name} 模型下载脚本
# 执行方式: bash download_models.sh

echo "开始下载 {model-name} 所需模型..."

# 模型1: {描述}
echo "下载 {model_id_1}..."
modelscope download --model {model_id_1} --local_dir ~/.cache/modelscope/hub/models/{model_id_1}

# 模型2: {描述}
echo "下载 {model_id_2}..."
modelscope download --model {model_id_2} --local_dir ~/.cache/modelscope/hub/models/{model_id_2}

echo "所有模型下载完成！"
echo "模型存储位置: ~/.cache/modelscope/hub/"
```

### 6.2 可选模型下载命令（如适用）

```bash
# 可选模型（用于高级功能）
# modelscope download --model {optional_model_id} --local_dir ~/.cache/modelscope/hub/models/{optional_model_id}
```

### 6.3 下载验证命令

```bash
# 验证模型是否已正确下载
ls -la ~/.cache/modelscope/hub/models/{model_id_1}/
ls -la ~/.cache/modelscope/hub/models/{model_id_2}/
```

### 6.4 环境准备阶段的软链接命令

**注意**：以下命令由 `diffsynth-environment` skill 自动执行，无需手动运行。

```bash
# 创建软链接（自动执行）
for model_rel in {model_id_1} {model_id_2}; do
    mkdir -p {target_path}/models/$(dirname ${model_rel})
    ln -sf ~/.cache/modelscope/hub/models/${model_rel} {target_path}/models/${model_rel}
    mkdir -p {diffsynth_root}/models/$(dirname ${model_rel})
    ln -sf ~/.cache/modelscope/hub/models/${model_rel} {diffsynth_root}/models/${model_rel}
done
```

### 6.5 模型信息汇总

| 序号 | 模型名称 | ModelScope ID | 预估大小 | 用途 | 优先级 |
|------|---------|--------------|---------|------|--------|
| 1 | {名称} | `{model_id}` | ~X GB | {用途} | P0-必需 |
| 2 | {名称} | `{model_id}` | ~Y GB | {用途} | P0-必需 |

**总预估下载大小**：约 XX GB
**预估下载时间**：约 XX 分钟（取决于网络速度）

## 7. 接入类型判断依据

### 7.1 整体接入类型判断（new_series vs version_upgrade）

**判断标准**：

| 场景 | 判断结果 | 判定依据 |
|------|---------|---------|
| **全新模型家族** | `new_series` | DiffSynth 中没有类似架构的模型系列（如 ACE-Step 是全新的音频生成模型） |
| **已有系列的变体** | `version_upgrade` | DiffSynth 中已有相同架构的模型，新模型只是参数变化或功能增强（如 Wan2.1-T2V-1.3B → 14B） |

**关键判断问题**：
1. DiffSynth 中是否已有相同架构的模型系列？
   - 无 → `new_series`
   - 有 → 继续判断
2. 新模型的 `__init__` 和 `forward` 签名与已有组件是否相同？
   - 完全相同 → 可能是 `version_upgrade`（完全复用）
   - 有变化 → 继续判断
3. 结构是否有变化（新增模块、参数变化）？
   - 无变化，仅参数变化 → `version_upgrade`
   - 有结构变化 → `version_upgrade`（需要修改模型文件）

### 7.2 组件接入方式判断

**组件接入方式**是针对每个组件的独立判断，与整体接入类型和代码来源共同决定最终处理方式。

**与 model-code skill 的对应关系**：

| 整体接入类型 | 组件情况 | 组件接入方式 | 说明 |
|-------------|---------|-------------|------|
| `new_series` | 目标库自有代码 | `自有源码` | 复制目标库代码，仅替换 attention 和 gradient checkpointing |
| `new_series` | 依赖 diffusers | `diffusers 重构` | 从 diffusers 源码重构为独立实现 |
| `new_series` | 依赖 transformers | `transformers 包装` | 编写包装类接入 transformers 模型 |
| `version_upgrade` | 签名完全相同 | `完全复用` | 复用已有组件，仅注册新 hash |
| `version_upgrade` | 签名有变化 | `结构变化` | 修改现有模型文件 |

**判断流程**（每个组件独立执行）：

```
Step 1: 整体接入类型是什么？
├── new_series → 进入 Step 2（判断代码来源）
└── version_upgrade → 检查签名
    ├── 完全相同 → 完全复用
    └── 有变化 → 结构变化

Step 2: 代码来源是什么？（new_series 时）
├── 目标库自有源码 → 自有源码
├── diffusers 依赖 → diffusers 重构
└── transformers 依赖 → transformers 包装
```

**示例**（Wan2.1-T2V-14B 相对于 Wan2.1-T2V-1.3B）：

| 组件 | 整体接入类型 | 代码来源 | 组件接入方式 | 说明 |
|------|-------------|---------|-------------|------|
| DiT | version_upgrade | 目标库自有 | `结构变化` | 参数量变化（30层→40层），`__init__` 参数变化 |
| VAE | version_upgrade | diffusers | `完全复用` | 完全复用 WanVAE，签名相同 |
| TextEncoder | version_upgrade | transformers | `transformers 包装` | Wan2.1-T2V-1.3B 没有 TextEncoder，14B 版本新增 |

### 7.3 最终接入方案确定

**与 model-code skill 的对应**：analyze-target 输出的"组件接入方式"直接对应 model-code 的决策路径

| 组件接入方式 | model-code 处理方案 | 模型文件操作 | Converter |
|-------------|-------------------|-------------|-----------|
| `自有源码` | 自有源码方案 | 创建 `{series}_{component}.py` | 按需编写 |
| `diffusers 重构` | diffusers 重构方案 | 创建 `{series}_{component}.py`（从 diffusers 重构） | 按需编写 |
| `transformers 包装` | transformers 包装方案 | 创建 `{series}_{component}.py`（包装类） | 通常不需要 |
| `完全复用` | 完全复用方案 | 无操作 | 复用已有 |
| `结构变化` | 结构变化方案 | 修改已有 `{series}_{component}.py` | 按需修改 |

**注意**：
- `完全复用` 情况下，不复用代码，只复用 DiffSynth 已有实现
- `结构变化` 情况下，基于已有组件的代码进行修改
