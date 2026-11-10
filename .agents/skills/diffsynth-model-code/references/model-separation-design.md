# DiffSynth-Studio 模型拆分设计原则

本文档指导如何将目标库中耦合的模型代码拆分为 DiffSynth 风格的清晰架构。

## 核心问题

许多目标库将训练逻辑、推理逻辑、条件准备、模型前向传播全部耦合在一个或少数几个文件中。这与 DiffSynth 的设计哲学相悖。

**DiffSynth 的设计原则**：
- **模型只负责 forward**：接收输入张量，返回输出张量
- **Pipeline 负责编排**：通过 PipelineUnit 链 + 三字典系统 + `cfg_guided_model_fn` 组织完整流程
- **条件准备是独立模型**：TextEncoder、AudioEncoder 等各自独立

> **Pipeline 架构的完整定义**见 [pipeline-template.md](../../diffsynth-pipeline/references/pipeline-template.md)。本文档聚焦于模型拆分，Pipeline 部分仅引用核心概念。

## 拆分原则

### 原则 1：DiT 只负责去噪预测

**DiT 的职责边界**：
```
输入：
  - hidden_states: 当前时间步的 latent（形状由具体模型决定）
  - timestep: 当前时间步
  - encoder_hidden_states: 已编码的条件（文本、音频、图像等）
  - 其他控制信号（如 ControlNet 输入、image_ids 等）

输出：
  - 预测的噪声/速度（形状与 hidden_states 一致）
```

**DiT 不应该做的事**：
- ❌ 文本编码（这是 TextEncoder 的职责）
- ❌ 音频编码（这是 AudioEncoder 的职责）
- ❌ VAE 编解码（这是 VAE 的职责）
- ❌ 条件准备逻辑（如 prompt 拼接、mask 生成）
- ❌ 采样循环（这是 Pipeline 的职责）
- ❌ 训练损失计算（这是训练脚本的职责）
- ❌ CFG 逻辑（这是 Pipeline 的 `cfg_guided_model_fn` 处理的）

**DiT forward 签名示例**：
```python
def forward(
    self,
    hidden_states: torch.Tensor,           # noise latent
    timestep: torch.LongTensor,            # 时间步
    encoder_hidden_states: torch.Tensor,   # 文本条件
    encoder_audio_states: torch.Tensor = None,  # 音频条件（可选）
    attention_mask: torch.Tensor = None,
    return_dict: bool = True,
) -> Union[Tuple, BaseModelOutput]:
    # 只执行去噪预测
    ...
    return (predicted_noise,)
```

### 原则 2：条件准备是独立模型

每个条件编码器应该是独立的 `nn.Module`：

| 条件类型 | 模型类 | 输入 | 输出 |
|---------|--------|------|------|
| 文本 | `{Series}TextEncoder` | input_ids, attention_mask | encoder_hidden_states |
| 音频参考 | `{Series}AudioEncoder` | audio_waveform | encoder_audio_states |
| 图像参考 | `{Series}ImageEncoder` | image_tensor | encoder_image_states |
| 音色 | `{Series}TimbreEncoder` | timbre_features | timbre_embeds |

**条件编码器的职责**：
- ✅ 接收原始输入（token IDs、波形、像素）
- ✅ 返回嵌入表示（hidden states）
- ✅ 处理 tokenizer（如果适用）
- ❌ 不参与去噪过程
- ❌ 不知道 DiT 的存在

**在 Pipeline 中的调用方式**：条件编码器不直接在 `__call__` 中调用，而是通过 PipelineUnit 链处理。Unit 负责加载模型、执行编码、将结果写入 `inputs_posi`/`inputs_nega`。详见 [pipeline-template.md](../../diffsynth-pipeline/references/pipeline-template.md) 中的 PipelineUnit 设计。

### 原则 3：VAE 独立编解码

VAE 是完全独立的组件：

```python
class {Series}VAE(nn.Module):
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """音频/图像 → latent"""
        ...
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """latent → 音频/图像"""
        ...
```

**VAE 不应该做的事**：
- ❌ 参与扩散过程
- ❌ 知道 DiT 的存在
- ❌ 处理条件信息

### 原则 4：Pipeline 负责完整流程

Pipeline 是唯一的编排者，使用三字典系统（`inputs_shared`/`inputs_posi`/`inputs_nega`）+ PipelineUnit 链 + `cfg_guided_model_fn`。

**完整 Pipeline 模板**见 [pipeline-template.md](../../diffsynth-pipeline/references/pipeline-template.md)。

**核心流程**：
```python
@torch.no_grad()
def __call__(self, prompt, negative_prompt="", cfg_scale=1.0, ...):
    # 1. Scheduler
    self.scheduler.set_timesteps(num_inference_steps, ...)
    
    # 2. 三字典输入
    inputs_posi = {"prompt": prompt}
    inputs_nega = {"negative_prompt": negative_prompt}
    inputs_shared = {"cfg_scale": cfg_scale, ...}
    
    # 3. Unit 链执行（条件编码在这里完成）
    for unit in self.units:
        inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
    
    # 4. Denoise loop（cfg_guided_model_fn 自动处理 CFG）
    self.load_models_to_device(self.in_iteration_models)
    models = {name: getattr(self, name) for name in self.in_iteration_models}
    for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
        noise_pred = self.cfg_guided_model_fn(
            self.model_fn, cfg_scale,
            inputs_shared, inputs_posi, inputs_nega,
            **models, timestep=timestep, progress_id=progress_id
        )
        inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)
    
    # 5. VAE 解码
    self.load_models_to_device(['vae'])
    output = self.vae.decode(inputs_shared["latents"])
    return self.vae_output_to_image(output)
```

**关键区别**：
- 条件编码不在 `__call__` 中直接调用，而是通过 PipelineUnit 处理
- 去噪循环通过 `cfg_guided_model_fn` 调用 `model_fn`，不直接调用 DiT
- `model_fn` 是独立函数（不是类方法），负责将 Unit 产出的参数传递给 DiT

## 拆分步骤

### 步骤 1：识别耦合点

分析目标库代码，找出以下耦合模式：

| 耦合模式 | 特征 | 拆分方法 |
|---------|------|---------|
| **训练+推理耦合** | forward 中包含损失计算 | 分离 forward（推理）和 compute_loss（训练） |
| **条件+DiT 耦合** | DiT 内部直接调用 tokenizer | 提取 TextEncoder 为独立模型 |
| **VAE+DiT 耦合** | DiT 内部调用 VAE decode | 将 VAE 移到 Pipeline 中 |
| **采样+模型耦合** | 模型类包含 sample/generate 方法 | 将采样逻辑移到 Pipeline |

### 步骤 2：定义模型边界

为每个组件定义清晰的接口：

```
┌─────────────────────────────────────────────────────────────┐
│                        Pipeline                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ TextEncoder │  │ AudioEncoder│  │    TimbreEncoder    │  │
│  │  (条件准备)  │  │  (条件准备)  │  │    (条件准备)       │  │
│  │  (通过 Unit  │  │  (通过 Unit  │  │    (通过 Unit       │  │
│  │   调用)      │  │   调用)      │  │     调用)           │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                     │             │
│         └────────────────┼─────────────────────┘             │
│                          │                                   │
│                   ┌──────▼──────┐                             │
│                   │     DiT     │                             │
│                   │  (去噪预测)  │                             │
│                   │  (通过       │                             │
│                   │   model_fn)  │                             │
│                   └──────┬──────┘                             │
│                          │                                   │
│                   ┌──────▼──────┐                             │
│                   │     VAE     │                             │
│                   │  (编解码)    │                             │
│                   └─────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

### 步骤 3：实现拆分

### 3a. 提取条件编码器

从目标库的耦合代码中：
```python
# 目标库（耦合）
class AceStepModel(nn.Module):
    def forward(self, prompt, audio, ...):
        # 在模型内部处理文本
        text_ids = self.tokenizer(prompt)
        text_embeds = self.text_model(text_ids)
        
        # 在模型内部处理音频
        audio_embeds = self.audio_processor(audio)
        
        # 然后才调用 DiT
        return self.dit(text_embeds, audio_embeds)
```

拆分为 DiffSynth 风格：
```python
# DiffSynth（分离）
class AceStepTextEncoder(nn.Module):
    def __init__(self):
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(...)
        self.model = AutoModel.from_pretrained(...)
    
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask).last_hidden_state

class AceStepAudioEncoder(nn.Module):
    def forward(self, audio_waveform):
        # 音频编码逻辑
        ...

class AceStepDiT(nn.Module):
    def forward(self, hidden_states, timestep, encoder_hidden_states, ...):
        # 只负责去噪预测
        ...
```

### 3b. 处理训练逻辑

训练逻辑不应该在模型文件中：

```python
# ❌ 错误：在模型文件中包含训练逻辑
class AceStepDiT(nn.Module):
    def forward(self, ..., labels=None):
        ...
        if labels is not None:
            loss = compute_loss(predicted, labels)
            return loss, predicted
        return predicted

# ✅ 正确：训练逻辑在训练脚本中
class AceStepDiT(nn.Module):
    def forward(self, hidden_states, timestep, ...):
        # 只返回预测结果
        return predicted_noise

# 训练脚本中
def training_step(batch, dit, scheduler):
    predicted = dit(batch.latent, batch.timestep, batch.conditions)
    loss = F.mse_loss(predicted, batch.target)
    return loss
```

## ACE-Step 拆分示例

### 当前耦合情况

ACE-Step 的 `modeling_acestep_v15_xl_turbo.py` 包含：
- `AceStepLyricEncoder` - 歌词编码（应该是独立模型）
- `AceStepDiTLayer` - DiT 层（应该是 DiT 的核心）
- `AceStepTimbreEncoder` - 音色编码（应该是独立模型）
- `AudioTokenDetokenizer` - 音频解码（应该是独立模型或 VAE 一部分）

### 拆分方案

| 目标库组件 | DiffSynth 模型 | 文件 |
|-----------|---------------|------|
| AceStepLyricEncoder | `AceStepTextEncoder` | `ace_step_text_encoder.py` |
| AceStepTimbreEncoder | `AceStepTimbreEncoder` | `ace_step_timbre_encoder.py` |
| AceStepDiTLayer + 包装 | `AceStepDiT` | `ace_step_dit.py` |
| AudioTokenDetokenizer | `AceStepAudioDecoder` | `ace_step_audio_decoder.py` |
| diffusers.AutoencoderOobleck | `AceStepVAE` | `ace_step_vae.py` |

### Pipeline 中的调用关系

Pipeline 通过 Unit 链调用各模型，不直接调用模型方法：

```python
class AceStepPipeline(BasePipeline):
    def __init__(self, ...):
        self.units = [
            AceStepUnit_ShapeChecker(),
            AceStepUnit_PromptEmbedder(),      # 调用 text_encoder
            AceStepUnit_TimbreEmbedder(),      # 调用 timbre_encoder
            AceStepUnit_NoiseInitializer(),
            # ... 更多 Unit
        ]
        self.model_fn = model_fn_ace_step       # 调用 dit
        self.in_iteration_models = ("dit",)
```

每个 Unit 负责加载对应模型、执行编码、将结果写入三字典。`model_fn` 负责将 Unit 产出的参数传递给 DiT。

## 检查清单

在拆分完成后，检查以下事项：

- [ ] DiT forward 只接收张量输入，不接收原始文本/音频
- [ ] 每个条件编码器独立可测试
- [ ] VAE 可以独立编解码，不依赖其他组件
- [ ] Pipeline 使用三字典系统 + Unit 链（不是直接调用模型）
- [ ] Pipeline 使用 `cfg_guided_model_fn` + `model_fn`（不是直接调用 DiT）
- [ ] 模型文件中没有训练损失计算逻辑
- [ ] 模型文件中没有采样循环逻辑
- [ ] 所有模型类都继承 `nn.Module`（不是 `ModelMixin` 等）

## 参考实现

- `diffsynth/models/flux2_dit.py` - 纯 DiT 实现
- `diffsynth/models/flux2_vae.py` - 独立 VAE
- `diffsynth/models/qwen_image_text_encoder.py` - transformers 包装
- `diffsynth/pipelines/flux2_image.py` - Pipeline 编排示例（三字典 + Unit 链 + `cfg_guided_model_fn`）
- `diffsynth/pipelines/z_image.py` - 复杂 Unit 链示例（8 个 Unit）
- [pipeline-template.md](../../diffsynth-pipeline/references/pipeline-template.md) - Pipeline 架构唯一定义
