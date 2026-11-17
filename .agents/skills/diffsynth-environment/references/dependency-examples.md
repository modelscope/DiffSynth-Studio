# 依赖对比实例

## 实际案例：ACE-Step vs DiffSynth

### DiffSynth 核心依赖（11 个包）

`pip install -e .` 只安装这些：

```
torch>=2.0.0
torchvision
transformers
imageio[ffmpeg]
safetensors
einops
modelscope
ftfy
pandas
accelerate
peft
```

### DiffSynth 可选依赖组

基础安装**不含**以下依赖，需要时用 `pip install -e ".[audio,quant]"` 这样的写法追加：

| 组 | 包 | 何时需要 |
|---|---|---|
| `audio` | `av`、`torchaudio`、`torchcodec`、`librosa` | 音频/音视频模型（ACE-Step、MiniMax-Music3 等） |
| `quant` | `bitsandbytes`、`comfy-kitchen`、`torchao>=0.16` | 加载量化权重或做在线量化 |
| `training` | `deepspeed` | 分布式大规模训练 |
| `logger` | `tensorboard`、`swanlab`、`wandb` | 训练日志记录器 |
| `npu` / `npu_aarch64` | `torch-npu` 等 | Ascend NPU 设备 |
| `infiniteyou` / `ses` / `nexusgen` | 各自专属包 | 特定模型 |
| `all` | 除「特定模型依赖」外的全部 | — |

**判断依赖是否需要装时，先确认它是否已在上面两份清单中。** 注意 `sentencepiece`、`protobuf`、`datasets` **不在** DiffSynth 依赖中（曾经有，已移除），如目标库需要则必须显式安装。

### ACE-Step 依赖（50+ 个包，含平台特定）

```
# 平台特定 PyTorch（4 套，只装当前平台需要的）
torch==2.10.0+cu128; linux x86_64
torch==2.10.0+cu130; linux aarch64
torch==2.7.1+cu128; Windows
torch>=2.9.1; macOS arm64

# 与 DiffSynth 重叠的包（已有，跳过）
transformers, accelerate, peft, einops, safetensors, modelscope

# ACE-Step 独有，需要额外安装
diffusers>=0.37.0
gradio==6.2.0
matplotlib>=3.7.5
scipy>=1.10.1
soundfile>=0.13.1
loguru>=0.7.3
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
numba>=0.63.1
vector-quantize-pytorch>=1.27.15
torchcodec>=0.9.1
torchao>=0.16.0
toml
typer-slim>=0.21.1

# 平台特定，按需安装
flash-attn; linux x86_64（从 wheel 装）
mlx>=0.25.2; macOS arm64
mlx-lm>=0.20.0; macOS arm64

# 训练依赖（如不需要训练可跳过）
peft>=0.18.0（DiffSynth 已有）
lycoris-lora
lightning>=2.0.0
tensorboard>=2.0.0
```

### 实际安装建议（Linux x86_64）

**第 1 批：核心推理依赖**
```bash
pip install diffusers soundfile scipy vector-quantize-pytorch loguru toml typer-slim
```

**第 2 批：编译密集型 + DiffSynth 可选依赖组**
```bash
pip install flash-attn --no-build-isolation
pip install -e ".[audio,quant]"   # 覆盖 torchaudio/torchcodec/librosa/av + torchao 等
```

**第 3 批：可选（UI/API，验证脚本不需要则跳过）**
```bash
pip install gradio fastapi uvicorn numba
```

**不需要单独安装的**：
- PyTorch 相关：conda 克隆源环境时已自带，不要重装
- transformers、accelerate、peft、einops、safetensors、modelscope：DiffSynth 核心依赖已有
- torchaudio、torchcodec、librosa、av、torchao、bitsandbytes：由 DiffSynth 的 `[audio]` / `[quant]` extras 提供，用 extras 装，不要单独指定版本
- mlx、mlx-lm：仅 macOS，Linux 不需要
- lycoris-lora、lightning、tensorboard：训练用，如不跑训练可跳过

## 冲突处理示例

### 场景：目标库要求 `gradio==6.2.0`

DiffSynth 不依赖 gradio → 直接安装，无冲突。

### 场景：目标库要求 `transformers>=4.51.0,<4.58.0`

DiffSynth 已有 `transformers 4.52.0` → 在范围内，跳过。

### 场景：目标库要求 `diffusers>=0.37.0`

DiffSynth 没有 diffusers → 安装。注意 diffusers 和 DiffSynth 是不同的 diffusion 框架，共存无问题。

### 场景：目标库要求 `torch==2.7.1+cu128` 而当前环境是 `torch==2.10.0+cu126`

**不要降级**。CUDA 12.6 和 12.8 兼容，2.10.0 > 2.7.1，通常可以正常工作。如果目标库因版本不匹配报错，再考虑调整。

## 双向验证记录模板

```bash
# Step 7 后：DiffSynth OK 基线
pip list > ${EXEC_LOG_DIR}/checkpoints/pip_list_diffsynth_ok.txt

# Step 9 每批安装后
pip list > ${EXEC_LOG_DIR}/checkpoints/pip_list_after_batch_1.txt
diff ${EXEC_LOG_DIR}/checkpoints/pip_list_diffsynth_ok.txt ${EXEC_LOG_DIR}/checkpoints/pip_list_after_batch_1.txt

# Step 12 后：最终环境
pip list > ${EXEC_LOG_DIR}/checkpoints/pip_list_final.txt
diff ${EXEC_LOG_DIR}/checkpoints/pip_list_diffsynth_ok.txt ${EXEC_LOG_DIR}/checkpoints/pip_list_final.txt
```

对比结果记录哪些包是新装的、哪些版本变化了，方便 Step 12 失败时回滚。
