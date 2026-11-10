# 推理示例脚本指南

**适用场景**：为 Pipeline 的每个功能创建推理示例脚本和低显存版本。

## 核心原则

推理示例脚本是用户了解和使用模型的入口。脚本必须**简洁、可直接运行、覆盖该功能的核心参数**。

## 文件结构

| 文件类型 | 目录 | 命名 |
|---------|------|------|
| 标准推理 | `examples/{series}/model_inference/` | `{ModelName}-{Feature}.py` |
| 低显存推理 | `examples/{series}/model_inference_low_vram/` | 与标准版本相同文件名 |

## 标准推理示例脚本

### 基本结构（15-25 行）

```python
from diffsynth.pipelines.{series} import {SeriesName}Pipeline, ModelConfig
import torch

pipe = {SeriesName}Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="{model_id}", origin_file_pattern="{dit_pattern}"),
        ModelConfig(model_id="{model_id}", origin_file_pattern="{text_encoder_pattern}"),
        ModelConfig(model_id="{model_id}", origin_file_pattern="{vae_pattern}"),
    ],
    tokenizer_config=ModelConfig(model_id="{model_id}", origin_file_pattern="{tokenizer_pattern}"),
)
prompt = "..."
image = pipe(prompt=prompt, seed=42, num_inference_steps=30)
image.save("output.jpg")
```

### 各组件 ModelConfig 填写规则

| 组件类型 | `origin_file_pattern` 典型值 | 说明 |
|---------|---------------------------|------|
| DiT/Transformer | `"transformer/diffusion_pytorch_model*.safetensors"` | `*` 匹配分片文件 |
| TextEncoder | `"text_encoder/model*.safetensors"` | `*` 匹配分片 |
| VAE | `"vae/diffusion_pytorch_model.safetensors"` | 单文件写完整路径 |
| Tokenizer/Processor | `"tokenizer/"` 或 `"processor/"` | 目录路径 |

**注意**：多个组件如果来自同一个 checkpoint 文件（如 LTX-2 的单个 `.safetensors` 同时包含 DiT、VAE、text encoder），它们的 `origin_file_pattern` 相同。

### 功能特定参数示例

根据功能不同，`pipe()` 调用需要传入对应的参数：

**Text-to-Image（最简）**：
```python
image = pipe(prompt=prompt, seed=42, num_inference_steps=30)
```

**Image-to-Video**：
```python
image = pipe(prompt=prompt, input_image=Image.open("input.jpg"),
             seed=42, num_inference_steps=30, height=480, width=832,
             num_frames=81, frame_rate=24)
```

**图像编辑**：
```python
image = pipe(prompt=prompt, edit_image=Image.open("input.jpg"),
             seed=42, num_inference_steps=30, edit_image_auto_resize=True)
```

**局部重采样**：
```python
image = pipe(prompt=prompt, edit_image=Image.open("input.jpg"),
             retake_regions=[{"x": 0.2, "y": 0.3, "width": 0.3, "height": 0.4}],
             seed=42, num_inference_steps=30)
```

### prompt 编写原则

- 使用能体现模型能力的描述性 prompt，不要只用 "a dog"
- 中文 prompt 优先（如果模型支持），展示中文能力
- 如果模型对某些参数敏感（如 cfg_scale、num_inference_steps），设置合理的默认值

## 低显存推理示例脚本

### 核心差异

低显存版本与标准版本**完全相同**，唯一的区别是：
1. 定义 `vram_config` 字典
2. 在每个 `ModelConfig`（组件模型）中传入 `**vram_config`
3. 在 `from_pretrained` 中传入 `vram_limit`

### 从标准版本生成低显存版本

**Step 1：在标准脚本开头添加 vram_config**

```python
vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
```

**Step 2：为每个组件 ModelConfig 传入 `**vram_config`**

```python
ModelConfig(model_id="...", origin_file_pattern="...", **vram_config),
```

**注意**：只有**组件模型**的 ModelConfig 需要 `**vram_config`，**tokenizer_config** 等非组件配置**不需要**。

**Step 3：在 `from_pretrained` 中添加 `vram_limit`**

```python
vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
```

### 低显存版本完整示例

```python
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
import torch

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Tongyi-MAI/Z-Image", origin_file_pattern="transformer/*.safetensors", **vram_config),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors", **vram_config),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
prompt = "..."
image = pipe(prompt=prompt, seed=42, num_inference_steps=30)
image.save("output.jpg")
```

## 创建流程

### Step 1：分析目标库推理脚本

从蓝图报告中获取目标库推理脚本的路径，了解：
- 模型初始化方式
- 输入数据准备（图像/视频加载方式）
- 推理参数（seed、steps、cfg_scale 等）
- 输出保存方式

### Step 2：创建标准推理示例

将目标库的推理逻辑映射为 DiffSynth Pipeline 调用：

| 目标库操作 | DiffSynth 等效操作 |
|-----------|-------------------|
| `model = SomeModel.from_pretrained(...)` | `pipe = {SeriesName}Pipeline.from_pretrained(...)` |
| `output = model(prompt, image, ...)` | `image = pipe(prompt=prompt, input_image=image, ...)` |
| `output.save("out.jpg")` | `image.save("out.jpg")` |

### Step 3：创建低显存版本

复制标准推理示例脚本，按照「从标准版本生成低显存版本」的步骤添加 VRAM 管理。

### Step 4：确认脚本可运行

在 `{diffsynth_root}` 下运行两个版本，确认：
- 标准版本：无报错，成功输出
- 低显存版本：无报错，成功输出（速度可能较慢，属正常）

## 常见注意事项

1. **`rand_device` 参数**：建议在 `pipe()` 中传入 `rand_device="cuda"` 以获得更快的噪声生成速度
2. **seed 设置**：使用固定 seed（如 42）确保可复现性
3. **输出文件名**：使用有意义的名称，如 `image_Z-Image.jpg`、`video_LTX-2-T2AV.mp4`
4. **多输出示例**：如果需要展示多种效果，可以在同一个脚本中多次调用 `pipe()`，每次使用不同 prompt 和 seed
5. **特殊输出格式**：视频输出使用 `save("output.mp4")`，音频输出可能需要额外的保存逻辑
