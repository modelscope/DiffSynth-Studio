# 训练代码模板

**适用场景**：Step 5 创建训练代码时使用。本文件包含完整的训练代码模板，每段代码都标注了「固定写法」和「需调整」部分。

> `.sh` 脚本模板、参数分析和需配置项决定方案见 [references/training-scripts.md](training-scripts.md)。

## InputEmbedder 编写指导

InputEmbedder 的功能是将输入数据（图像/视频/音频）通过 VAE 编码成 `input_latents`。**撰写 InputEmbedder 最好的方式是参考已有的 DiffSynth 实现**，有两个搜索方向：

1. **其他 Pipeline 中使用相同 VAE 的 InputEmbedder** — 学习 VAE 编码方式和参数传递模式
2. **本 Pipeline 中其他使用过 VAE 的 Unit** — 复用同一 Pipeline 中已有的 VAE 调用约定

### 方向一：参考其他 Pipeline 的 InputEmbedder

#### FLUX2 图像 InputEmbedder

文件：`diffsynth/pipelines/flux2_image.py`

```python
class Flux2Unit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: Flux2ImagePipeline, input_image, noise):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        pipe.load_models_to_device(['vae'])
        image = pipe.preprocess_image(input_image)
        input_latents = pipe.vae.encode(image)
        input_latents = rearrange(input_latents, "B C H W -> B (H W) C")
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents, "input_latents": input_latents}
```

**要点**：
- `input_image is None` 时返回 `{"latents": noise, "input_latents": None}`（推理时无输入图）
- 调用 `pipe.preprocess_image()` 预处理
- `pipe.vae.encode(image)` 编码，之后根据模型需要 reshape
- 训练模式返回 `input_latents`，推理模式先加噪再返回

#### LTX-2 视频 InputEmbedder

文件：`diffsynth/pipelines/ltx2_audio_video.py`

```python
class LTX2AudioVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "video_noise", "tiled", "tile_size_in_pixels", "tile_overlap_in_pixels"),
            output_params=("video_latents", "input_latents"),
            onload_model_names=("video_vae_encoder")
        )

    def process(self, pipe: LTX2AudioVideoPipeline, input_video, video_noise, tiled, tile_size_in_pixels, tile_overlap_in_pixels):
        if input_video is None or not pipe.scheduler.training:
            return {"video_latents": video_noise}
        else:
            pipe.load_models_to_device(self.onload_model_names)
            input_video = pipe.preprocess_video(input_video)
            input_latents = pipe.video_vae_encoder.encode(input_video, tiled, tile_size_in_pixels, tile_overlap_in_pixels).to(dtype=pipe.torch_dtype, device=pipe.device)
            return {"video_latents": input_latents, "input_latents": input_latents}
```

**要点**：
- 视频/音频等非图像 VAE 通常使用 `pipe.xxx_vae_encoder` 而非 `pipe.vae`
- Tile 参数（`tiled`, `tile_size_in_pixels`, `tile_overlap_in_pixels`）需要从 `input_params` 透传到 `encode`
- 推理模式（`not pipe.scheduler.training`）不执行 VAE 编码，直接返回 noise

#### LTX-2 音频 InputEmbedder

文件：`diffsynth/pipelines/ltx2_audio_video.py`

```python
class LTX2AudioVideoUnit_InputAudioEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_audio", "audio_noise"),
            output_params=("audio_latents", "audio_input_latents", "audio_positions", "audio_latent_shape"),
            onload_model_names=("audio_vae_encoder",)
        )

    def process(self, pipe: LTX2AudioVideoPipeline, input_audio, audio_noise):
        if input_audio is None or not pipe.scheduler.training:
            return {"audio_latents": audio_noise}
        else:
            input_audio, sample_rate = input_audio
            input_audio = convert_to_stereo(input_audio)
            pipe.load_models_to_device(self.onload_model_names)
            input_audio = pipe.audio_processor.waveform_to_mel(input_audio.unsqueeze(0), waveform_sample_rate=sample_rate).to(dtype=pipe.torch_dtype)
            audio_input_latents = pipe.audio_vae_encoder(input_audio)
            audio_latent_shape = AudioLatentShape.from_torch_shape(audio_input_latents.shape)
            audio_positions = pipe.audio_patchifier.get_patch_grid_bounds(audio_latent_shape, device=pipe.device)
            return {"audio_latents": audio_input_latents, "audio_input_latents": audio_input_latents, "audio_positions": audio_positions, "audio_latent_shape": audio_latent_shape}
```

**要点**：
- 音频需要先做波形 → Mel 频谱图的转换（`waveform_to_mel`）
- 返回多个额外字段：`audio_positions`（位置信息）、`audio_latent_shape`（形状信息）
- 音频 VAE 编码器直接用 `pipe.audio_vae_encoder(input_audio)` 调用（非 `.encode()` 方法）

### 方向二：参考本 Pipeline 中其他使用 VAE 的 Unit

当目标 Pipeline 中没有现成的 InputEmbedder 时，查看该 Pipeline 中是否有其他 Unit 使用了 VAE 编码。这些 Unit 的 VAE 调用方式（模型名称、编码方法签名、tile 参数）可以直接复用。

#### JoyAI-Image 示例：EditImageEmbedder → InputImageEmbedder

JoyAI-Image Pipeline 中，`EditImageEmbedder` 先于 `InputImageEmbedder` 实现，使用了 VAE 编码。分析 `EditImageEmbedder` 可以得到：

| 关键信息 | EditImageEmbedder 中的写法 | InputImageEmbedder 中的复用 |
|---------|---------------------------|---------------------------|
| VAE 模型 | `onload_model_names=("wan_video_vae",)` | 用同一个 `wan_video_vae` |
| 编码方法 | `pipe.vae.encode(images, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)` | 相同签名 |
| 预处理 | `pipe.preprocess_image(img).transpose(0, 1)` | 相同预处理 |
| Tile 参数 | `tiled`, `tile_size`, `tile_stride` | 相同参数名 |

由此推导出的 `InputImageEmbedder`：

```python
class JoyAIImageUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe: JoyAIImagePipeline, input_image, noise, tiled, tile_size, tile_stride):
        if input_image is None:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        if isinstance(input_image, Image.Image):
            input_image = [input_image]
        input_image = [pipe.preprocess_image(img).transpose(0, 1) for img in input_image]
        latents = pipe.vae.encode(input_image, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        input_latents = rearrange(latents, "(b n) c 1 h w -> b n c 1 h w", n=(len(input_image)))
        return {"latents": noise, "input_latents": input_latents}
```

**对比分析**：

| 项 | EditImageEmbedder | InputImageEmbedder | 原因 |
|---|---|---|---|
| `onload_model_names` | `("wan_video_vae",)` | `("vae",)` | Pipeline 中 `wan_video_vae` 注册为 `vae` 属性 |
| 编码方法 | `pipe.vae.encode(images, ...)` | 完全相同 | 同一 VAE，相同签名 |
| Tile 参数 | `tiled`, `tile_size`, `tile_stride` | 完全相同 | 复用同一套参数 |
| 预处理 | `.transpose(0, 1)` | 完全相同 | 该 VAE 需要 transposed 格式 |
| 返回值 | `ref_latents`, `num_items`, `is_multi_item` | `latents`, `input_latents` | 用途不同，字段名不同 |
| 训练/推理分支 | 无（仅推理侧使用） | 无 | JoyAI 的 InputEmbedder 不做训练分支判断 |

### InputEmbedder 标准实现模式

```
训练模式（training=True）：
  输入 = data["image"] / data["video"]
  → VAE 编码 → input_latents
  → 返回 {"latents": noise, "input_latents": input_latents, ...}

推理模式（training=False）：
  输入 = 用户提供的 latent/图像
  → 加噪 → latents
  → 返回 {"latents": latents}（不返回 input_latents）
```

**LTX-2 标准实现（权威参考）**：

```python
class LTX2AudioVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "video_noise", "tiled", "tile_size_in_pixels", "tile_overlap_in_pixels"),
            output_params=("video_latents", "input_latents"),
            onload_model_names=("video_vae_encoder")
        )

    def process(self, pipe: LTX2AudioVideoPipeline, input_video, video_noise, tiled, tile_size_in_pixels, tile_overlap_in_pixels):
        if input_video is None or not pipe.scheduler.training:
            return {"video_latents": video_noise}
        else:
            pipe.load_models_to_device(self.onload_model_names)
            input_video = pipe.preprocess_video(input_video)
            input_latents = pipe.video_vae_encoder.encode(input_video, tiled, tile_size_in_pixels, tile_overlap_in_pixels).to(dtype=pipe.torch_dtype, device=pipe.device)
            return {"video_latents": input_latents, "input_latents": input_latents}
```

**接入检查清单**：
- 推理模式（`training=False`）**不返回 `input_latents`**
- 训练模式返回包含 `input_latents` 的 dict
- VAE 编码路径正确（复用 Pipeline 已有的 `preprocess_xxx`、`vae.encode` 等方法）
- `output_params` 声明了所有可能返回的 key（框架按模式动态返回）
- Tile 参数需验证 VAE 本身是否支持，不同 VAE 的 `encode` 签名不同
- **禁止**在 `InputXXXEmbedder` 中处理特殊功能输入（编辑图像、mask、retake 等），这些由各自专属的 Embedder Unit 处理

## TrainingModule 结构

继承 `DiffusionTrainingModule`，实现三个方法：

| 方法 | 职责 | 自定义？ |
|------|------|---------|
| `__init__` | 加载 Pipeline、设置训练模式、定义 task_to_loss | 大部分复用 |
| `get_pipeline_inputs` | 将数据集字段映射到 Pipeline 输入 | **必须自定义** |
| `forward` | 入口函数 | **标准实现，不要修改** |

## 任务模式

三种任务模式的关系：

- `sft` = `sft:data_process` + `sft:train` 的完整流程。`sft` 在一个训练循环中同时完成前处理（VAE 编码、文本编码等）和 DiT 训练；而拆分模式将这两个阶段分开执行。

| 任务模式 | 行为 | 适用场景 |
|---------|------|---------|
| `sft` | 前处理 + 训练在一个循环中完成 | **首选方案**，显存足够时使用 |
| `sft:data_process` | 仅执行前处理，将结果缓存到硬盘 | 两阶段拆分训练的第一阶段 |
| `sft:train` | 从缓存读取前处理结果，仅训练 DiT | 两阶段拆分训练的第二阶段 |

所有训练脚本默认使用 `--task sft` 参数。对应地，`task_to_loss` 和 `launcher_map` 中只注册这三个 SFT 相关条目。

## Loss 函数

`task_to_loss` 和 `launcher_map` 是训练模板中唯二需要根据任务类型选择具体 Loss 类的地方。

### 为什么需要 `task_to_loss`

训练数据流：`get_pipeline_inputs` → `transfer_data_to_device` → Unit 链执行 → `task_to_loss[task](pipe, *inputs)`。
`task_to_loss` 将 Unit 链输出的 `inputs` 元组解包后传给 Loss 类，计算训练损失。

### `FlowMatchSFTLoss`（Flow Matching）

**适用于所有 Flow Matching 架构的 DiT**（FLUX、Qwen-Image、Wan、LTX-2 等绝大多数现代模型）。
接收 `inputs_shared` 和 `inputs_posi` 中的字段，核心依赖 `input_latents`（由 `InputXXXEmbedder` 在训练模式下返回）。

```python
self.task_to_loss = {
    "sft:data_process": lambda pipe, *args: args,
    "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
    "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
}
```

- `sft:data_process` 返回原始 `*args`，不计算 Loss，只做前处理缓存
- `sft` 和 `sft:train` 使用相同的 `FlowMatchSFTLoss`

### `FlowMatchSFTAudioVideoLoss`（音视频联合）

**适用于音视频联合生成任务**（如 LTX-2 T2AV）。在 `FlowMatchSFTLoss` 基础上额外接收 `audio_input_latents`。
`inputs_shared` 中必须包含 `audio_input_latents` 字段（由对应的 `InputAudioEmbedder` 返回）。

```python
self.task_to_loss = {
    "sft:data_process": lambda pipe, *args: args,
    "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTAudioVideoLoss(pipe, **inputs_shared, **inputs_posi),
    "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTAudioVideoLoss(pipe, **inputs_shared, **inputs_posi),
}
```

### 选择规则

| 任务类型 | Loss 类 | `input_latents` 来源 | 额外字段 |
|---------|---------|---------------------|---------|
| 图像生成（T2I） | `FlowMatchSFTLoss` | `InputImageEmbedder` | 无 |
| 视频生成（T2V） | `FlowMatchSFTLoss` | `InputVideoEmbedder` | 无 |
| 音视频联合（T2AV） | `FlowMatchSFTAudioVideoLoss` | `InputVideoEmbedder` | `audio_input_latents` |
| 图像编辑 | `FlowMatchSFTLoss` | `InputImageEmbedder` | 无 |

**distillation 场景**：按需添加 `distillation` 相关 task 条目和对应 Loss 类，不要添加到 SFT 模板中。

## `special_operator_map` 使用指南

### 数据加载流程

`UnifiedDataset` 对 `data_file_keys` 中的字段按以下优先级加载：

```
字段在 special_operator_map 中？
  ├─ 是 → 调用 special_operator_map[key](data[key])
  └─ 否 → 字段在 data_file_keys 中？
            ├─ 是 → 调用 main_data_operator(data[key])
            └─ 否 → 不处理，直接输出原始值
```

**`special_operator_map` 优先级高于 `main_data_operator`**。当某个字段需要独立于主数据算子的处理逻辑时，必须放入 `special_operator_map`。

### 何时使用

| 场景 | 用 `main_data_operator` | 用 `special_operator_map` |
|------|------------------------|--------------------------|
| 基础 T2I，只有 `image` 字段 | ✅ 使用 `default_image_operator` | — |
| 基础 T2V，只有 `video` 字段 | ✅ 使用 `default_video_operator` | — |
| 图像编辑，`edit_image` 需要和 `image` 相同处理 | — | ✅ 加入 `special_operator_map` |
| 音视频联合，`input_audio` 需要音频算子 | — | ✅ 加入 `special_operator_map` |
| 多层图像，`layer_input_image` 需要 RGBA 加载 | — | ✅ 加入 `special_operator_map` |
| 面部动画，`animate_face_video` 固定分辨率 | — | ✅ 加入 `special_operator_map` |

### 常用算子

从 `diffsynth.core.data.operators` 导入：

| 算子 | 用途 | 示例 |
|------|------|------|
| `ToAbsolutePath(base_path)` | 相对路径转绝对路径 | `ToAbsolutePath(args.dataset_base_path)` |
| `LoadImage()` | 加载图像为 PIL Image | 默认 RGB 模式 |
| `LoadImage(convert_RGB=False, convert_RGBA=True)` | 加载图像保留 Alpha 通道 | 多层图像场景 |
| `LoadVideo(num_frames, time_division_factor, time_division_remainder, ...)` | 加载视频 | 与 `main_data_operator` 中参数一致 |
| `LoadAudioWithTorchaudio(...)` | 加载音频 | T2AV 场景 |
| `ImageCropAndResize(height, width, max_pixels, h_div, w_div)` | 图像裁剪缩放 | 确保尺寸可被模型整除 |
| `RouteByType(operator_map=[...])` | 按数据类型路由 | JSON 数组 vs 单路径 |
| `SequencialProcess(operator)` | 对列表中每个元素应用算子 | 多图输入场景 |

算子用 `>>` 组合，形成处理管道。

### 复杂场景：`RouteByType` + `SequencialProcess`

当 `metadata.csv` 中同一列可能是**单个路径字符串**或**JSON 数组**（如 `["images/001_a.jpg", "images/001_b.jpg"]`）时，使用 `RouteByType` 根据数据类型分发到不同处理链：

```python
from diffsynth.core.data.operators import ToAbsolutePath, LoadImage, ImageCropAndResize, RouteByType, SequencialProcess

special_operator_map={
    "image": RouteByType(operator_map=[
        # 单个路径字符串 → 直接加载为 RGB 图像
        (str, ToAbsolutePath(args.dataset_base_path) >> LoadImage() >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16)),
        # JSON 数组（list） → 对每个元素分别加载（保留 RGBA），再逐一处理
        (list, SequencialProcess(ToAbsolutePath(args.dataset_base_path) >> LoadImage(convert_RGB=False, convert_RGBA=True) >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16))),
    ]),
}
```

**原理**：`RouteByType` 检查 `data[key]` 的 Python 类型（`str` 或 `list`），匹配到对应的处理链。`SequencialProcess` 内部对列表中的每个元素逐一应用其包含的算子链，返回处理后的列表。

**典型场景**：Qwen-Image 等多图像输入模型，数据集的 `image` 列可能是单图路径（`images/001.jpg`）或多图 JSON 数组（`["images/001_a.jpg", "images/001_b.jpg"]`）。此时 `image` 列既在 `main_data_operator` 中定义了默认行为，又在 `special_operator_map` 中通过 `RouteByType` 覆盖，以支持两种数据格式。

### 图像任务标准配置

```python
from diffsynth.core.data.operators import ToAbsolutePath, LoadImage, ImageCropAndResize, RouteByType

dataset = UnifiedDataset(
    ...
    main_data_operator=UnifiedDataset.default_image_operator(...),
    special_operator_map={
        # 基础 T2I：image 直接处理
        "image": ToAbsolutePath(args.dataset_base_path) >> LoadImage() >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16),
        # 图像编辑：edit_image 与 image 处理方式相同
        "edit_image": ToAbsolutePath(args.dataset_base_path) >> LoadImage() >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16),
        # 多层图像：layer_input_image 需要 RGBA
        "layer_input_image": ToAbsolutePath(args.dataset_base_path) >> LoadImage(convert_RGB=False, convert_RGBA=True) >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16),
    },
)
```

### 视频任务标准配置

```python
from diffsynth.core.data.operators import ToAbsolutePath, LoadAudioWithTorchaudio

video_processor = UnifiedDataset.default_video_operator(...)
dataset = UnifiedDataset(
    ...
    main_data_operator=video_processor,
    special_operator_map={
        # T2AV 场景：audio 需要专门的加载
        "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudioWithTorchaudio(
            num_frames=args.num_frames, time_division_factor=8, time_division_remainder=1, frame_rate=args.frame_rate
        ),
    },
)
```

## 图像任务 — `train.py`

```python
import torch, os, argparse, accelerate
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import ToAbsolutePath, LoadImage, ImageCropAndResize, RouteByType, SequencialProcess
from diffsynth.pipelines.{series} import {SeriesName}Pipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class {SeriesName}TrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        quant_options=None,
        template_model_id_or_path=None,
        resume_from_checkpoint=None, remove_prefix_in_ckpt=None,
        enable_lora_hot_loading=False,
        device="cpu",
        task="sft",
    ):
        super().__init__()
        # ===== 解析模型配置（固定写法） =====
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, quant_options=quant_options, device=device)
        # ===== Tokenizer 配置 =====
        # ⚠️ 需调整：default_value 中的 model_id 和 origin_file_pattern 为实际 tokenizer 路径
        tokenizer_config = self.parse_path_or_model_id(tokenizer_path, default_value=ModelConfig(model_id="model-owner/Model-Name", origin_file_pattern="tokenizer/"))
        # ===== 构建 Pipeline =====
        # ⚠️ 需调整：{SeriesName}Pipeline 为实际 Pipeline 类名
        self.pipe = {SeriesName}Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config)
        # ===== 拆分 Pipeline Units（固定写法） =====
        self.pipe = self.load_training_template_model(self.pipe, template_model_id_or_path, use_gradient_checkpointing, use_gradient_checkpointing_offload)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)
        if enable_lora_hot_loading: self.pipe.dit = self.pipe.enable_lora_hot_loading(self.pipe.dit)

        # ===== 切换到训练模式（固定写法） =====
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )

        # ===== 其他配置（固定写法） =====
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        # ===== 任务模式路由（固定写法） =====
        # 仅保留 SFT 相关条目，distillation 场景按需添加
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
        }

    def get_pipeline_inputs(self, data):
        # ===== 正向提示词：从数据集 data 中读取（固定写法） =====
        inputs_posi = {"prompt": data["prompt"]}
        # ===== 负向提示词：必须为空（固定写法，不可修改） =====
        # CFG 是仅推理侧的行为，训练不需要负向提示词
        inputs_nega = {"negative_prompt": ""}
        # ===== 共享参数 =====
        inputs_shared = {
            # ===== 核心字段映射：⚠️ 需调整 key 名 =====
            # input_image 必须与 InputImageEmbedder 的 input_params 一致
            # 如果模型处理多张图（如编辑任务），需要额外处理 list 分支
            "input_image": data["image"],
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            # ===== 框架控制参数（固定写法） =====
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        # ===== 额外字段注入：通过 --extra_inputs 配置的数据集列名（固定写法） =====
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        # ===== 标准实现，不要修改（固定写法） =====
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def {series}_parser():
    parser = argparse.ArgumentParser(description="{Series Name} training.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    return parser


if __name__ == "__main__":
    parser = {series}_parser()
    args = parser.parse_args()
    # ===== Accelerator 配置（固定写法） =====
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    # ===== 数据集定义 =====
    # division_factor 必须与 pipeline.py 中的声明一致
    # ⚠️ 需调整：height_division_factor / width_division_factor
    # ⚠️ 需调整：special_operator_map 中的 key 为 dataset 列名，operator 按实际数据类型选择
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        ),
        special_operator_map={
            # ⚠️ 需调整：列名与 metadata.csv 一致
            # 如果列可能是 JSON 数组（多图输入），使用 RouteByType 按类型路由
            "image": RouteByType(operator_map=[
                (str, ToAbsolutePath(args.dataset_base_path) >> LoadImage() >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16)),
                (list, SequencialProcess(ToAbsolutePath(args.dataset_base_path) >> LoadImage(convert_RGB=False, convert_RGBA=True) >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16))),
            ]),
        },
    )
    # ===== TrainingModule =====
    # ⚠️ 需调整：{SeriesName}TrainingModule 为实际类名
    model = {SeriesName}TrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        quant_options=args.quant_options,
        template_model_id_or_path=args.template_model_id_or_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_lora_hot_loading=args.enable_lora_hot_loading,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
    )
    # ===== ModelLogger（固定写法） =====
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
        enable_csv_log=args.enable_csv_log,
    )
    # ===== 任务路由（固定写法） =====
    # 仅保留 SFT 相关条目，与 task_to_loss 保持一致
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
```

## 视频任务 — `train.py`

```python
import torch, os, argparse, accelerate
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import ToAbsolutePath, LoadAudioWithTorchaudio, RouteByType, SequencialProcess
from diffsynth.pipelines.{series} import {SeriesName}Pipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class {SeriesName}TrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        quant_options=None,
        template_model_id_or_path=None,
        resume_from_checkpoint=None, remove_prefix_in_ckpt=None,
        enable_lora_hot_loading=False,
        device="cpu",
        task="sft",
    ):
        super().__init__()
        # ===== 解析模型配置（固定写法） =====
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, quant_options=quant_options, device=device)
        # ===== Tokenizer 配置 =====
        # ⚠️ 需调整：default_value 为实际 tokenizer 路径
        tokenizer_config = self.parse_path_or_model_id(tokenizer_path, default_value=ModelConfig(model_id="model-owner/Model-Name", origin_file_pattern="tokenizer/"))
        # ===== 构建 Pipeline =====
        # ⚠️ 需调整：{SeriesName}Pipeline 为实际 Pipeline 类名
        self.pipe = {SeriesName}Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config)
        # ===== 拆分 Pipeline Units（固定写法） =====
        self.pipe = self.load_training_template_model(self.pipe, template_model_id_or_path, use_gradient_checkpointing, use_gradient_checkpointing_offload)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)
        if enable_lora_hot_loading: self.pipe.dit = self.pipe.enable_lora_hot_loading(self.pipe.dit)

        # ===== 切换到训练模式（固定写法） =====
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )

        # ===== 其他配置（固定写法） =====
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        # ===== 任务模式路由（固定写法） =====
        # 仅保留 SFT 相关条目，distillation 场景按需添加
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
        }

    def get_pipeline_inputs(self, data):
        # ===== 正向提示词：从数据集 data 中读取（固定写法） =====
        inputs_posi = {"prompt": data["prompt"]}
        # ===== 负向提示词：必须为空（固定写法，不可修改） =====
        inputs_nega = {"negative_prompt": ""}
        # ===== 共享参数 =====
        inputs_shared = {
            # ===== 核心字段映射：⚠️ 需调整 key 名 =====
            # input_video 必须与 InputVideoEmbedder 的 input_params 一致
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "frame_rate": data.get("frame_rate", 24),
            # ===== 框架控制参数（固定写法） =====
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        # ===== 额外字段注入：通过 --extra_inputs 配置的数据集列名（固定写法） =====
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        # ===== 标准实现，不要修改（固定写法） =====
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def {series}_parser():
    parser = argparse.ArgumentParser(description="{Series Name} training.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--frame_rate", type=float, default=24, help="frame rate of the training videos.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    return parser


if __name__ == "__main__":
    parser = {series}_parser()
    args = parser.parse_args()
    # ===== Accelerator 配置（固定写法） =====
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    # ===== 视频数据加载器 =====
    # ⚠️ 需调整：所有 division_factor 值必须与 pipeline.py 中一致
    video_processor = UnifiedDataset.default_video_operator(
        base_path=args.dataset_base_path,
        max_pixels=args.max_pixels,
        height=args.height,
        width=args.width,
        height_division_factor=32,
        width_division_factor=32,
        num_frames=args.num_frames,
        time_division_factor=8,
        time_division_remainder=1,
        frame_rate=args.frame_rate,
        fix_frame_rate=True,
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=video_processor,
        special_operator_map={
            # ⚠️ 需调整：列名与 metadata.csv 一致，operator 按实际数据类型选择
            # 基础 T2V 视频任务不需要 special_operator_map，仅用 main_data_operator 即可
            # 如有 audio 等额外字段，参考下方「special_operator_map 使用指南」添加
        },
    )
    # ===== TrainingModule =====
    # ⚠️ 需调整：{SeriesName}TrainingModule 为实际类名
    model = {SeriesName}TrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        quant_options=args.quant_options,
        template_model_id_or_path=args.template_model_id_or_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_lora_hot_loading=args.enable_lora_hot_loading,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
    )
    # ===== ModelLogger（固定写法） =====
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
        enable_csv_log=args.enable_csv_log,
    )
    # ===== 任务路由（固定写法） =====
    # 仅保留 SFT 相关条目，与 task_to_loss 保持一致
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
```
