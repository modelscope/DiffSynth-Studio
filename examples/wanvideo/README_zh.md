# 通义万相 2.1（Wan 2.1）

|模型 ID|类型|额外参数|推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|-|
|[Wan-AI/Wan2.1-T2V-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B)|基础模型||[code](./model_inference/Wan2.1-T2V-1.3B.py)|[code](./model_training/full/Wan2.1-T2V-1.3B.sh)|[code](./model_training/validate_full/Wan2.1-T2V-1.3B.py)|[code](./model_training/lora/Wan2.1-T2V-1.3B.sh)|[code](./model_training/validate_lora/Wan2.1-T2V-1.3B.py)|
|[Wan-AI/Wan2.1-T2V-14B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B)|基础模型||[code](./model_inference/Wan2.1-T2V-14B.py)|[code](./model_training/full/Wan2.1-T2V-14B.sh)|[code](./model_training/validate_full/Wan2.1-T2V-14B.py)|[code](./model_training/lora/Wan2.1-T2V-14B.sh)|[code](./model_training/validate_lora/Wan2.1-T2V-14B.py)|
|[Wan-AI/Wan2.1-I2V-14B-480P](https://modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P)|基础模型|`input_image`|[code](./model_inference/Wan2.1-I2V-14B-480P.py)|[code](./model_training/full/Wan2.1-I2V-14B-480P.sh)|[code](./model_training/validate_full/Wan2.1-I2V-14B-480P.py)|[code](./model_training/lora/Wan2.1-I2V-14B-480P.sh)|[code](./model_training/validate_lora/Wan2.1-I2V-14B-480P.py)|
|[Wan-AI/Wan2.1-I2V-14B-720P](https://modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P)|基础模型|`input_image`|[code](./model_inference/Wan2.1-I2V-14B-720P.py)|[code](./model_training/full/Wan2.1-I2V-14B-720P.sh)|[code](./model_training/validate_full/Wan2.1-I2V-14B-720P.py)|[code](./model_training/lora/Wan2.1-I2V-14B-720P.sh)|[code](./model_training/validate_lora/Wan2.1-I2V-14B-720P.py)|
|[Wan-AI/Wan2.1-FLF2V-14B-720P](https://modelscope.cn/models/Wan-AI/Wan2.1-FLF2V-14B-720P)|基础模型|`input_image`, `end_image`|[code](./model_inference/Wan2.1-FLF2V-14B-720P.py)|[code](./model_training/full/Wan2.1-FLF2V-14B-720P.sh)|[code](./model_training/validate_full/Wan2.1-FLF2V-14B-720P.py)|[code](./model_training/lora/Wan2.1-FLF2V-14B-720P.sh)|[code](./model_training/validate_lora/Wan2.1-FLF2V-14B-720P.py)|
|[PAI/Wan2.1-Fun-1.3B-InP](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-InP)|基础模型|`input_image`, `end_image`|[code](./model_inference/Wan2.1-Fun-1.3B-InP.py)|[code](./model_training/full/Wan2.1-Fun-1.3B-InP.sh)|[code](./model_training/validate_full/Wan2.1-Fun-1.3B-InP.py)|[code](./model_training/lora/Wan2.1-Fun-1.3B-InP.sh)|[code](./model_training/validate_lora/Wan2.1-Fun-1.3B-InP.py)|
|[PAI/Wan2.1-Fun-1.3B-Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-Control)|基础模型|`control_video`|[code](./model_inference/Wan2.1-Fun-1.3B-Control.py)|[code](./model_training/full/Wan2.1-Fun-1.3B-Control.sh)|[code](./model_training/validate_full/Wan2.1-Fun-1.3B-Control.py)|[code](./model_training/lora/Wan2.1-Fun-1.3B-Control.sh)|[code](./model_training/validate_lora/Wan2.1-Fun-1.3B-Control.py)|
|[PAI/Wan2.1-Fun-14B-InP](https://modelscope.cn/models/PAI/Wan2.1-Fun-14B-InP)|基础模型|`input_image`, `end_image`|[code](./model_inference/Wan2.1-Fun-14B-InP.py)|[code](./model_training/full/Wan2.1-Fun-14B-InP.sh)|[code](./model_training/validate_full/Wan2.1-Fun-14B-InP.py)|[code](./model_training/lora/Wan2.1-Fun-14B-InP.sh)|[code](./model_training/validate_lora/Wan2.1-Fun-14B-InP.py)|
|[PAI/Wan2.1-Fun-14B-Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-14B-Control)|基础模型|`control_video`|[code](./model_inference/Wan2.1-Fun-14B-Control.py)|[code](./model_training/full/Wan2.1-Fun-14B-Control.sh)|[code](./model_training/validate_full/Wan2.1-Fun-14B-Control.py)|[code](./model_training/lora/Wan2.1-Fun-14B-Control.sh)|[code](./model_training/validate_lora/Wan2.1-Fun-14B-Control.py)|
|[PAI/Wan2.1-Fun-V1.1-1.3B-Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control)|基础模型|`control_video`, `reference_image`|[code](./model_inference/Wan2.1-Fun-V1.1-1.3B-Control.py)|[code](./model_training/full/Wan2.1-Fun-V1.1-1.3B-Control.sh)|[code](./model_training/validate_full/Wan2.1-Fun-V1.1-1.3B-Control.py)|[code](./model_training/lora/Wan2.1-Fun-V1.1-1.3B-Control.sh)|[code](./model_training/validate_lora/Wan2.1-Fun-V1.1-1.3B-Control.py)|
|[PAI/Wan2.1-Fun-V1.1-14B-Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control)|基础模型|`control_video`, `reference_image`|[code](./model_inference/Wan2.1-Fun-V1.1-14B-Control.py)|[code](./model_training/full/Wan2.1-Fun-V1.1-14B-Control.sh)|[code](./model_training/validate_full/Wan2.1-Fun-V1.1-14B-Control.py)|[code](./model_training/lora/Wan2.1-Fun-V1.1-14B-Control.sh)|[code](./model_training/validate_lora/Wan2.1-Fun-V1.1-14B-Control.py)|
|[PAI/Wan2.1-Fun-V1.1-1.3B-InP](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-InP)|基础模型|`input_image`, `end_image`||||||
|[PAI/Wan2.1-Fun-V1.1-14B-InP](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-InP)|基础模型|`input_image`, `end_image`||||||
|[PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera)|基础模型|||||||
|[PAI/Wan2.1-Fun-V1.1-14B-Control-Camera](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control-Camera)|基础模型|||||||
|[iic/VACE-Wan2.1-1.3B-Preview](https://modelscope.cn/models/iic/VACE-Wan2.1-1.3B-Preview)|适配器|`vace_control_video`, `vace_reference_image`|[code](./model_inference/Wan2.1-VACE-1.3B-Preview.py)|[code](./model_training/full/VACE-Wan2.1-1.3B-Preview.sh)|[code](./model_training/validate_full/VACE-Wan2.1-1.3B-Preview.py)|[code](./model_training/lora/VACE-Wan2.1-1.3B-Preview.sh)|[code](./model_training/validate_lora/VACE-Wan2.1-1.3B-Preview.py)|
|[Wan-AI/Wan2.1-VACE-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-VACE-1.3B)|适配器|`vace_control_video`, `vace_reference_image`||||||
|[Wan-AI/Wan2.1-VACE-14B](https://modelscope.cn/models/Wan-AI/Wan2.1-VACE-14B)|适配器|`vace_control_video`, `vace_reference_image`||||||
|[DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1](https://modelscope.cn/models/DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1)|适配器|`motion_bucket_id`|[code](./model_inference/Wan2.1-1.3b-speedcontrol-v1.py)|[code](./model_training/full/Wan2.1-1.3b-speedcontrol-v1.sh)|[code](./model_training/validate_full/Wan2.1-1.3b-speedcontrol-v1.py)|[code](./model_training/lora/Wan2.1-1.3b-speedcontrol-v1.sh)|[code](./model_training/validate_lora/Wan2.1-1.3b-speedcontrol-v1.py)|

## 模型推理

### 加载模型

模型通过 `from_pretrained` 加载：

```python
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
)
```

其中 `torch_dtype` 和 `device` 是计算精度和计算设备。`model_configs` 可通过多种方式配置模型路径：

* 从[魔搭社区](https://modelscope.cn/)下载模型并加载。此时需要填写 `model_id` 和 `origin_file_pattern`，例如

```python
ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
```

* 从本地文件路径加载模型。此时需要填写 `path`，例如

```python
ModelConfig(path="models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
```

对于从多个文件加载的单一模型，使用列表即可，例如

```python
ModelConfig(path=[
    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors",
])
```

`from_pretrained` 还提供了额外的参数用于控制模型加载时的行为：

* `tokenizer_config`: Wan 模型的 tokenizer 路径，默认值为 `ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*")`。
* `local_model_path`: 用于保存下载模型的路径，默认值为 `"./models"`。
* `skip_download`: 是否跳过下载，默认值为 `False`。当您的网络无法访问[魔搭社区](https://modelscope.cn/)时，请手动下载必要的文件，并将其设置为 `True`。
* `redirect_common_files`: 是否重定向重复模型文件，默认值为 `True`。由于 Wan 系列模型包括多个基础模型，每个基础模型的 text encoder 等模块都是相同的，为避免重复下载，我们会对模型路径进行重定向。

### 显存管理

DiffSynth-Studio 为 Wan 模型提供了细粒度的显存管理，让模型能够在低显存设备上进行推理，可通过以下代码开启 offload 功能，在显存有限的设备上将部分模块 offload 到内存中。

```python
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()
```

FP8 量化功能也是支持的：

```python
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_dtype=torch.float8_e4m3fn),
    ],
)
pipe.enable_vram_management()
```

FP8 量化和 offload 可同时开启：

```python
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
    ],
)
pipe.enable_vram_management()
```

FP8 量化能够大幅度减少显存占用，但不会加速，部分模型在 FP8 量化下会出现精度不足导致的画面模糊、撕裂、失真问题，请谨慎使用 FP8 量化。

`enable_vram_management` 函数提供了以下参数，用于控制显存使用情况：

* `vram_limit`: 显存占用量（GB），默认占用设备上的剩余显存。注意这不是一个绝对限制，当设置的显存不足以支持模型进行推理，但实际可用显存足够时，将会以最小化显存占用的形式进行推理。
* `vram_buffer`: 显存缓冲区大小（GB），默认为 0.5GB。由于部分较大的神经网络层在 onload 阶段会不可控地占用更多显存，因此一个显存缓冲区是必要的，理论上的最优值为模型中最大的层所占的显存。
* `num_persistent_param_in_dit`: DiT 模型中常驻显存的参数数量（个），默认为无限制。我们将会在未来删除这个参数，请不要依赖这个参数。

### 输入参数

Pipeline 在推理阶段能够接收以下输入参数：

* `prompt`: 提示词，描述画面中出现的内容。
* `negative_prompt`: 负向提示词，描述画面中不应该出现的内容，默认值为 `""`。
* `input_image`: 输入图片，适用于图生视频模型，例如 [`Wan-AI/Wan2.1-I2V-14B-480P`](https://modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P)、[`PAI/Wan2.1-Fun-1.3B-InP`](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-InP)，以及首尾帧模型，例如 [`Wan-AI/Wan2.1-FLF2V-14B-720P`](Wan-AI/Wan2.1-FLF2V-14B-720P)。
* `end_image`: 结尾帧，适用于首尾帧模型，例如 [`Wan-AI/Wan2.1-FLF2V-14B-720P`](Wan-AI/Wan2.1-FLF2V-14B-720P)。
* `input_video`: 输入视频，用于视频生视频，适用于任意 Wan 系列模型，需与参数 `denoising_strength` 配合使用。
* `denoising_strength`: 去噪强度，范围为 [0, 1]。数值越小，生成的视频越接近 `input_video`。
* `control_video`: 控制视频，适用于带控制能力的 Wan 系列模型，例如 [`PAI/Wan2.1-Fun-1.3B-Control`](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-Control)。
* `reference_image`: 参考图片，适用于带参考图能力的 Wan 系列模型，例如 [`PAI/Wan2.1-Fun-V1.1-1.3B-Control`](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control)。
* `vace_video`: VACE 模型的输入视频，适用于 VACE 系列模型，例如 [`iic/VACE-Wan2.1-1.3B-Preview`](https://modelscope.cn/models/iic/VACE-Wan2.1-1.3B-Preview)。
* `vace_video_mask`: VACE 模型的 mask 视频，适用于 VACE 系列模型，例如 [`iic/VACE-Wan2.1-1.3B-Preview`](https://modelscope.cn/models/iic/VACE-Wan2.1-1.3B-Preview)。
* `vace_reference_image`: VACE 模型的参考图片，适用于 VACE 系列模型，例如 [`iic/VACE-Wan2.1-1.3B-Preview`](https://modelscope.cn/models/iic/VACE-Wan2.1-1.3B-Preview)。
* `vace_scale`: VACE 模型对基础模型的影响程度，默认为1。数值越大，控制强度越高，但画面崩坏概率越大。
* `seed`: 随机种子。默认为 `None`，即完全随机。
* `rand_device`: 生成随机高斯噪声矩阵的计算设备，默认为 `"cpu"`。当设置为 `cuda` 时，在不同 GPU 上会导致不同的生成结果。
* `height`: 帧高度，默认为 480。需设置为 16 的倍数，不满足时向上取整。
* `width`: 帧宽度，默认为 832。需设置为 16 的倍数，不满足时向上取整。
* `num_frames`: 帧数，默认为 81。需设置为 4 的倍数 + 1，不满足时向上取整，最小值为 1。
* `cfg_scale`: Classifier-free guidance 机制的数值，默认为 5。数值越大，提示词的控制效果越强，但画面崩坏的概率越大。
* `cfg_merge`: 是否合并 Classifier-free guidance 的两侧进行统一推理，默认为 `False`。该参数目前仅在基础的文生视频和图生视频模型上生效。
* `num_inference_steps`: 推理次数，默认值为 50。
* `sigma_shift`: Rectified Flow 理论中的参数，默认为 5。数值越大，模型在去噪的开始阶段停留的步骤数越多，可适当调大这个参数来提高画面质量，但会因生成过程与训练过程不一致导致生成的视频内容与训练数据存在差异。
* `motion_bucket_id`: 运动幅度，范围为 [0, 100]。适用于速度控制模块，例如 [`DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1`](https://modelscope.cn/models/DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1)，数值越大，运动幅度越大。
* `tiled`: 是否启用 VAE 分块推理，默认为 `False`。设置为 `True` 时可显著减少 VAE 编解码阶段的显存占用，会产生少许误差，以及少量推理时间延长。
* `tile_size`: VAE 编解码阶段的分块大小，默认为 (30, 52)，仅在 `tiled=True` 时生效。
* `tile_stride`: VAE 编解码阶段的分块步长，默认为 (15, 26)，仅在 `tiled=True` 时生效，需保证其数值小于或等于 `tile_size`。
* `sliding_window_size`: DiT 部分的滑动窗口大小。实验性功能，效果不稳定。
* `sliding_window_stride`: DiT 部分的滑动窗口步长。实验性功能，效果不稳定。
* `tea_cache_l1_thresh`: TeaCache 的阈值，数值越大，速度越快，画面质量越差。请注意，开启 TeaCache 后推理速度并非均匀，因此进度条上显示的剩余时间将会变得不准确。
* `tea_cache_model_id`: TeaCache 的参数模板，可选 `"Wan2.1-T2V-1.3B"`、`Wan2.1-T2V-14B`、`Wan2.1-I2V-14B-480P`、`Wan2.1-I2V-14B-720P` 之一。
* `progress_bar_cmd`: 进度条，默认为 `tqdm.tqdm`。可通过设置为 `lambda x:x` 来屏蔽进度条。

## 模型训练

Wan 系列模型训练通过统一的 [`./model_training/train.py`](./model_training/train.py) 脚本进行。

脚本包含以下参数：

* 数据集
  * `--dataset_base_path`: 数据集的根路径。
  * `--dataset_metadata_path`: 数据集的元数据文件路径。
  * `--height`: 图像或视频的高度。将 `height` 和 `width` 留空以启用动态分辨率。
  * `--width`: 图像或视频的宽度。将 `height` 和 `width` 留空以启用动态分辨率。
  * `--num_frames`: 每个视频中的帧数。帧从视频前缀中采样。
  * `--data_file_keys`: 元数据中的数据文件键。用逗号分隔。
  * `--dataset_repeat`: 每个 epoch 中数据集重复的次数。
* 模型
  * `--model_paths`: 要加载的模型路径。JSON 格式。
  * `--model_id_with_origin_paths`: 带原始路径的模型 ID，例如 Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors。用逗号分隔。
* 训练
  * `--learning_rate`: 学习率。
  * `--num_epochs`: 轮数（Epoch）数量。
  * `--output_path`: 保存路径。
  * `--remove_prefix_in_ckpt`: 在 ckpt 中移除前缀。
* 可训练模块
  * `--trainable_models`: 可训练的模型，例如 dit、vae、text_encoder。
  * `--lora_base_model`: LoRA 添加到哪个模型上。
  * `--lora_target_modules`: LoRA 添加到哪一层上。
  * `--lora_rank`: LoRA 的秩（Rank）。
* 额外模型输入
  * `--input_contains_input_image`: 模型输入包含 `input_image`
  * `--input_contains_end_image`: 模型输入包含 `end_image`。
  * `--input_contains_control_video`: 模型输入包含 `control_video`。
  * `--input_contains_reference_image`: 模型输入包含 `reference_image`。
  * `--input_contains_vace_video`: 模型输入包含 `vace_video`。
  * `--input_contains_vace_reference_image`: 模型输入包含 `vace_reference_image`。
  * `--input_contains_motion_bucket_id`: 模型输入包含 `motion_bucket_id`。
* 显存管理
  * `--use_gradient_checkpointing_offload`: 是否将 gradient checkpointing 卸载到内存中。

### Step 1: 准备数据集

数据集包含一系列文件，我们建议您这样组织数据集文件：

```
data/example_video_dataset/
├── metadata.csv
├── video1.mp4
└── video2.mp4
```

其中 `video1.mp4`、`video2.mp4` 为训练用视频数据，`metadata.csv` 为元数据列表，例如

```
video,prompt
video1.mp4,"from sunset to night, a small town, light, house, river"
video2.mp4,"a dog is running"
```

数据集支持视频和图片混合训练，支持的视频文件格式包括 `"mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"`，支持的图片格式包括 `"jpg", "jpeg", "png", "webp"`。

视频的尺寸可通过脚本参数 `--height`、`--width`、`--num_frames` 控制。在每个视频中，前 `num_frames` 帧会被用于训练，因此当视频长度不足 `num_frames` 帧时会报错，图片文件会被视为单帧视频。当 `--height` 和 `--width` 为空时将会开启动态分辨率，按照数据集中每个视频或图片的实际宽高训练。

**我们强烈建议使用固定分辨率训练，并避免图像和视频混合训练，因为在多卡训练中存在负载均衡问题。**

当模型需要额外输入时，例如具备控制能力的模型 [`PAI/Wan2.1-Fun-1.3B-Control`](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-Control) 所需的 `control_video`，请在数据集中补充相应的列，例如：

```
video,prompt,control_video
video1.mp4,"from sunset to night, a small town, light, house, river",video1_softedge.mp4
```

额外输入若包含视频和图像文件，则需要在 `--data_file_keys` 参数中指定要解析的列名。该参数的默认值为 `"image,video"`，即解析列名为 `image` 和 `video` 的列。可根据额外输入增加相应的列名，例如 `--data_file_keys "image,video,control_video"`，同时启用 `--input_contains_control_video`。

### Step 2: 加载模型

类似于推理时的模型加载逻辑，可直接通过模型 ID 配置要加载的模型。例如，推理时我们通过以下设置加载模型

```python
model_configs=[
    ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
    ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
    ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth"),
]
```

那么在训练时，填入以下参数即可加载对应的模型。

```shell
--model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth"
```

如果您希望从本地文件加载模型，例如推理时

```python
model_configs=[
    ModelConfig(path=[
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors",
    ]),
    ModelConfig(path="models/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth"),
    ModelConfig(path="models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth"),
]
```

那么训练时需设置为

```shell
--model_paths '[
    [
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors"
    ],
    "models/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
    "models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth"
]' \
```

### 设置可训练模块

训练框架支持训练基础模型，或 LoRA 模型。以下是几个例子：

* 全量训练 DiT 部分：`--trainable_models dit`
* 训练 DiT 部分的 LoRA 模型：`--lora_base_model dit --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank 32`
* 训练 DiT 部分的 LoRA 和 Motion Controller 部分（是的，可以训练这种花里胡哨的结构）：`--trainable_models motion_controller --lora_base_model dit --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank 32`

此外，由于训练脚本中加载了多个模块（text encoder、dit、vae），保存模型文件时需要移除前缀，例如在全量训练 DiT 部分或者训练 DiT 部分的 LoRA 模型时，请设置 `--remove_prefix_in_ckpt pipe.dit.`

### 启动训练程序

我们构建了一个样例视频数据集，以方便您进行测试，通过以下命令可以下载这个数据集：

```shell
modelscope download --dataset DiffSynth-Studio/example_video_dataset README.md --local_dir ./data/example_video_dataset
```

我们为每一个模型编写了训练命令，请参考本文档开头的表格。

请注意，14B 模型全量训练需要8个GPU，每个GPU的显存至少为80G。全量训练这些14B模型时需要安装 `deepspeed`（`pip install deepspeed`），我们编写了建议的[配置文件](./model_training/full/accelerate_config_14B.yaml)，这个配置文件会在对应的训练脚本中被加载，这些脚本已在 8*A100 上测试过。

训练脚本的默认视频尺寸为 `480*832*81`，提升分辨率将可能导致显存不足，请添加参数 `--use_gradient_checkpointing_offload` 降低显存占用。
