# MiniMax-H3

MiniMax H3 是一个通用的全模态生成系统。它支持对由文本、图像、视频和音频组成的多模态上下文进行统一理解，并能生成最高达 2K 分辨率、最长 15 秒、包含原生立体声音频的视频。得益于其面向任务泛化的系统设计，H3 在预训练阶段就已具备广泛的多模态上下文理解与生成能力，从而在遵循复杂多模态指令方面表现出色。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。

## 快速开始

运行以下代码可以快速加载 [DiffSynth-Studio/MiniMax-H3-NF4](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4) NF4 量化模型并进行文生音视频推理。显存管理已启动，框架会自动根据剩余显存控制模型参数的加载，最低 7G 显存即可运行。

```python
import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-fl2va-nf4.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-text-encoder-nf4.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="video_vae_nf4.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="audio_vae_nf4.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)

# Text -> Video + Audio
prompt = "A girl is very happy, she is speaking in english: “I enjoy working with Diffsynth-Studio, it's a perfect framework.”"
video, audio = pipe(
    prompt=prompt,
    height=480, width=832, num_frames=124, num_inference_steps=50, seed=0,
)
write_video_audio(
    video=video, audio=audio,
    output_path="t2va.mp4", fps=24, audio_sample_rate=32000,
)
```

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
|[MiniMax/MiniMax-H3: FL2VA](https://www.modelscope.cn/models/MiniMax/MiniMax-H3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-FL2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-FL2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/full/MiniMax-H3-FL2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_full/MiniMax-H3-FL2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-FL2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-FL2VA.py)|
|[MiniMax/MiniMax-H3: Ref2VA](https://www.modelscope.cn/models/MiniMax/MiniMax-H3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-Ref2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Ref2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/full/MiniMax-H3-Ref2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_full/MiniMax-H3-Ref2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-Ref2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Ref2VA.py)|
|[MiniMax/MiniMax-H3: Retake](https://www.modelscope.cn/models/MiniMax/MiniMax-H3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-Retake.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Retake.py)|-|-|-|-|
|[DiffSynth-Studio/MiniMax-H3-NF4: FL2VA](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-NF4-FL2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-NF4-FL2VA.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-NF4-FL2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-NF4-FL2VA.py)|
|[DiffSynth-Studio/MiniMax-H3-NF4: Ref2VA](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-NF4-Ref2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-NF4-Ref2VA.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-NF4-Ref2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-NF4-Ref2VA.py)|

模型权重分为两个分区：`FL2VA` 分区服务文生音视频与首尾帧引导生成，`Ref2VA` 分区服务参考驱动生成，两者的 DiT 与文本编码器权重不同，需按任务选择对应分区的 `origin_file_pattern`。

## 模型推理

模型通过 `MiniMaxH3Pipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。加载时除 `model_configs` 外还包括：

* `processor_config`: Qwen3-VL processor 的 `ModelConfig`，用于对提示词及参考图像进行 tokenize，默认指向 `FL2VA/processor/`。使用 `Ref2VA` 分区时需显式改为 `Ref2VA/processor/`。
* `vram_limit`: 显存管理的显存上限（单位 GB），留空则不限制。

`MiniMaxH3Pipeline` 推理的输入参数包括：

* `prompt`: 提示词，描述视频中出现的内容以及人物说出的台词。
* `negative_prompt`: 负向提示词，默认值为 `" "`。该模型为 CFG 蒸馏模型，默认不生效。
* `height`: 视频高度，默认值为 768，需保证为 32 的倍数。
* `width`: 视频宽度，默认值为 1344，需保证为 32 的倍数。
* `num_frames`: 视频帧数，默认值为 124，会被向上对齐到最近的 `17n+5`，因此实际输出可能略长于请求值。视频帧率固定为 24。
* `num_inference_steps`: 推理步数，默认值为 50。
* `seed`: 随机种子，默认值为 42。
* `rand_device`: 生成随机高斯噪声矩阵的计算设备，默认为 `"cpu"`。当设置为 `cuda` 时，在不同 GPU 上会导致不同的生成结果。
* `cfg_scale`: Classifier-free guidance 的参数，默认值为 1.0。该模型为 CFG 蒸馏模型，建议保持默认值。
* `flow_shift`: 视频模态的 flow matching 时间步 shift，默认值为 12.0。
* `audio_flow_shift`: 音频模态的 flow matching 时间步 shift，默认值为 3.0。视频与音频使用两条独立的 sigma 调度。
* `tiled`: 是否启用 VAE 分块推理，默认为 `True`。设置为 `True` 时可显著减少 VAE 编解码阶段的显存占用，会产生少许误差，以及少量推理时间延长。
* `tile_size`: VAE 编解码阶段的分块大小，默认为 256。
* `tile_overlap`: VAE 编解码阶段的分块重叠大小，默认为 64。
* `keyframes`: 关键帧图像列表，用于首尾帧引导生成，图像会被缩放到目标画幅。
* `keyframe_indices`: 关键帧在视频中的帧索引列表，取值为 `0`（首帧）或 `-1`（尾帧），与 `keyframes` 一一对应。
* `references`: 参考条件列表，按请求顺序给出，每个元素为字典，支持以下四种形式：
    * `{"type": "image", "image": PIL.Image}`
    * `{"type": "video", "video": list[PIL.Image]}`（无声视频）
    * `{"type": "audio", "audio": Tensor[C, L], "sample_rate": int}`
    * `{"type": "video_audio", "video": list[PIL.Image], "audio": Tensor[C, L], "sample_rate": int}`

    其中传入的视频帧列表必须已经是 24fps，Pipeline 不会重采样帧率。`video` 一律按无声处理，因此若要把参考视频自带的声轨也作为音频条件，必须使用 `video_audio` 并显式传入波形 —— Pipeline 接收的是帧列表而非文件，无法自行探测声轨。可用 `diffsynth.utils.data.audio_video.read_video_audio` 从同一个文件同时读出画面与声轨，两者的时长会自动对齐：

    ```python
    from diffsynth.utils.data.audio_video import read_video_audio

    frames, waveform, sample_rate = read_video_audio(
        "video.mp4", height=480, width=832, num_frames=124, fps=24,
        audio_sample_rate=pipe.audio_vae.sample_rate,
    )
    ```
* `ref_image_short_edge`: 参考图像的短边目标长度，默认值为 2048。参考图像保持长宽比缩放至该短边（允许放大），两轴各自向最近的 32 倍数取整，不受面积上限约束。
* `ref_video_short_edge`: 参考视频的短边目标长度，默认值为 768。
* `ref_video_max_pixels`: 参考视频的面积软上限，默认值为 `768 * 1344`。参考视频先按短边定标，若面积超过该上限则等比缩回，最后两轴各自取整到 32 的倍数。宽于 16:9 的素材通常会触发该上限。
* `retake_video`: 视频重绘（retake）的源视频帧列表，必须已经是 24fps。帧会被缩放到目标画幅并截取前 `num_frames` 帧。由于 `num_frames` 会先向上对齐到最近的 `17n+5`，源视频常常会差几帧（例如 121 帧的素材对应对齐后的 124 帧），此时会重复最后一帧补齐，且补出的尾部会被重新生成而非冻结。源素材帧数不少于 `num_frames` 即可避免。
* `frame_regions_to_retake`: `retake_video` 中需要重新生成的**帧号**区间，左闭右开、帧从 0 计数，例如 `[(17, 51)]`。区间之外的内容会从源视频原样保留。一个 VAE clip 的 17 帧在隐空间是耦合的，重绘 clip 内任何一帧就等于重绘整个 clip，因此区间会向外扩展到 clip 边界；传 17 的倍数即可得到与请求完全一致的范围。不传该参数（或传入空列表）时整段源视频都会被保留，此时 `retake_video` 相当于视频驱动的音频生成。
* `retake_audio`: 音频重绘（retake）的源波形 `Tensor[C, L]`。会被转为立体声并重采样到音频 VAE 的采样率，再按视频时长截断或补齐。
* `retake_audio_sample_rate`: `retake_audio` 的采样率，默认值为 32000。
* `seconds_regions_to_retake`: `retake_audio` 中需要重新生成的时间区间，单位为**秒**、左闭右开，例如 `[(0, 1), (4, 5)]`。音频 VAE 是均匀压缩（每秒 40 个隐空间帧）、没有 clip 结构，因此区间按给定值直接使用，时间分辨率为 1/40 秒。不传该参数时整段源音频都会被保留，此时 `retake_audio` 相当于音频驱动的视频生成。

    视频与音频的 retake 相互独立：可以只用其中之一，注意两者单位不同 —— 视频用帧号，音频用秒。推荐用 `read_video_audio` 从同一个文件中读出时长已对齐的 `(帧列表, 波形, 采样率)`：

    ```python
    source_video, source_audio, audio_sample_rate = read_video_audio(
        "video.mp4", height=480, width=832, num_frames=124, fps=24,
        audio_sample_rate=pipe.audio_vae.sample_rate,
    )

    def align_to_clips(start, end, total_frames, clip_frames=17):
        """把左闭右开的帧区间 [start, end) 扩展到完整 clip，帧从 0 计数。"""
        first_clip, last_clip = start // clip_frames, (end - 1) // clip_frames
        return first_clip * clip_frames, min((last_clip + 1) * clip_frames, total_frames)

    video, audio = pipe(
        prompt=prompt, height=480, width=832, num_frames=124,
        retake_video=source_video,
        frame_regions_to_retake=[align_to_clips(24, 48, 124)],   # 帧 [24,48) -> (17, 51)
        retake_audio=source_audio,
        retake_audio_sample_rate=audio_sample_rate,
        seconds_regions_to_retake=[(0, 1), (4, 5)],              # 秒
    )
    ```
* `progress_bar_cmd`: 进度条，默认为 `tqdm`。可通过设置为 `lambda x: x` 来屏蔽进度条。

Pipeline 返回 `(video, audio)` 二元组，视频为 PIL 图像列表，音频为波形张量，可通过 `diffsynth.utils.data.audio_video.write_video_audio` 混流写出 MP4：

```python
write_video_audio(video=video, audio=audio, output_path="video.mp4", fps=24, audio_sample_rate=pipe.audio_vae.sample_rate)
```

如果显存不足，请开启[显存管理](../Pipeline_Usage/VRAM_management.md)，我们在示例代码中提供了每个模型推荐的低显存配置，详见前文“模型总览”中的表格。此外我们还提供了 NF4 量化版本的权重，可进一步降低显存需求，对应脚本同样见“模型总览”表格。

## 模型训练

MiniMax-H3 系列模型统一通过 [`examples/minimax_h3/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/train.py) 进行训练，脚本的参数包括：

* 通用训练参数
    * 数据集基础配置
        * `--dataset_base_path`: 数据集的根目录。
        * `--dataset_metadata_path`: 数据集的元数据文件路径。
        * `--dataset_repeat`: 每个 epoch 中数据集重复的次数。
        * `--dataset_num_workers`: 每个 Dataloader 的进程数量。
        * `--data_file_keys`: 元数据中需要加载的字段名称，通常是图像或视频文件的路径，以 `,` 分隔。
    * 模型加载配置
        * `--model_paths`: 要加载的模型路径。JSON 格式。
        * `--model_id_with_origin_paths`: 带原始路径的模型 ID。用逗号分隔。
        * `--extra_inputs`: 模型 Pipeline 所需的额外输入参数，以 `,` 分隔。
        * `--fp8_models`: 以 FP8 格式加载的模型，目前仅支持参数不被梯度更新的模型。
    * 训练基础配置
        * `--learning_rate`: 学习率。
        * `--num_epochs`: 轮数（Epoch）。
        * `--trainable_models`: 可训练的模型，例如 `dit`、`vae`、`text_encoder`。
        * `--find_unused_parameters`: DDP 训练中是否存在未使用的参数。
        * `--weight_decay`: 权重衰减大小。
        * `--task`: 训练任务，默认为 `sft`。
    * 输出配置
        * `--output_path`: 模型保存路径。
        * `--remove_prefix_in_ckpt`: 在模型文件的 state dict 中移除前缀。
        * `--save_steps`: 保存模型的训练步数间隔。
    * LoRA 配置
        * `--lora_base_model`: LoRA 添加到哪个模型上。
        * `--lora_target_modules`: LoRA 添加到哪些层上。
        * `--lora_rank`: LoRA 的秩（Rank）。
        * `--lora_checkpoint`: LoRA 检查点的路径。
        * `--preset_lora_path`: 预置 LoRA 检查点路径，用于 LoRA 差分训练。
        * `--preset_lora_model`: 预置 LoRA 融入的模型，例如 `dit`。
    * 梯度配置
        * `--use_gradient_checkpointing`: 是否启用 gradient checkpointing。
        * `--use_gradient_checkpointing_offload`: 是否将 gradient checkpointing 卸载到内存中。
        * `--gradient_accumulation_steps`: 梯度累积步数。
    * 分辨率配置
        * `--height`: 视频的高度。将 `height` 和 `width` 留空以启用动态分辨率。
        * `--width`: 视频的宽度。将 `height` 和 `width` 留空以启用动态分辨率。
        * `--max_pixels`: 最大像素面积，动态分辨率时大于此值的图片会被缩小。
        * `--num_frames`: 视频的帧数。
* MiniMax-H3 专有参数
    * `--processor_path`: Qwen3-VL processor 的路径，支持 `model_id:origin_file_pattern` 形式，用于对 prompt 进行 tokenize。
    * `--initialize_model_on_cpu`: 是否在 CPU 上初始化模型。

我们构建了一个样例数据集，以方便您进行测试，通过以下命令可以下载这个数据集：

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

LoRA 训练脚本采用两阶段流程：先以 `--task "sft:data_process"` 预处理并缓存数据集，再以 `--task "sft:train"` 执行正式训练。之所以必须分阶段，是因为 DiT 与 Qwen3-VL 文本编码器无法同时载入单卡。LoRA 默认作用于 DiT 的 `qkv_proj,out_proj` 模块，rank 为 32。全量训练同样采用两阶段流程，第二阶段通过 [`accelerate_config_zero3.yaml`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/full/accelerate_config_zero3.yaml) 启用 DeepSpeed ZeRO-3，并以 `--trainable_models "dit"` 指定训练对象。

NF4 量化版本的 LoRA 训练为单阶段流程：量化后所有组件可同时载入单卡，无需预先缓存数据集。此时必须显式指定 `--lora_target_modules`，因为量化权重在 state dict 中以打包形式存储，自动探测无法识别其形状。

首尾帧引导（FL2VA）训练在 `--extra_inputs` 中追加 `input_image,end_image`，分别取训练视频的首帧与尾帧作为条件，数据集无需额外列。参考驱动（Ref2VA）训练使用 `metadata.json`，其中 `references` 字段为一组参考块，支持 `image`、`video`、`audio`、`video_audio` 四种类型：

```json
[
  {
    "video": "train_video.mp4",
    "prompt": "...",
    "input_audio": "train_video.mp4",
    "references": [
      {"type": "image", "image": "0.png"}
    ],
    "frame_rate": 24
  }
]
```

`references` 需同时出现在 `--data_file_keys` 与 `--extra_inputs` 中，前者负责按类型加载文件，后者负责将参考块注入 Pipeline。参考图像以原生分辨率交给 Pipeline（其内部会按参考短边重新缩放），参考视频则裁剪到训练画布并按 24fps 采样。

我们为每个模型编写了推荐的训练脚本，请参考前文“模型总览”中的表格。关于如何编写模型训练脚本，请参考[模型训练](../Pipeline_Usage/Model_Training.md)；更多高阶训练算法，请参考[训练框架详解](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/zh/Training/)。
