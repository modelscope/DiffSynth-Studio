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
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
    ],
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)
prompt = "A girl is very happy, she is speaking in english: “I enjoy working with Diffsynth-Studio, it's a perfect framework.”"
video, audio = pipe(
    prompt=prompt,
    height=480, width=832, num_frames=124,
    num_inference_steps=50, seed=0,
)
write_video_audio(
    video=video,
    audio=audio,
    output_path="minimax_h3_t2va_quant_nf4.mp4",
    fps=24,
    audio_sample_rate=pipe.audio_vae.sample_rate,
)
print("saved minimax_h3_t2va_quant_nf4.mp4", "frames:", len(video), "audio:", tuple(audio.shape))
```

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
|[MiniMax/MiniMax-H3: TI2VA](https://www.modelscope.cn/models/MiniMax/MiniMax-H3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-TI2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-TI2VA.py)|-|-|-|-|
|[MiniMax/MiniMax-H3: Ref2VA](https://www.modelscope.cn/models/MiniMax/MiniMax-H3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-Ref2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Ref2VA.py)|-|-|-|-|
|[DiffSynth-Studio/MiniMax-H3-NF4: TI2VA](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-TI2VA-nf4.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-TI2VA-nf4.py)|-|-|-|-|
|[DiffSynth-Studio/MiniMax-H3-NF4: Ref2VA](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-Ref2VA-nf4.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Ref2VA-nf4.py)|-|-|-|-|

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
* `imgvid_cond_noise_aug`: 图像 / 视频条件的噪声增强锚定时间步，默认值为 0.999。
* `references`: 参考条件列表，按请求顺序给出，每个元素为字典，支持以下四种形式：
    * `{"type": "image", "image": PIL.Image}`
    * `{"type": "video", "video": list[PIL.Image]}`（无声视频）
    * `{"type": "audio", "audio": Tensor[C, L], "sample_rate": int}`
    * `{"type": "video_audio", "video": list[PIL.Image], "audio": Tensor[C, L], "sample_rate": int}`

    其中传入的视频帧列表必须已经是 24fps，Pipeline 不会重采样帧率。
* `audio_cond_noise_aug`: 参考音频条件的噪声增强锚定时间步，默认值为 1.0。
* `progress_bar_cmd`: 进度条，默认为 `tqdm`。可通过设置为 `lambda x: x` 来屏蔽进度条。

Pipeline 返回 `(video, audio)` 二元组，视频为 PIL 图像列表，音频为波形张量，可通过 `diffsynth.utils.data.audio_video.write_video_audio` 混流写出 MP4：

```python
write_video_audio(video=video, audio=audio, output_path="video.mp4", fps=24, audio_sample_rate=pipe.audio_vae.sample_rate)
```

如果显存不足，请开启[显存管理](../Pipeline_Usage/VRAM_management.md)，我们在示例代码中提供了每个模型推荐的低显存配置，详见前文“模型总览”中的表格。此外我们还提供了 NF4 量化版本的权重，可进一步降低显存需求，对应脚本同样见“模型总览”表格。
