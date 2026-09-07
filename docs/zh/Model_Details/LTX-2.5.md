# LTX-2.5

LTX-2.5 是 Lightricks 发布的音视频联合生成模型。DiffSynth-Studio 通过 `LTX2AudioVideoPipeline` 提供其推理与训练支持（与 LTX-2 / LTX-2.3 共用同一个 Pipeline 类）。相比 LTX-2.3，LTX-2.5 引入了微调版 Gemma4 12B 文本编码器（内嵌 tokenizer 资产与音视频双 connector）、DiffVAE 扩散视频解码器、Duration Head 自动时长预测，以及 INT8 量化权重。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。LTX-2.5 的 Gemma4 文本编码器需要 `transformers>=5.8,<5.15`。

## 快速开始

运行以下代码可以快速加载 [Lightricks/LTX-2.5](https://www.modelscope.cn/models/Lightricks/LTX-2.5) 模型并进行推理。开启 `auto_duration` 后，Pipeline 会先用 Duration Head 从提示词预测视频时长，再完成音视频生成，整个过程只需一次 `pipe(...)` 调用。

```python
import torch
from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig
from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cuda",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = LTX2AudioVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="vae/ltx-2.5-video-vae-bf16.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="vae/ltx-2.5-audio-vae-bf16.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="model_patches/ltx-2.5-duration-head-bf16.safetensors", **vram_config),
    ],
    load_duration_head=True,
)
prompt = "A girl is very happy, she is speaking: “I enjoy working with Diffsynth-Studio, it's a perfect framework.”"
negative_prompt = pipe.default_negative_prompt["LTX-2.3"]
video, audio = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    seed=43,
    height=1024, width=1536, frame_rate=24,
    auto_duration=True,
    cfg_scale=1.0, num_inference_steps=8,
    use_distilled_pipeline=True, use_two_stage_pipeline=True,
    tiled=True,
)
write_video_audio_ltx2(video=video, audio=audio, output_path='video.mp4', fps=24, audio_sample_rate=pipe.audio_vocoder.output_sampling_rate)
```

## 模型总览

|模型 ID|额外参数|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|-|
|[Lightricks/LTX-2.5: DistilledPipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2.5)|`auto_duration`,`load_duration_head=True`|[code](/examples/ltx2/model_inference/LTX-2.5-T2AV-DistilledPipeline.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2.5-T2AV-DistilledPipeline.py)|[code](/examples/ltx2/model_training/full/LTX-2.5-T2AV-splited.sh)|[code](/examples/ltx2/model_training/validate_full/LTX-2.5-T2AV.py)|[code](/examples/ltx2/model_training/lora/LTX-2.5-T2AV-splited.sh)|[code](/examples/ltx2/model_training/validate_lora/LTX-2.5-T2AV.py)|
|[Lightricks/LTX-2.5: DistilledPipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2.5)|`input_images`,`input_images_indexes`|[code](/examples/ltx2/model_inference/LTX-2.5-I2AV-DistilledPipeline.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2.5-I2AV-DistilledPipeline.py)|-|-|-|-|
|[Lightricks/LTX-2.5: TwoStagePipeline-A2V](https://www.modelscope.cn/models/Lightricks/LTX-2.5)|`retake_audio`,`audio_sample_rate`,`stage2_lora_config`|[code](/examples/ltx2/model_inference/LTX-2.5-A2V-TwoStage.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2.5-A2V-TwoStage.py)|-|-|-|-|
|[Lightricks/LTX-2.5: TwoStagePipeline-Retake](https://www.modelscope.cn/models/Lightricks/LTX-2.5)|`retake_video`,`retake_video_regions`,`stage2_lora_config`|[code](/examples/ltx2/model_inference/LTX-2.5-T2AV-TwoStage-Retake.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2.5-T2AV-TwoStage-Retake.py)|-|-|-|-|
|[Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler](https://www.modelscope.cn/models/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler)|`in_context_videos`,`in_context_downsample_factor`|[code](/examples/ltx2/model_inference/LTX-2.5-IC-LoRA-Pixel-Spatial-Upscaler.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2.5-IC-LoRA-Pixel-Spatial-Upscaler.py)|-|-|-|-|
|[Lightricks/LTX-2.5: T2A](https://www.modelscope.cn/models/Lightricks/LTX-2.5)|`generate_video=False`|[code](/examples/ltx2/model_inference/LTX-2.5-T2A.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2.5-T2A.py)|-|-|-|-|
|[Lightricks/LTX-2.5: INT8-ConvRot](https://www.modelscope.cn/models/Lightricks/LTX-2.5)|INT8 DiT + INT8 Gemma4|[code](/examples/ltx2/model_inference/LTX-2.5-T2AV-INT8-ConvRot.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2.5-T2AV-INT8-ConvRot.py)|-|-|-|-|

## 模型推理

模型通过 `LTX2AudioVideoPipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。LTX-2.5 与 LTX-2.3 共用 `LTX2AudioVideoPipeline`，框架根据加载到的权重自动识别模型版本。

`from_pretrained` 的 LTX-2.5 相关参数：

* `load_duration_head`: 是否要求加载 Duration Head（自动时长预测所需），默认为 `False`。
* `gemma_path`: Gemma4 权重路径，用于加载内嵌的 tokenizer 资产。留空时自动从 `model_configs` 中的 text encoder 路径推导。
* `stage2_lora_config`: Dev 权重两阶段推理时使用的第二阶段 distilled-LoRA。

`LTX2AudioVideoPipeline` 的通用推理参数见 [LTX-2 文档](LTX-2.md#模型推理)，LTX-2.5 新增或特有的参数为：

* `auto_duration`: 是否根据提示词自动预测视频时长，默认为 `False`。开启后无需传入 `num_frames`，需要加载 Duration Head。
* `auto_duration_min_seconds` / `auto_duration_max_seconds`: 自动时长的上下界（秒），默认为 1.0 和 20.0。
* `generate_video`: 是否生成视频，默认为 `True`。设置为 `False` 时只生成音频（T2A），此时无需加载视频 VAE 与 latent upsampler。
* `use_diffusion_vae`: 视频解码器选择。`None`（默认）表示按模型版本自动选择（LTX-2.5 使用 DiffVAE 扩散解码器），`False` 表示使用 ConvVAE 卷积解码器（需加载 `ltx-2.5-video-vae-conv-bf16.safetensors`）。
* `input_images` / `input_images_indexes`: 关键帧图像及其帧索引。传入首帧即为图生视频，传入首尾（或多帧）即为关键帧插值。
* `retake_audio` / `audio_sample_rate` / `retake_audio_regions`: 音频驱动视频（A2V）与音频区域重生成。

几何约束：`num_frames % 8 == 1`；单阶段的宽高为 32 的倍数，两阶段的宽高为 64 的倍数。

如果显存不足，请开启[显存管理](../Pipeline_Usage/VRAM_management.md)，我们在示例代码中提供了每个模型推荐的低显存配置（FP8 CPU 权重卸载 + 细粒度显存管理），详见前文"模型总览"中的表格。

## 模型训练

LTX-2.5 与 LTX-2 / LTX-2.3 共用训练脚本 [`examples/ltx2/model_training/train.py`](/examples/ltx2/model_training/train.py)，通用训练参数的说明见 [LTX-2 文档](LTX-2.md#模型训练)。

由于 22B DiT 与 12B Gemma4 编码器无法同时放入单卡，LTX-2.5 的训练脚本采用双阶段（splited）方案：

1. `--task "sft:data_process"`：运行文本编码与 VAE 编码，把结果缓存到硬盘。
2. `--task "sft:train"`：从缓存读取前处理结果，只训练 DiT。

两个阶段都保留完整的 `--model_id_with_origin_paths`，并用 `--fp8_models` 声明该阶段不需要前向的模型（阶段二的 TextEncoder 与 VAE 使用 FP8 加载）。训练数据集的字段为 `video,prompt,input_audio,frame_rate`，对应 `--data_file_keys "video,input_audio"` 与 `--extra_inputs "input_audio"`。

我们构建了一个样例视频数据集，以方便您进行测试，通过以下命令可以下载这个数据集：

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "ltx2/LTX-2.3-T2AV-splited/*" --local_dir ./data/diffsynth_example_dataset
```

训练完成后，可以使用 `examples/ltx2/model_training/validate_lora/LTX-2.5-T2AV.py`（LoRA）或 `examples/ltx2/model_training/validate_full/LTX-2.5-T2AV.py`（全量）加载训练产物进行推理验证。关于如何编写模型训练脚本，请参考[模型训练](../Pipeline_Usage/Model_Training.md)。

## 暂不支持的功能

以下 LTX-2.5 官方能力暂未接入：

* DFR（Diffusion Frame Rate）
* Native HDR / EXR 输出
* HDR IC-LoRA
* Dub-It 配音
