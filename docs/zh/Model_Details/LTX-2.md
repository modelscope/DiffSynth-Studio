# LTX-2

LTX-2 是由 Lightricks 开发的音视频生成模型系列。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](/docs/zh/Pipeline_Usage/Setup.md)。

## 快速开始

运行以下代码可以快速加载 [Lightricks/LTX-2](https://www.modelscope.cn/models/Lightricks/LTX-2) 模型并进行推理。显存管理已启动，框架会自动根据剩余显存控制模型参数的加载，最低 8GB 显存即可运行。

```python
import torch
from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig
from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2

vram_config = {
    "offload_dtype": torch.float8_e5m2,
    "offload_device": "cpu",
    "onload_dtype": torch.float8_e5m2,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e5m2,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = LTX2AudioVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized", origin_file_pattern="model-*.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-dev.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
prompt = "A girl is very happy, she is speaking: “I enjoy working with Diffsynth-Studio, it's a perfect framework.”"
negative_prompt = "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
height, width, num_frames = 512, 768, 121
video, audio = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    seed=43,
    height=height,
    width=width,
    num_frames=num_frames,
    tiled=True,
)
write_video_audio_ltx2(
    video=video,
    audio=audio,
    output_path='ltx2_onestage.mp4',
    fps=24,
    audio_sample_rate=24000,
)
```

## 模型总览
|模型 ID|额外参数|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|-|
|[Lightricks/LTX-2: OneStagePipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2)||[code](/examples/ltx2/model_inference/LTX-2-T2AV-OneStage.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-OneStage.py)|-|-|-|-|
|[Lightricks/LTX-2: TwoStagePipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2)||[code](/examples/ltx2/model_inference/LTX-2-T2AV-TwoStage.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-TwoStage.py)|-|-|-|-|
|[Lightricks/LTX-2: DistilledPipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2)||[code](/examples/ltx2/model_inference/LTX-2-T2AV-DistilledPipeline.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-DistilledPipeline.py)|-|-|-|-|
|[Lightricks/LTX-2: OneStagePipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2)|`input_images`|[code](/examples/ltx2/model_inference/LTX-2-I2AV-OneStage.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2-I2AV-OneStage.py)|-|-|-|-|
|[Lightricks/LTX-2: TwoStagePipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2)|`input_images`|[code](/examples/ltx2/model_inference/LTX-2-I2AV-TwoStage.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2-I2AV-TwoStage.py)|-|-|-|-|
|[Lightricks/LTX-2: DistilledPipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2)|`input_images`|[code](/examples/ltx2/model_inference/LTX-2-I2AV-DistilledPipeline.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2-I2AV-DistilledPipeline.py)|-|-|-|-|

## 模型推理

模型通过 `LTX2AudioVideoPipeline.from_pretrained` 加载，详见[加载模型](/docs/zh/Pipeline_Usage/Model_Inference.md#加载模型)。

`LTX2AudioVideoPipeline` 推理的输入参数包括：

* `prompt`: 提示词，描述视频中出现的内容。
* `negative_prompt`: 负向提示词，描述视频中不应该出现的内容，默认值为 `""`。
* `cfg_scale`: Classifier-free guidance 的参数，默认值为 3.0。
* `input_images`: 输入图像列表，用于图生视频。
* `input_images_indexes`: 输入图像在视频中的帧索引列表。
* `input_images_strength`: 输入图像的强度，默认值为 1.0。
* `denoising_strength`: 去噪强度，范围是 0～1，默认值为 1.0。
* `seed`: 随机种子。默认为 `None`，即完全随机。
* `rand_device`: 生成随机高斯噪声矩阵的计算设备，默认为 `"cpu"`。当设置为 `cuda` 时，在不同 GPU 上会导致不同的生成结果。
* `height`: 视频高度，需保证高度为 32 的倍数（单阶段）或 64 的倍数（两阶段）。
* `width`: 视频宽度，需保证宽度为 32 的倍数（单阶段）或 64 的倍数（两阶段）。
* `num_frames`: 视频帧数，默认值为 121，需保证为 8 的倍数 + 1。
* `num_inference_steps`: 推理次数，默认值为 40。
* `tiled`: 是否启用 VAE 分块推理，默认为 `True`。设置为 `True` 时可显著减少 VAE 编解码阶段的显存占用，会产生少许误差，以及少量推理时间延长。
* `tile_size_in_pixels`: VAE 编解码阶段的像素分块大小，默认为 512。
* `tile_overlap_in_pixels`: VAE 编解码阶段的像素分块重叠大小，默认为 128。
* `tile_size_in_frames`: VAE 编解码阶段的帧分块大小，默认为 128。
* `tile_overlap_in_frames`: VAE 编解码阶段的帧分块重叠大小，默认为 24。
* `use_two_stage_pipeline`: 是否使用两阶段管道，默认为 `False`。
* `use_distilled_pipeline`: 是否使用蒸馏管道，默认为 `False`。
* `progress_bar_cmd`: 进度条，默认为 `tqdm.tqdm`。可通过设置为 `lambda x:x` 来屏蔽进度条。

如果显存不足，请开启[显存管理](/docs/zh/Pipeline_Usage/VRAM_management.md)，我们在示例代码中提供了每个模型推荐的低显存配置，详见前文"支持的推理脚本"中的表格。

## 模型训练

LTX-2 系列模型目前暂不支持训练功能。我们将尽快添加相关支持。
