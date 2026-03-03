# LTX-2

LTX-2 是由 Lightricks 开发的音视频生成模型系列。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。

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
"""
Offical model repo: https://www.modelscope.cn/models/Lightricks/LTX-2
Repackaged model repo: https://www.modelscope.cn/models/DiffSynth-Studio/LTX-2-Repackage
For base models of LTX-2, offical checkpoint (with model config ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-dev.safetensors"))
and repackaged checkpoints (with model config ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="*.safetensors")) are both supported.
We have repackeged the official checkpoints in DiffSynth-Studio/LTX-2-Repackage repo to support separate loading of different submodules,
and avoid redundant memory usage when users only want to use part of the model.
"""
# use the repackaged modelconfig from "DiffSynth-Studio/LTX-2-Repackage" to avoid redundant model loading
pipe = LTX2AudioVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized", origin_file_pattern="model-*.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="transformer.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="text_encoder_post_modules.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="video_vae_decoder.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="audio_vae_decoder.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="audio_vocoder.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="video_vae_encoder.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-spatial-upscaler-x2-1.0.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized"),
    stage2_lora_config=ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-distilled-lora-384.safetensors"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

# use the following modelconfig if you want to initialize model from offical checkpoints from "Lightricks/LTX-2"
# pipe = LTX2AudioVideoPipeline.from_pretrained(
#     torch_dtype=torch.bfloat16,
#     device="cuda",
#     model_configs=[
#         ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized", origin_file_pattern="model-*.safetensors", **vram_config),
#         ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-dev.safetensors", **vram_config),
#         ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-spatial-upscaler-x2-1.0.safetensors", **vram_config),
#     ],
#     tokenizer_config=ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized"),
#     stage2_lora_config=ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-distilled-lora-384.safetensors"),
#     vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
# )

prompt = "A girl is very happy, she is speaking: \"I enjoy working with Diffsynth-Studio, it's a perfect framework.\""
negative_prompt = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)
height, width, num_frames = 512 * 2, 768 * 2, 121
video, audio = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    seed=43,
    height=height,
    width=width,
    num_frames=num_frames,
    tiled=True,
    use_two_stage_pipeline=True,
)
write_video_audio_ltx2(
    video=video,
    audio=audio,
    output_path='ltx2_twostage.mp4',
    fps=24,
    audio_sample_rate=24000,
)
```

## 模型总览
|模型 ID|额外参数|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|-|
|[Lightricks/LTX-2: OneStagePipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2)||[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-T2AV-OneStage.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-OneStage.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_training/full/LTX-2-T2AV-splited.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_training/validate_full/LTX-2-T2AV.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_training/lora/LTX-2-T2AV-splited.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_training/validate_lora/LTX-2-T2AV.py)|
|[Lightricks/LTX-2-19b-IC-LoRA-Union-Control](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-IC-LoRA-Union-Control)|`in_context_videos`,`in_context_downsample_factor`|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-T2AV-IC-LoRA-Union-Control.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-IC-LoRA-Union-Control.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_training/lora/LTX-2-T2AV-IC-LoRA-splited.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_training/validate_lora/LTX-2-T2AV-IC-LoRA.py)|
|[Lightricks/LTX-2-19b-IC-LoRA-Detailer](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-IC-LoRA-Detailer)|`in_context_videos`,`in_context_downsample_factor`|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-T2AV-IC-LoRA-Detailer.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-IC-LoRA-Detailer.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_training/lora/LTX-2-T2AV-IC-LoRA-splited.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_training/validate_lora/LTX-2-T2AV-IC-LoRA.py)|
|[Lightricks/LTX-2: TwoStagePipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2)||[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-T2AV-TwoStage.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-TwoStage.py)|-|-|-|-|
|[Lightricks/LTX-2: DistilledPipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2)||[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-T2AV-DistilledPipeline.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-DistilledPipeline.py)|-|-|-|-|
|[Lightricks/LTX-2: OneStagePipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2)|`input_images`|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-I2AV-OneStage.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-I2AV-OneStage.py)|-|-|-|-|
|[Lightricks/LTX-2: TwoStagePipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2)|`input_images`|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-I2AV-TwoStage.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-I2AV-TwoStage.py)|-|-|-|-|
|[Lightricks/LTX-2: DistilledPipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2)|`input_images`|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-I2AV-DistilledPipeline.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-I2AV-DistilledPipeline.py)|-|-|-|-|
|[Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In)||[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Dolly-In.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Dolly-In.py)|-|-|-|-|
|[Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out)||[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Dolly-Out.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Dolly-Out.py)|-|-|-|-|
|[Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left)||[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Dolly-Left.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Dolly-Left.py)|-|-|-|-|
|[Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right)||[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Dolly-Right.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Dolly-Right.py)|-|-|-|-|
|[Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up)||[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Jib-Up.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Jib-Up.py)|-|-|-|-|
|[Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down)||[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Jib-Down.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Jib-Down.py)|-|-|-|-|
|[Lightricks/LTX-2-19b-LoRA-Camera-Control-Static](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Static)||[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Static.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Static.py)|-|-|-|-|

## 模型推理

模型通过 `LTX2AudioVideoPipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。

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

如果显存不足，请开启[显存管理](../Pipeline_Usage/VRAM_management.md)，我们在示例代码中提供了每个模型推荐的低显存配置，详见前文"支持的推理脚本"中的表格。

## 模型训练

LTX-2 系列模型统一通过 [`examples/ltx2/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_training/train.py) 进行训练，脚本的参数包括：

* 通用训练参数
    * 数据集基础配置
        * `--dataset_base_path`: 数据集的根目录。
        * `--dataset_metadata_path`: 数据集的元数据文件路径。
        * `--dataset_repeat`: 每个 epoch 中数据集重复的次数。
        * `--dataset_num_workers`: 每个 Dataloder 的进程数量。
        * `--data_file_keys`: 元数据中需要加载的字段名称，通常是图像或视频文件的路径，以 `,` 分隔。
    * 模型加载配置
        * `--model_paths`: 要加载的模型路径。JSON 格式。
        * `--model_id_with_origin_paths`: 带原始路径的模型 ID，例如 `"Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors"`。用逗号分隔。
        * `--extra_inputs`: 模型 Pipeline 所需的额外输入参数，例如训练图像编辑模型时需要额外参数，以 `,` 分隔。
        * `--fp8_models`：以 FP8 格式加载的模型，格式与 `--model_paths` 或 `--model_id_with_origin_paths` 一致，目前仅支持参数不被梯度更新的模型（不需要梯度回传，或梯度仅更新其 LoRA）。
    * 训练基础配置
        * `--learning_rate`: 学习率。
        * `--num_epochs`: 轮数（Epoch）。
        * `--trainable_models`: 可训练的模型，例如 `dit`、`vae`、`text_encoder`。
        * `--find_unused_parameters`: DDP 训练中是否存在未使用的参数，少数模型包含不参与梯度计算的冗余参数，需开启这一设置避免在多 GPU 训练中报错。
        * `--weight_decay`：权重衰减大小，详见 [torch.optim.AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)。
        * `--task`: 训练任务，默认为 `sft`，部分模型支持更多训练模式，请参考每个特定模型的文档。
    * 输出配置
        * `--output_path`: 模型保存路径。
        * `--remove_prefix_in_ckpt`: 在模型文件的 state dict 中移除前缀。
        * `--save_steps`: 保存模型的训练步数间隔，若此参数留空，则每个 epoch 保存一次。
    * LoRA 配置
        * `--lora_base_model`: LoRA 添加到哪个模型上。
        * `--lora_target_modules`: LoRA 添加到哪些层上。
        * `--lora_rank`: LoRA 的秩（Rank）。
        * `--lora_checkpoint`: LoRA 检查点的路径。如果提供此路径，LoRA 将从此检查点加载。
        * `--preset_lora_path`: 预置 LoRA 检查点路径，如果提供此路径，这一 LoRA 将会以融入基础模型的形式加载。此参数用于 LoRA 差分训练。
        * `--preset_lora_model`: 预置 LoRA 融入的模型，例如 `dit`。
    * 梯度配置
        * `--use_gradient_checkpointing`: 是否启用 gradient checkpointing。
        * `--use_gradient_checkpointing_offload`: 是否将 gradient checkpointing 卸载到内存中。
        * `--gradient_accumulation_steps`: 梯度累积步数。
    * 视频宽高配置
        * `--height`: 视频的高度。将 `height` 和 `width` 留空以启用动态分辨率。
        * `--width`: 视频的宽度。将 `height` 和 `width` 留空以启用动态分辨率。
        * `--max_pixels`: 视频帧的最大像素面积，当启用动态分辨率时，分辨率大于这个数值的视频帧都会被缩小，分辨率小于这个数值的视频帧保持不变。
        * `--num_frames`: 视频的帧数。
* LTX-2 系列特定参数
    * `--tokenizer_path`: 分词器路径，适用于文生视频模型，留空则从远程自动下载。
    * `--frame_rate`: 训练视频的帧率。

我们构建了一个样例视频数据集，以方便您进行测试，通过以下命令可以下载这个数据集：

```shell
modelscope download --dataset DiffSynth-Studio/example_video_dataset --local_dir ./data/example_video_dataset
```

我们为每个模型编写了推荐的训练脚本，请参考前文"模型总览"中的表格。关于如何编写模型训练脚本，请参考[模型训练](../Pipeline_Usage/Model_Training.md)；更多高阶训练算法，请参考[训练框架详解](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/zh/Training/)。
