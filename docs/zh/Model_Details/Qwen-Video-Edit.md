# Qwen-Video-Edit

Qwen-Video-Edit 是基于 Qwen-Image 架构的视频编辑模型。该模型接收一段输入视频和文本提示词，生成符合提示词描述的编辑后视频。模型采用 QwenImageDiT 作为核心 DiT 主干，结合 Wan2.1 VAE 进行视频编解码，并通过 QwenVideoEditAdapter 将视频特征投影到 DiT 的特征空间中。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。

## 快速开始

运行以下代码可以快速加载 [yunpeng1998/Qwen-Video-Edit](https://www.modelscope.cn/models/yunpeng1998/Qwen-Video-Edit) 模型并进行推理。显存管理已启动，框架会自动根据剩余显存控制模型参数的加载。

```python
import torch
from modelscope import dataset_snapshot_download
from diffsynth.core import ModelConfig
from diffsynth.pipelines.qwen_video_edit import QwenVideoEditPipeline
from diffsynth.utils.data import VideoData, save_video

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

dataset_snapshot_download(
    "DiffSynth-Studio/diffsynth_example_dataset",
    local_dir="data/diffsynth_example_dataset",
    allow_file_pattern="qwen_video_edit/Qwen-Video-Edit/*"
)

edit_video = VideoData("data/diffsynth_example_dataset/qwen_video_edit/Qwen-Video-Edit/source.mp4")
prompts = [
    "Transform the video into Japanese anime style",
]
pipe = QwenVideoEditPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="yunpeng1998/Qwen-Video-Edit", origin_file_pattern="360P/step-30000.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
    ],
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
video = pipe(edit_video=edit_video, prompts=prompts, height=640, width=384, num_frames=45, cfg_scale=4.0, num_inference_steps=40, seed=0)
save_video(video, "video_Qwen-Video-Edit.mp4", fps=16)
```

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
|[yunpeng1998/Qwen-Video-Edit](https://www.modelscope.cn/models/yunpeng1998/Qwen-Video-Edit)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_video_edit/model_inference/Qwen-Video-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_video_edit/model_inference_low_vram/Qwen-Video-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_video_edit/model_training/full/Qwen-Video-Edit.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_video_edit/model_training/validate_full/Qwen-Video-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_video_edit/model_training/lora/Qwen-Video-Edit.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_video_edit/model_training/validate_lora/Qwen-Video-Edit.py)|

## 模型推理

模型通过 `QwenVideoEditPipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。

`QwenVideoEditPipeline` 推理的输入参数包括：

* `edit_video`: 输入视频，即待编辑的源视频。类型为 `list[PIL.Image.Image]`，通过 `VideoData` 加载。
* `num_frames`: 视频帧数，默认值为 45。模型以 45 帧为一个 chunk 进行处理，每个 chunk 对应 `prompts` 列表中的一条提示词。
* `height`: 视频高度，需保证高度为 16 的倍数。
* `width`: 视频宽度，需保证宽度为 16 的倍数。
* `tiled`: 是否启用 VAE 分块推理，默认为 `False`。设置为 `True` 时可显著减少 VAE 编解码阶段的显存占用，会产生少许误差，以及少量推理时间延长。
* `tile_size`: VAE 编解码阶段的分块大小，默认为 `(30, 52)`，仅在 `tiled=True` 时生效。
* `tile_stride`: VAE 编解码阶段的分块步长，默认为 `(15, 26)`，仅在 `tiled=True` 时生效，需保证其数值小于或等于 `tile_size`。
* `prompts`: 提示词列表，每个元素对应一个 chunk 的编辑指令。
* `negative_prompt`: 负向提示词，描述画面中不应该出现的内容，默认值为 `" "`。
* `cfg_scale`: Classifier-free guidance 的参数，默认值为 4，当设置为 1 时不再生效。
* `seed`: 随机种子。默认为 `None`，即完全随机。
* `rand_device`: 生成随机高斯噪声矩阵的计算设备，默认为 `"cpu"`。当设置为 `cuda` 时，在不同 GPU 上会导致不同的生成结果。
* `num_inference_steps`: 推理次数，默认值为 40。
* `zero_cond_t`: 是否在时间步 t=0 时将条件特征置零。
* `progress_bar_cmd`: 进度条，默认为 `tqdm.tqdm`。可通过设置为 `lambda x:x` 来屏蔽进度条。

## 模型训练

Qwen-Video-Edit 通过 [`examples/qwen_video_edit/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_video_edit/model_training/train.py) 进行训练，脚本的参数包括：

* 通用训练参数
    * 数据集基础配置
        * `--dataset_base_path`: 数据集的根目录。
        * `--dataset_metadata_path`: 数据集的元数据文件路径。
        * `--dataset_repeat`: 每个 epoch 中数据集重复的次数。
        * `--dataset_num_workers`: 每个 Dataloader 的进程数量。
        * `--data_file_keys`: 元数据中需要加载的字段名称，以 `,` 分隔。Qwen-Video-Edit 需要设置为 `"input_video,video"`，其中 `input_video` 是条件视频（源视频），`video` 是目标视频。
    * 模型加载配置
        * `--model_paths`: 要加载的模型路径。JSON 格式。
        * `--model_id_with_origin_paths`: 带原始路径的模型 ID，例如 `"yunpeng1998/Qwen-Video-Edit:360P/step-30000.safetensors"`。用逗号分隔。
        * `--extra_inputs`: 模型 Pipeline 所需的额外输入参数，以 `,` 分隔。
        * `--fp8_models`：以 FP8 格式加载的模型，格式与 `--model_paths` 或 `--model_id_with_origin_paths` 一致，目前仅支持参数不被梯度更新的模型。
        * `--quant_options`：对加载的模型进行动态量化。以 `;` 分隔多个条目，每个为 `<模型字符串>:<method>[/<exclude_modules>]`。
    * 训练基础配置
        * `--learning_rate`: 学习率。
        * `--num_epochs`: 轮数（Epoch）。
        * `--trainable_models`: 可训练的模型，例如 `dit`、`adapter`。
        * `--find_unused_parameters`: DDP 训练中是否存在未使用的参数，需开启这一设置避免在多 GPU 训练中报错。
        * `--weight_decay`：权重衰减大小，详见 [torch.optim.AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)。
        * `--task`: 训练任务，默认为 `sft`。
    * 输出配置
        * `--output_path`: 模型保存路径。
        * `--remove_prefix_in_ckpt`: 在模型文件的 state dict 中移除前缀。
        * `--save_steps`: 保存模型的训练步数间隔，若此参数留空，则每个 epoch 保存一次。
    * LoRA 配置
        * `--lora_base_model`: LoRA 添加到哪个模型上。
        * `--lora_target_modules`: LoRA 添加到哪些层上。
        * `--lora_rank`: LoRA 的秩（Rank）。
        * `--lora_checkpoint`: LoRA 检查点的路径。如果提供此路径，LoRA 将从此检查点加载。
        * `--preset_lora_path`: 预置 LoRA 检查点路径，如果提供此路径，这一 LoRA 将会以融入基础模型的形式加载。
        * `--preset_lora_model`: 预置 LoRA 融入的模型，例如 `dit`。
    * 梯度配置
        * `--use_gradient_checkpointing`: 是否启用 gradient checkpointing。
        * `--use_gradient_checkpointing_offload`: 是否将 gradient checkpointing 卸载到内存中。
        * `--gradient_accumulation_steps`: 梯度累积步数。
    * 视频宽高配置
        * `--height`: 视频的高度。
        * `--width`: 视频的宽度。
        * `--num_frames`: 视频的帧数，默认为 45。
        * `--max_pixels`: 视频帧的最大像素面积。
* Qwen-Video-Edit 专有参数
    * `--tokenizer_path`: tokenizer 的路径，留空则自动从远程下载。
    * `--processor_path`: processor 的路径，留空则自动从远程下载。
    * `--zero_cond_t`: 是否在时间步 t=0 时将条件特征置零。

我们构建了一个样例视频数据集，以方便您进行测试，通过以下命令可以下载这个数据集：

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "qwen_video_edit/Qwen-Video-Edit/*" --local_dir ./data/diffsynth_example_dataset
```

我们为模型编写了推荐的训练脚本，请参考前文"模型总览"中的表格。关于如何编写模型训练脚本，请参考[模型训练](../Pipeline_Usage/Model_Training.md)；更多高阶训练算法，请参考[训练框架详解](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/zh/Training/)。
