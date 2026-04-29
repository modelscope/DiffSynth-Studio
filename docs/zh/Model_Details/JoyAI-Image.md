# JoyAI-Image

JoyAI-Image 是京东开源的统一多模态基础模型，支持图像理解、文生图生成和指令引导的图像编辑。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。

## 快速开始

运行以下代码可以快速加载 [jd-opensource/JoyAI-Image-Edit](https://modelscope.cn/models/jd-opensource/JoyAI-Image-Edit) 模型并进行推理。显存管理已启动，框架会自动根据剩余显存控制模型参数的加载，最低 4G 显存即可运行。

```python
from diffsynth.pipelines.joyai_image import JoyAIImagePipeline, ModelConfig
import torch
from PIL import Image
from modelscope import dataset_snapshot_download

# Download dataset
dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/diffsynth_example_dataset",
    local_dir="data/diffsynth_example_dataset",
    allow_file_pattern="joyai_image/JoyAI-Image-Edit/*"
)

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

pipe = JoyAIImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="jd-opensource/JoyAI-Image-Edit", origin_file_pattern="transformer/transformer.pth", **vram_config),
        ModelConfig(model_id="jd-opensource/JoyAI-Image-Edit", origin_file_pattern="JoyAI-Image-Und/model*.safetensors", **vram_config),
        ModelConfig(model_id="jd-opensource/JoyAI-Image-Edit", origin_file_pattern="vae/Wan2.1_VAE.pth", **vram_config),
    ],
    processor_config=ModelConfig(model_id="jd-opensource/JoyAI-Image-Edit", origin_file_pattern="JoyAI-Image-Und/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

# Use first sample from dataset
dataset_base_path = "data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit"
prompt = "将裙子改为粉色"
edit_image = Image.open(f"{dataset_base_path}/edit/image1.jpg").convert("RGB")

output = pipe(
    prompt=prompt,
    edit_image=edit_image,
    height=1024,
    width=1024,
    seed=0,
    num_inference_steps=30,
    cfg_scale=5.0,
)

output.save("output_joyai_edit_low_vram.png")
```

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
|[jd-opensource/JoyAI-Image-Edit](https://modelscope.cn/models/jd-opensource/JoyAI-Image-Edit)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/joyai_image/model_inference/JoyAI-Image-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/joyai_image/model_inference_low_vram/JoyAI-Image-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/joyai_image/model_training/full/JoyAI-Image-Edit.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/joyai_image/model_training/validate_full/JoyAI-Image-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/joyai_image/model_training/lora/JoyAI-Image-Edit.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/joyai_image/model_training/validate_lora/JoyAI-Image-Edit.py)|

## 模型推理

模型通过 `JoyAIImagePipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。

`JoyAIImagePipeline` 推理的输入参数包括：

* `prompt`: 文本提示词，用于描述期望的图像编辑效果。
* `negative_prompt`: 负向提示词，指定不希望出现在结果中的内容，默认为空字符串。
* `cfg_scale`: 分类器自由引导的缩放系数，默认为 5.0。值越大，生成结果越贴近 prompt 描述。
* `edit_image`: 待编辑的单张图像。
* `denoising_strength`: 降噪强度，控制输入图像被重绘的程度，默认为 1.0。
* `height`: 输出图像的高度，默认为 1024。需能被 16 整除。
* `width`: 输出图像的宽度，默认为 1024。需能被 16 整除。
* `seed`: 随机种子，用于控制生成的可复现性。设为 `None` 时使用随机种子。
* `max_sequence_length`: 文本编码器处理的最大序列长度，默认为 4096。
* `num_inference_steps`: 推理步数，默认为 30。步数越多，生成质量通常越好。
* `tiled`: 是否启用分块处理，用于降低显存占用，默认为 False。
* `tile_size`: 分块大小，默认为 (30, 52)。
* `tile_stride`: 分块步幅，默认为 (15, 26)。
* `shift`: 调度器的 shift 参数，用于控制 Flow Match 的调度曲线，默认为 4.0。
* `progress_bar_cmd`: 进度条显示方式，默认为 tqdm。

## 模型训练

joyai_image 系列模型统一通过 `examples/joyai_image/model_training/train.py` 进行训练，脚本的参数包括：

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
        * `--height`: 图像/视频的高度。留空启用动态分辨率。
        * `--width`: 图像/视频的宽度。留空启用动态分辨率。
        * `--max_pixels`: 最大像素面积，动态分辨率时大于此值的图片会被缩小。
        * `--num_frames`: 视频的帧数（仅视频生成模型）。
* JoyAI-Image 专有参数
    * `--processor_path`: Processor 路径，用于处理文本和图像的编码器输入。
    * `--initialize_model_on_cpu`: 是否在 CPU 上初始化模型，默认在加速设备上初始化。

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

关于如何编写模型训练脚本，请参考[模型训练](../Pipeline_Usage/Model_Training.md)；更多高阶训练算法，请参考[训练框架详解](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/zh/Training/)。
