# Boogu-Image

Boogu-Image 支持文生图、图生图和指令引导的图像编辑。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。

## 快速开始

运行以下代码可以快速加载 [Boogu/Boogu-Image-0.1-Base](https://modelscope.cn/models/Boogu/Boogu-Image-0.1-Base) 模型并进行推理。显存管理已启动，框架会自动根据剩余显存控制模型参数的加载，最低 8G 显存即可运行。

```python
from diffsynth.pipelines.boogu_image import BooguImagePipeline, ModelConfig
import torch


vram_config = {
    "offload_dtype": torch.float8_e4m3fn,
    "offload_device": "cpu",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = BooguImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Boogu/Boogu-Image-0.1-Base", origin_file_pattern="transformer/*.safetensors", **vram_config),
        ModelConfig(model_id="Boogu/Boogu-Image-0.1-Base", origin_file_pattern="mllm/*.safetensors", **vram_config),
        ModelConfig(model_id="Boogu/Boogu-Image-0.1-Base", origin_file_pattern="vae/*.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="Boogu/Boogu-Image-0.1-Base", origin_file_pattern="mllm/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

output = pipe(
    prompt="a cat",
    negative_prompt="",
    height=1024,
    width=1024,
    seed=42,
    num_inference_steps=50,
    cfg_scale=4.0,
)
output.save("image_Boogu-Image-0.1-Base.jpg")
```

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
|[Boogu/Boogu-Image-0.1-Base](https://modelscope.cn/models/Boogu/Boogu-Image-0.1-Base)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_inference/Boogu-Image-0.1-Base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_inference_low_vram/Boogu-Image-0.1-Base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_training/full/Boogu-Image-0.1-Base.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_training/validate_full/Boogu-Image-0.1-Base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_training/lora/Boogu-Image-0.1-Base.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_training/validate_lora/Boogu-Image-0.1-Base.py)|
|[Boogu/Boogu-Image-0.1-Turbo](https://modelscope.cn/models/Boogu/Boogu-Image-0.1-Turbo)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_inference/Boogu-Image-0.1-Turbo.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_inference_low_vram/Boogu-Image-0.1-Turbo.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_training/full/Boogu-Image-0.1-Turbo.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_training/validate_full/Boogu-Image-0.1-Turbo.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_training/lora/Boogu-Image-0.1-Turbo.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_training/validate_lora/Boogu-Image-0.1-Turbo.py)|
|[Boogu/Boogu-Image-0.1-Edit](https://modelscope.cn/models/Boogu/Boogu-Image-0.1-Edit)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_inference/Boogu-Image-0.1-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_inference_low_vram/Boogu-Image-0.1-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_training/full/Boogu-Image-0.1-Edit.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_training/validate_full/Boogu-Image-0.1-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_training/lora/Boogu-Image-0.1-Edit.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_training/validate_lora/Boogu-Image-0.1-Edit.py)|

## 模型推理

模型通过 `BooguImagePipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。

`BooguImagePipeline` 推理的输入参数包括：

* `prompt`: 文本提示词，用于描述期望的生成内容或编辑指令。
* `negative_prompt`: 负向提示词，指定不希望出现在结果中的内容，默认为空字符串。
* `cfg_scale`: 分类器自由引导的缩放系数，默认为 4.0。值越大，生成结果越贴近 prompt 描述。
* `input_image`: 输入图像，用于图生图（img2img）。提供后会根据 `denoising_strength` 对输入图像加噪再去噪。
* `edit_image`: 待编辑的图像，用于指令引导的图像编辑。提供后模型会根据 `prompt` 中的指令对图像进行修改。
* `height`: 输出图像的高度，默认为 1024。需能被 16 整除。
* `width`: 输出图像的宽度，默认为 1024。需能被 16 整除。
* `seed`: 随机种子，用于控制生成的可复现性。设为 `None` 时使用随机种子。
* `denoising_strength`: 降噪强度，控制输入图像被重绘的程度，默认为 1.0。仅在提供 `input_image` 时生效。
* `sigmas`: 自定义 sigma 调度序列，用于覆盖默认的调度策略。Turbo 模型需要指定此参数。
* `num_inference_steps`: 推理步数，默认为 20。步数越多，生成质量通常越好。
* `max_sequence_length`: 文本编码器处理的最大序列长度，默认为 1280。
* `max_input_image_pixels`: 输入图像的最大像素面积，默认为 4194304。超过此值的图像会被缩小。
* `max_input_image_side_length`: 输入图像的最大边长，默认为 4096。
* `max_vlm_input_pil_pixels`: VLM 输入图像的最大像素面积，默认为 147456。仅在图像编辑模式下生效。
* `max_vlm_input_pil_side_length`: VLM 输入图像的最大边长，默认为 768。仅在图像编辑模式下生效。
* `rand_device`: 生成初始噪声的设备，默认为 "cpu"。
* `progress_bar_cmd`: 进度条显示方式，默认为 tqdm。

显存不足时，请参考[显存管理](../Pipeline_Usage/VRAM_management.md)启用显存管理功能。

## 模型训练

boogu_image 系列模型统一通过 `examples/boogu_image/model_training/train.py` 进行训练，脚本的参数包括：

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
* Boogu-Image 专有参数
    * `--processor_path`: Processor 路径，用于处理文本和图像的编码器输入。
    * `--initialize_model_on_cpu`: 是否在 CPU 上初始化模型，默认在加速设备上初始化。

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

我们为每个模型编写了推荐的训练脚本，请参考前文"模型总览"中的表格。关于如何编写模型训练脚本，请参考[模型训练](../Pipeline_Usage/Model_Training.md)；更多高阶训练算法，请参考[训练框架详解](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/zh/Training/)。
