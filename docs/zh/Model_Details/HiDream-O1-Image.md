# HiDream-O1-Image

HiDream-O1-Image 是由 HiDream.ai 开源的基于 Pixel-Level Unified Transformer (UiT) 架构的图像生成模型。该模型将 VAE、DiT 和 TextEncoder 统一在单一的 Qwen3VLModel 中，直接在 pixel patch 空间进行扩散去噪，无需独立的 VAE 组件。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。

## 快速开始

运行以下代码可以快速加载 [HiDream-ai/HiDream-O1-Image](https://www.modelscope.cn/models/HiDream-ai/HiDream-O1-Image) 模型并进行推理。显存管理已启动，框架会自动根据剩余显存控制模型参数的加载，最低 3G 显存即可运行。

```python
from diffsynth.pipelines.hidream_o1_image import HiDreamO1ImagePipeline
from diffsynth.core.loader.config import ModelConfig
import torch


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


pipe = HiDreamO1ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="HiDream-ai/HiDream-O1-Image", origin_file_pattern="model-*.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="HiDream-ai/HiDream-O1-Image", origin_file_pattern="./"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
image = pipe(
    prompt="medium shot, eye-level, front view. A woman is seated in an ornate bedroom, illuminated by candlelight, with a calm and composed expression. The subject is a young woman with fair skin, light brown hair styled in an updo with loose tendrils framing her face, and blue eyes. She wears a cream-colored satin robe with delicate floral embroidery and lace trim along the neckline. Her ears are adorned with pearl drop earrings. She is seated on a bed with a dark, intricately carved wooden headboard. To her left, a wooden nightstand holds three lit white candles and a candelabra with multiple lit candles in the background. The bed is covered with patterned pillows and a dark, textured blanket. The walls are paneled with dark wood and feature a large, ornate tapestry with muted earth tones. The lighting creates soft highlights on her face and robe, with warm shadows cast across the room.",
    negative_prompt=" ",
    cfg_scale=4.0,
    height=2048,
    width=2048,
    seed=42,
    num_inference_steps=50,
)
image.save("image.jpg")
```

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
|[HiDream-ai/HiDream-O1-Image](https://www.modelscope.cn/models/HiDream-ai/HiDream-O1-Image)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hidream_o1_image/model_inference/HiDream-O1-Image.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hidream_o1_image/model_inference_low_vram/HiDream-O1-Image.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hidream_o1_image/model_training/full/HiDream-O1-Image.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hidream_o1_image/model_training/validate_full/HiDream-O1-Image.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hidream_o1_image/model_training/lora/HiDream-O1-Image.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hidream_o1_image/model_training/validate_lora/HiDream-O1-Image.py)|
|[HiDream-ai/HiDream-O1-Image-Dev](https://www.modelscope.cn/models/HiDream-ai/HiDream-O1-Image-Dev)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hidream_o1_image/model_inference/HiDream-O1-Image-Dev.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hidream_o1_image/model_inference_low_vram/HiDream-O1-Image-Dev.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hidream_o1_image/model_training/full/HiDream-O1-Image-Dev.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hidream_o1_image/model_training/validate_full/HiDream-O1-Image-Dev.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hidream_o1_image/model_training/lora/HiDream-O1-Image-Dev.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hidream_o1_image/model_training/validate_lora/HiDream-O1-Image-Dev.py)|

## 模型推理

模型通过 `HiDreamO1ImagePipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。

`HiDreamO1ImagePipeline` 推理的输入参数包括：

* `prompt`: 文本提示词。
* `negative_prompt`: 负向提示词，默认为 `" "`。
* `cfg_scale`: Classifier-Free Guidance 缩放系数，默认为 4.0。Dev 模型建议设为 1.0。
* `height`: 输出图像高度，默认为 2048。
* `width`: 输出图像宽度，默认为 2048。
* `seed`: 随机种子，默认为随机。
* `rand_device`: 噪声生成设备，默认为 `"cpu"`。
* `num_inference_steps`: 推理步数，Full 模型默认为 50，Dev 模型默认为 28。
* `model_type`: 模型类型，`"full"` 表示 Full 模型，`"dev"` 表示 Dev 蒸馏模型。
* `shift`: 时间步偏移量，影响 sigma 计算，默认为 3.0。
* `noise_scale`: 噪声缩放系数，默认为 8.0，Dev 模型建议设为 7.5。
* `edit_image`: 参考图像列表，用于图像编辑功能。默认为 None（文生图模式）。
* `keep_original_aspect`: 是否保持参考图像原始宽高比，默认为 True。

> **显存提示**: HiDream-O1-Image 模型参数量较大（~8B），生成 2048x2048 图像时建议开启显存管理（vram_config），或使用低显存推理脚本。

## 模型训练

hidream_o1_image 系列模型统一通过 `examples/hidream_o1_image/model_training/train.py` 进行训练，脚本的参数包括：

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
* HiDream-O1-Image 专有参数
    * `--processor_config`: Processor 配置文件路径，用于加载 AutoProcessor 进行文本 tokenization。
    * `--noise_scale`: 噪声缩放系数，默认为 8.0。
    * `--initialize_model_on_cpu`: 是否在 CPU 上初始化模型，启用后可降低 GPU 显存峰值。

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

我们为每个模型编写了推荐的训练脚本，请参考前文"模型总览"中的表格。关于如何编写模型训练脚本，请参考[模型训练](../Pipeline_Usage/Model_Training.md)；更多高阶训练算法，请参考[训练框架详解](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/zh/Training/)。
