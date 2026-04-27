# Stable Diffusion XL

Stable Diffusion XL (SDXL) 是由 Stability AI 开发的开源扩散式文本到图像生成模型，支持 1024x1024 分辨率的高质量文本到图像生成，采用双文本编码器（CLIP-L + CLIP-bigG）架构。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。

## 快速开始

运行以下代码可以快速加载 [stabilityai/stable-diffusion-xl-base-1.0](https://www.modelscope.cn/models/stabilityai/stable-diffusion-xl-base-1.0) 模型并进行推理。显存管理已启动，框架会自动根据剩余显存控制模型参数的加载，最低 6GB 显存即可运行。

```python
import torch
from diffsynth.core import ModelConfig
from diffsynth.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline

vram_config = {
    "offload_dtype": torch.float32,
    "offload_device": "cpu",
    "onload_dtype": torch.float32,
    "onload_device": "cpu",
    "preparing_dtype": torch.float32,
    "preparing_device": "cuda",
    "computation_dtype": torch.float32,
    "computation_device": "cuda",
}
pipe = StableDiffusionXLPipeline.from_pretrained(
    torch_dtype=torch.float32,
    model_configs=[
        ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="text_encoder/model.safetensors", **vram_config),
        ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="text_encoder_2/model.safetensors", **vram_config),
        ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="unet/diffusion_pytorch_model.safetensors", **vram_config),
        ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="tokenizer/"),
    tokenizer_2_config=ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="tokenizer_2/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

image = pipe(
    prompt="a photo of an astronaut riding a horse on mars",
    negative_prompt="",
    cfg_scale=5.0,
    height=1024,
    width=1024,
    seed=42,
    num_inference_steps=50,
)
image.save("image.jpg")
```

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
|[stabilityai/stable-diffusion-xl-base-1.0](https://www.modelscope.cn/models/stabilityai/stable-diffusion-xl-base-1.0)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/stable_diffusion_xl/model_inference/stable-diffusion-xl-base-1.0.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/stable_diffusion_xl/model_inference_low_vram/stable-diffusion-xl-base-1.0.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/stable_diffusion_xl/model_training/full/stable-diffusion-xl-base-1.0.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/stable_diffusion_xl/model_training/validate_full/stable-diffusion-xl-base-1.0.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/stable_diffusion_xl/model_training/lora/stable-diffusion-xl-base-1.0.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/stable_diffusion_xl/model_training/validate_lora/stable-diffusion-xl-base-1.0.py)|

## 模型推理

模型通过 `StableDiffusionXLPipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。

`StableDiffusionXLPipeline` 的推理输入参数包括：

* `prompt`: 文本提示词。
* `negative_prompt`: 负面提示词，默认为空字符串。
* `cfg_scale`: Classifier-Free Guidance 缩放系数，默认 5.0。
* `height`: 输出图像高度，默认 1024。
* `width`: 输出图像宽度，默认 1024。
* `seed`: 随机种子，默认不设置时使用随机种子。
* `rand_device`: 噪声生成设备，默认 "cpu"。
* `num_inference_steps`: 推理步数，默认 50。
* `guidance_rescale`: Guidance rescale 系数，默认 0.0。
* `progress_bar_cmd`: 进度条回调函数。

> `StableDiffusionXLPipeline` 需要双 tokenizer 配置（`tokenizer_config` 和 `tokenizer_2_config`），分别对应 CLIP-L 和 CLIP-bigG 文本编码器。

## 模型训练

stable_diffusion_xl 系列模型通过 `examples/stable_diffusion_xl/model_training/train.py` 进行训练，脚本的参数包括：

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
* Stable Diffusion XL 专有参数
    * `--tokenizer_path`: 第一个 Tokenizer 路径。
    * `--tokenizer_2_path`: 第二个 Tokenizer 路径，默认为 `stabilityai/stable-diffusion-xl-base-1.0:tokenizer_2/`。

样例数据集下载：

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "stable_diffusion_xl/*" --local_dir ./data/diffsynth_example_dataset
```

[stable-diffusion-xl-base-1.0 训练脚本](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/stable_diffusion_xl/model_training/lora/stable-diffusion-xl-base-1.0.sh)

我们为每个模型编写了推荐的训练脚本，请参考前文"模型总览"中的表格。关于如何编写模型训练脚本，请参考[模型训练](../Pipeline_Usage/Model_Training.md)；更多高阶训练算法，请参考[训练框架详解](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/zh/Training/)。
