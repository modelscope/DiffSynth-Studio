# Qwen-Image-2.1

Qwen-Image-2.1 是由阿里巴巴通义实验室通义千问团队训练并开源的统一文生图与图像编辑模型，采用单流 block-causal DiT、64 通道 RGBA VAE 与 Qwen3-VL 文本编码器，可以直接生成带透明通道的 RGBA 图像。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。

## 快速开始

运行以下代码可以快速加载 [Qwen/Qwen-Image-2.1](https://www.modelscope.cn/models/Qwen/Qwen-Image-2.1) 模型并进行推理。显存管理已启动，框架会自动根据剩余显存控制模型参数的加载，最低 7G 显存即可运行。

```python
from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline, ModelConfig
import torch
from PIL import Image

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = QwenImage21Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="vae/diffusion_pytorch_model*.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

# Text-to-Image, the output is an RGBA image
prompt = "Flat anime-style illustration, a girl with long black hair, wearing a JK uniform."
image = pipe(prompt, seed=0)
image.save("image1.png")

prompt = "Flat anime-style illustration, a sunny and cheerful high school girl."
image_2 = pipe(prompt=prompt, seed=0)
image_2.save("image2.png")

# Image Editing, the generated RGBA image is fed back as the condition
prompt = "Generate a group photo of these two characters."
edit_image = [Image.open("image1.png"), Image.open("image2.png")]
image_3 = pipe(prompt, edit_image=edit_image, seed=1)
image_3.save("image3.png")
```

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
|[Qwen/Qwen-Image-2.1](https://www.modelscope.cn/models/Qwen/Qwen-Image-2.1)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_inference/Qwen-Image-2.1.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_inference_low_vram/Qwen-Image-2.1.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/full/Qwen-Image-2.1.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/validate_full/Qwen-Image-2.1.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/lora/Qwen-Image-2.1.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/validate_lora/Qwen-Image-2.1.py)|

## 模型推理

模型通过 `QwenImage21Pipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。

`QwenImage21Pipeline` 推理的输入参数包括：

* `prompt`: 提示词，描述画面中出现的内容。默认 `" "`；传入空字符串时会按单个空格处理，因为 Qwen 没有 bos token，空串会让文本编码器无内容可读。
* `negative_prompt`: 负向提示词，默认 `" "`，在 `cfg_scale` 大于 1 时作为负向分支条件。
* `cfg_scale`: CFG 强度，默认值为 1.0，是否启用 CFG 仅由该值是否大于 1 决定。
* `edit_image`: 待编辑图像，仅支持 PIL 图像（`PIL.Image`）或 PIL 图像列表。留空时执行文生图，提供时执行图像编辑。
* `height`: 图像高度，默认 1024，会对齐到 32 的倍数；编辑模式下 `edit_image` 按自身宽高比缩放到 `height * width` 的面积内。
* `width`: 图像宽度，默认 1024，规则同 `height`。
* `seed`: 随机种子。默认为 `None`，即完全随机。
* `rand_device`: 生成随机高斯噪声矩阵的计算设备，默认为 `"cpu"`。
* `num_inference_steps`: 推理次数，默认值为 40。
* `use_kv_cache`: 是否在 block-causal 条件下启用逐层 KV cache，默认值为 `True`。
* `tiled`: 是否启用 VAE 分块推理，默认为 `False`。
* `tile_size`: VAE 编解码阶段的分块大小，默认为 256，仅在 `tiled=True` 时生效。
* `tile_stride`: VAE 编解码阶段的分块步长，默认为 192，仅在 `tiled=True` 时生效。

Pipeline 的返回值为 PIL 图像（RGBA 模式）。

如果显存不足，请开启[显存管理](../Pipeline_Usage/VRAM_management.md)，我们在示例代码中提供了每个模型推荐的低显存配置，详见前文“模型总览”中的表格。

### 透明背景图像

模型的 VAE 工作在 4 通道 RGBA 像素空间，解码得到的第 4 个通道就是 alpha，因此透明度完全由模型根据提示词生成，Pipeline 不提供任何与 alpha 相关的开关。要得到透明背景的图像：

* 在提示词中明确声明这是一份带透明通道的素材，例如“完全透明背景”“无背景”“alpha 通道抠图”“PNG 透明贴纸风格”，英文的 `isolated on a fully transparent background`、`alpha matte`、`die-cut sticker` 同样有效。
* 避免描述环境和环境光（例如“水下”“室内”“光影透澈”），否则模型会把画布填满。把这类氛围改写为主体自身的属性，并补充“除主体外没有任何背景元素”。
* 保存时必须使用支持透明通道的格式，例如 `image.save("image.png")`；`image.convert("RGB")` 或保存为 jpg 都会丢掉 alpha。

发丝、飘带、水体光效这类主体容易产生大范围的半透明过渡（alpha 取中间值），若需要更硬的边缘，可以在提示词中加入“边缘干净利落”“clear silhouette”一类的描述。

## 模型训练

Qwen-Image-2.1 系列模型统一通过 [`examples/qwen_image_21/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/train.py) 进行训练，脚本的参数包括：

* 通用训练参数
    * 数据集基础配置
        * `--dataset_base_path`: 数据集的根目录。
        * `--dataset_metadata_path`: 数据集的元数据文件路径。
        * `--dataset_repeat`: 每个 epoch 中数据集重复的次数。
        * `--dataset_num_workers`: 每个 Dataloder 的进程数量。
        * `--data_file_keys`: 元数据中需要加载的字段名称，通常是图像或视频文件的路径，以 `,` 分隔。
    * 模型加载配置
        * `--model_paths`: 要加载的模型路径。JSON 格式。
        * `--model_id_with_origin_paths`: 带原始路径的模型 ID，例如 `"Qwen/Qwen-Image-2.1:transformer/diffusion_pytorch_model*.safetensors"`。用逗号分隔。
        * `--extra_inputs`: 模型 Pipeline 所需的额外输入参数，例如训练图像编辑时需要额外参数 `edit_image`，以 `,` 分隔。
        * `--fp8_models`：以 FP8 格式加载的模型，格式与 `--model_paths` 或 `--model_id_with_origin_paths` 一致，目前仅支持参数不被梯度更新的模型（不需要梯度回传，或梯度仅更新其 LoRA）。
        * `--quant_options`：对加载的模型进行动态量化。以 `;` 分隔多个条目，每个为 `<模型字符串>:<method>[/<exclude_modules>]`，`<模型字符串>` 需与 `--model_paths`/`--model_id_with_origin_paths` 中的一致，`method` 为已注册的量化方法（如 `bitsandbytes_nf4`），`exclude_modules` 为可选的保持全精度的层。
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
    * 图像宽高配置（适用于图像生成模型和视频生成模型）
        * `--height`: 图像或视频的高度。将 `height` 和 `width` 留空以启用动态分辨率。
        * `--width`: 图像或视频的宽度。将 `height` 和 `width` 留空以启用动态分辨率。
        * `--max_pixels`: 图像或视频帧的最大像素面积，当启用动态分辨率时，分辨率大于这个数值的图片都会被缩小，分辨率小于这个数值的图片保持不变。
* Qwen-Image-2.1 专有参数
    * `--processor_path`: processor 的路径，留空则自动从远程下载。
    * `--initialize_model_on_cpu`: 是否在 CPU 上初始化模型，用于降低多卡训练启动阶段的显存峰值。

训练图像默认以 RGBA 加载，因此带透明通道的数据集可以直接用于训练透明生成。训练图像编辑时，需要在 `--data_file_keys` 中加入 `edit_image`，并通过 `--extra_inputs "edit_image"` 注入 Pipeline，完整配置参考 LoRA 训练脚本中注释掉的 Edit 段落。

我们构建了一个样例图像数据集，以方便您进行测试，通过以下命令可以下载这个数据集：

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

我们为每个模型编写了推荐的训练脚本，请参考前文“模型总览”中的表格。关于如何编写模型训练脚本，请参考[模型训练](../Pipeline_Usage/Model_Training.md)；更多高阶训练算法，请参考[训练框架详解](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/zh/Training/)。
