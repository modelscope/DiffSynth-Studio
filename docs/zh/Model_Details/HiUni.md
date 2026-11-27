# HiUni

HiUni 是由 ERNIE Research 开源的统一多模态模型，同一个 checkpoint 即可完成图像理解、文生图与指令图像编辑。它是一个小模型，在视觉语言骨干网络上增加了一个轻量的像素空间扩散头，直接以像素生成图像，无需 VAE 与独立的文本编码器。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。

## 快速开始

运行以下代码可以快速加载 [ernie-research/HiUni](https://modelscope.cn/models/ernie-research/HiUni) 模型并进行推理。显存管理已启动，框架会自动根据剩余显存控制模型参数的加载，最低 12G 显存即可运行。

```python
from diffsynth.pipelines.hiuni import HiUniPipeline, ModelConfig
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

pipe = HiUniPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="ernie-research/HiUni", origin_file_pattern="model-*.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="ernie-research/HiUni", origin_file_pattern="tokenizer*.json"),
    processor_config=ModelConfig(model_id="ernie-research/HiUni", origin_file_pattern="preprocessor_config.json"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

prompt = "A cozy reading nook by a rain-streaked window, a ginger cat asleep on a knitted blanket, warm lamp light, film photography."
image = pipe(prompt, height=2048, width=2048, seed=42, num_inference_steps=25, cfg_scale=4.0, shift=4.0)
image.save("image_HiUni.jpg")
```

## 模型总览

|模型 ID|推理|低显存推理|并行推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|-|
|[ernie-research/HiUni](https://modelscope.cn/models/ernie-research/HiUni)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hiuni/model_inference/HiUni.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hiuni/model_inference_low_vram/HiUni.py)|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hiuni/model_training/full/HiUni.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hiuni/model_training/validate_full/HiUni.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hiuni/model_training/lora/HiUni.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hiuni/model_training/validate_lora/HiUni.py)|

## 模型推理

模型通过 `HiUniPipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。`tokenizer_config` 与 `processor_config` 从仓库根目录加载分词器与图像处理器，只会下载对应的 JSON 文件。

`HiUniPipeline` 推理的输入参数包括：

* `prompt`: 文本提示词。图像编辑时为编辑指令。
* `cfg_scale`: 文本引导系数，默认为 4.0。
* `height`: 输出图像高度，默认为 None，即文生图使用 1024、图像编辑沿用参考图分辨率。必须为 32 的倍数。
* `width`: 输出图像宽度，默认为 None，规则同上。必须为 32 的倍数。
* `seed`: 随机种子，默认为 42。
* `num_inference_steps`: 推理步数，默认为 25。
* `shift`: 时间步偏移量，影响 sigma 计算，默认为 4.0。
* `edit_image`: 参考图像。传入后切换到图像编辑模式，默认为 None（文生图模式）。
* `image_cfg_scale`: 编辑时的图像引导系数，默认为 1.0。不等于 1.0 时会增加第三个无条件分支。
* `resize_to_edit_image`: 是否把编辑结果缩放回 `edit_image` 的尺寸，默认为 True。
* `min_pixels`、`max_pixels`: 编辑模式下参考图的像素预算，默认均为 262144（512x512）。

### 图像编辑

传入 `edit_image` 后切换到图像编辑模式。参考图经视觉编码器编码后放在用户轮的编辑指令之前，其像素也会在每一步输入扩散头。未指定 `height`/`width` 时，输出使用参考图预处理后的分辨率，在 `min_pixels`/`max_pixels` 预算内保持其宽高比。`resize_to_edit_image=True` 时，结果会缩放回参考图的原始尺寸。

```python
from PIL import Image

edit_image = Image.open("input.jpg").convert("RGB")
image = pipe("Change the background to a snowy mountain.", edit_image=edit_image, seed=42)
```

当 `image_cfg_scale != 1` 时，引导由三路预测组合：`full + cfg_scale * (posi - nega) + image_cfg_scale * (nega - full)`。其中 `posi` 同时看到参考图与指令，`nega` 只看到参考图，`full` 两者都看不到。默认 `image_cfg_scale=1.0` 时 `full` 分支相互抵消，不会计算。

### 图像理解与文本对话

`pipe.chat` 使用同一个 checkpoint 进行自回归文本生成。传入 `images`（单张图像或列表）进行图像理解，省略则为纯文本对话：

```python
answer = pipe.chat("Describe this image in detail.", images=edit_image)
answer = pipe.chat("Write a short poem about autumn.", think_mode="empty")
```

* `images`: 输入图像，单张图像或图像列表，默认为 None（纯文本对话）。
* `system`: 系统提示词，默认为空字符串。
* `think_mode`: `"think"`（默认）让模型先写 `<think>` 推理块再回答，`"empty"` 预填一个空推理块，`"none"` 不预填。
* `max_new_tokens`: 最大生成 token 数，默认为 1024。
* `max_length`: 整个序列的最大长度，默认为 9216。
* `temperature`、`top_k`、`top_p`、`seed`: 采样参数。默认 `temperature=0.0` 为贪心解码。
* `min_pixels`、`max_pixels`: 输入图像的像素预算，默认为 262144 与 4194304。
* `return_raw`: 返回包含推理块的完整解码文本。默认只返回最后一个 `</think>` 之后的回答。

与训练一致，解码不使用 KV cache，每生成一个 token 都会重新计算整个序列。

### 架构说明

* **共享骨干网络**：理解、生成与编辑都经过同一个语言模型。扩散头读取骨干网络在噪声图像 token 上的隐状态，因此 DiffSynth 只注册了一个 `hiuni_dit` 组件。
* **无 VAE 的像素空间去噪**：`latents` 存放 16x16 RGB 栅格 patch，形状为 `(4 * H/32 * W/32, 768)`，取值范围 `[-1, 1]`。模型预测干净图像 `x0`，`model_fn` 将其换算为 flow 速度后，引导在速度空间组合并交给调度器步进。
* **随分辨率缩放的噪声**：初始噪声乘以 `sqrt(4 * H/32 * W/32 / 256)`，与训练一致。

## 模型训练

HiUni 系列模型统一通过 `examples/hiuni/model_training/train.py` 进行训练，脚本的参数包括：

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
        * `--quant_options`：对加载的模型进行动态量化。以 `;` 分隔多个条目，每个为 `<模型字符串>:<method>[/<exclude_modules>]`，`<模型字符串>` 需与 `--model_paths`/`--model_id_with_origin_paths` 中的一致，`method` 为已注册的量化方法（如 `bitsandbytes_nf4`），`exclude_modules` 为可选的保持全精度的层。
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
        * `--height`: 图像的高度。留空启用动态分辨率。
        * `--width`: 图像的宽度。留空启用动态分辨率。
        * `--max_pixels`: 最大像素面积，动态分辨率时大于此值的图片会被缩小。
* HiUni 专有参数
    * `--tokenizer_path`: 分词器路径，留空时从 `ernie-research/HiUni` 加载。
    * `--processor_path`: 图像处理器路径，留空时从 `ernie-research/HiUni` 加载。
    * `--initialize_model_on_cpu`: 是否在 CPU 上初始化模型。

样例数据集：

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "hiuni/HiUni/*" --local_dir ./data/diffsynth_example_dataset
```

我们为每个模型编写了推荐的训练脚本，请参考前文“模型总览”中的表格。关于如何编写模型训练脚本，请参考[模型训练](../Pipeline_Usage/Model_Training.md)；更多高阶训练算法，请参考[训练框架详解](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/zh/Training/)。
