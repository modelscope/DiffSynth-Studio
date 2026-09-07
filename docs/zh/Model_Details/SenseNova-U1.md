# SenseNova-U1

SenseNova-U1 是由商汤科技开源的统一多模态模型系列，采用 Mixture-of-Transformers (MoT) 架构，在同一组 Transformer 层中并行维护理解分支与生成分支，直接在像素空间进行 flow matching 去噪，无需独立的 VAE 与文本编码器。

## 安装

在使用本项目进行模型推理和训练前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多关于安装的信息，请参考[安装依赖](../Pipeline_Usage/Setup.md)。

## 快速开始

运行以下代码可以快速加载 [SenseNova/SenseNova-U1.5-8B-MoT](https://www.modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT) 模型并进行推理。显存管理已启动，框架会自动根据剩余显存控制模型参数的加载，最低 4G 显存即可运行。

```python
from diffsynth.pipelines.sensenova_u1_image import SenseNovaU1ImagePipeline, ModelConfig
import torch

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

pipe = SenseNovaU1ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT", origin_file_pattern="model*.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT", origin_file_pattern="./"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

prompt = "A neon bar sign that clearly reads \"OPEN LATE\", dark interior, moody reflections, easy text rendering. Any text in the image must be rendered exactly as written in quotation marks, with correct spelling, clean typography, and strong readability."
image = pipe(prompt=prompt, seed=42, height=2048, width=2048, num_inference_steps=50, cfg_scale=4.0, shift=3.0)
image.save("image_SenseNova-U1.5-8B-MoT.jpg")
```

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
|[SenseNova/SenseNova-U1.5-8B-MoT: T2I](https://www.modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_inference/SenseNova-U1.5-8B-MoT.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_inference_low_vram/SenseNova-U1.5-8B-MoT.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/full/SenseNova-U1.5-8B-MoT.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/validate_full/SenseNova-U1.5-8B-MoT.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/lora/SenseNova-U1.5-8B-MoT.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/validate_lora/SenseNova-U1.5-8B-MoT.py)|
|[SenseNova/SenseNova-U1.5-8B-MoT: Edit](https://www.modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_inference/SenseNova-U1.5-8B-MoT-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_inference_low_vram/SenseNova-U1.5-8B-MoT-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/full/SenseNova-U1.5-8B-MoT-Edit.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/validate_full/SenseNova-U1.5-8B-MoT-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/lora/SenseNova-U1.5-8B-MoT-Edit.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/validate_lora/SenseNova-U1.5-8B-MoT-Edit.py)|
|[SenseNova/SenseNova-U1.5-8B-MoT-LoRAs: 8-step](https://www.modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT-LoRAs)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_inference/SenseNova-U1.5-8B-MoT-LoRA-8step.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_inference_low_vram/SenseNova-U1.5-8B-MoT-LoRA-8step.py)|-|-|-|-|
|[SenseNova/SenseNova-U1.5-8B-MoT-SFT: T2I](https://www.modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT-SFT)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_inference/SenseNova-U1.5-8B-MoT-SFT.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_inference_low_vram/SenseNova-U1.5-8B-MoT-SFT.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/full/SenseNova-U1.5-8B-MoT-SFT.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/validate_full/SenseNova-U1.5-8B-MoT-SFT.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/lora/SenseNova-U1.5-8B-MoT-SFT.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/validate_lora/SenseNova-U1.5-8B-MoT-SFT.py)|
|[SenseNova/SenseNova-U1.5-8B-MoT-SFT: Edit](https://www.modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT-SFT)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_inference/SenseNova-U1.5-8B-MoT-SFT-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_inference_low_vram/SenseNova-U1.5-8B-MoT-SFT-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/full/SenseNova-U1.5-8B-MoT-SFT-Edit.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/validate_full/SenseNova-U1.5-8B-MoT-SFT-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/lora/SenseNova-U1.5-8B-MoT-SFT-Edit.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/sensenova_u1/model_training/validate_lora/SenseNova-U1.5-8B-MoT-SFT-Edit.py)|

## 模型推理

模型通过 `SenseNovaU1ImagePipeline.from_pretrained` 加载，详见[加载模型](../Pipeline_Usage/Model_Inference.md#加载模型)。

`SenseNovaU1ImagePipeline` 推理的输入参数包括：

* `prompt`: 文本提示词。图像编辑时为编辑指令。
* `cfg_scale`: Classifier-Free Guidance 缩放系数，默认为 4.0。官方实现的无条件分支文本是固定的，不提供负向提示词，因此本 Pipeline 也没有 `negative_prompt` 参数。
* `height`: 输出图像高度，默认为 2048。必须为 32 的倍数。
* `width`: 输出图像宽度，默认为 2048。必须为 32 的倍数。
* `seed`: 随机种子，默认为随机。
* `rand_device`: 噪声生成设备，默认为 `"cuda"`。
* `num_inference_steps`: 推理步数，默认为 50。
* `shift`: 时间步偏移量，影响 sigma 计算，默认为 3.0。
* `think_mode`: 是否让模型先输出一段推理再生成图像，默认为 False。
* `edit_image`: 输入图像，可以是单张 `Image.Image` 或图像列表。传入后切换到图像编辑模式，默认为 None（文生图模式）。
* `input_image`: 训练时提供的目标图像，推理时无需设置。

> **显存提示**: SenseNova-U1.5-8B-MoT 参数量约 17.5B，BF16 权重常驻显存需约 35G。生成 2048x2048 图像时建议开启显存管理（vram_config），或使用低显存推理脚本，详见[显存管理](../Pipeline_Usage/Model_Inference.md)。

### 图像编辑

传入 `edit_image` 即切换到图像编辑模式。输入图像会经过理解分支的视觉编码器编码后拼接进条件前缀，负向分支携带输入图像但不带编辑指令，因此引导方向是"远离原图不变"而非"远离任意图像"。

输出尺寸需显式指定，不会从输入图像推导：

```python
from PIL import Image

edit_image = Image.open("input.jpg").convert("RGB")
image = pipe(prompt="Change the dress to pink.", edit_image=edit_image, height=2048, width=2048, seed=42)
```

传入图像列表即可进行多图编辑。图像按传入顺序编号，prompt 中可用 Figure 1、Figure 2 指代：

```python
image = pipe(
    prompt="Change the color of the dress in Figure 1 to the color shown in Figure 2.",
    edit_image=[edit_image, color_image],
    height=2048, width=2048, seed=42,
)
```

### 推理模式（Think Mode）

传入 `think_mode=True` 时，模型会先自回归写出一段推理（构图、环境、光影的规划），
再据此生成图像。这段推理只影响图像内容，不作为返回值：

```python
image = pipe(prompt="A neon bar sign that clearly reads \"OPEN LATE\"", think_mode=True, seed=42)
image.save("image.jpg")
```

解码是贪心的（无 temperature / top-p），最多 1024 个 token，遇到 `</think>` 或 `<|im_end|>` 停止。
只有正向分支参与推理，负向分支不变。

> 每生成一个 token 都要过一次 `lm_head`（约 0.6B 参数）。与 Disk Offload 叠加时这会成为瓶颈，
> 因此低显存推理脚本不提供该模式。

### 快速推理（8-step LoRA）

官方发布的蒸馏 LoRA 可以把去噪步数从 50 降到 8，并把 `cfg_scale` 设为 1.0。后者会让框架跳过负向分支，
每步少一次 17.5B 前向，因此整体约 12 倍加速：

```python
pipe.load_lora(pipe.dit, ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT-LoRAs", origin_file_pattern="SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors"))
image = pipe(prompt=prompt, seed=42, height=2048, width=2048, num_inference_steps=8, cfg_scale=1.0, shift=3.0)
```

该 LoRA 作用在生成分支的 attention 与 MLP 上（42 层 × 7 个模块，共 294 处），基座权重无需替换。

### SFT 版本

`SenseNova-U1.5-8B-MoT-SFT` 与上述模型架构完全一致，区别在训练阶段：它是 Unified SFT 之后的检查点，
而正式版在此基础上又经过 Multi-Expert RL 与 MOPD 训练。追求生成质量时使用正式版；作为微调起点或做消融
实验时可以选择 SFT 版，用法只需替换 `model_id`。

### 架构说明

SenseNova-U1 与常见的扩散模型在结构上有三点差异，使用时值得留意：

* **MoT 双分支共享层**：42 层 decoder 中每一层都同时持有理解分支与生成分支两套权重，按 token 选择走哪条分支。两个分支无法拆成独立模型，因此 DiffSynth 中只注册了一个 `sensenova_u1_dit` 组件。
* **无 VAE 的像素空间去噪**：`latents` 全程是 `(1, 3, H, W)` 的像素张量，最终输出不经过 VAE 解码，由像素头（PixelShuffle 卷积解码器）直接还原分辨率。
* **前缀 KV cache 做条件注入**：没有独立的文本编码器。提示词先经理解分支编码为 `past_key_values`，随后每个去噪步让图像 token 走生成分支对该缓存做交叉注意。

## 模型训练

sensenova_u1 系列模型统一通过 `examples/sensenova_u1/model_training/train.py` 进行训练，脚本的参数包括：

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
        * `--height`: 图像/视频的高度。留空启用动态分辨率。
        * `--width`: 图像/视频的宽度。留空启用动态分辨率。
        * `--max_pixels`: 最大像素面积，动态分辨率时大于此值的图片会被缩小。
        * `--num_frames`: 视频的帧数（仅视频生成模型）。
* SenseNova-U1 专有参数
    * `--tokenizer_config`: Tokenizer 配置文件路径，用于加载 Qwen2Tokenizer 进行文本 tokenization。
    * `--initialize_model_on_cpu`: 是否在 CPU 上初始化模型，启用后可降低 GPU 显存峰值。

LoRA 训练建议只作用于生成分支，即 `--lora_target_modules` 写全 `_mot_gen` 后缀：`q_proj_mot_gen,k_proj_mot_gen,v_proj_mot_gen,o_proj_mot_gen,mlp_mot_gen.gate_proj,mlp_mot_gen.up_proj,mlp_mot_gen.down_proj`。peft 按后缀匹配，若只写 `gate_proj` 会同时命中理解分支的 `mlp.gate_proj`。

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

我们为每个模型编写了推荐的训练脚本，请参考前文"模型总览"中的表格。关于如何编写模型训练脚本，请参考[模型训练](../Pipeline_Usage/Model_Training.md)；更多高阶训练算法，请参考[训练框架详解](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/zh/Training/)。
