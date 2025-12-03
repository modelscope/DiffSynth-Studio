# 模型训练

本文档介绍如何使用 `DiffSynth-Studio` 进行模型训练。

## 脚本参数

训练脚本通常包含以下参数：

* 数据集基础配置
    * `--dataset_base_path`: 数据集的根目录。
    * `--dataset_metadata_path`: 数据集的元数据文件路径。
    * `--dataset_repeat`: 每个 epoch 中数据集重复的次数。
    * `--dataset_num_workers`: 每个 Dataloder 的进程数量。
    * `--data_file_keys`: 元数据中需要加载的字段名称，通常是图像或视频文件的路径，以 `,` 分隔。
* 模型加载配置
    * `--model_paths`: 要加载的模型路径。JSON 格式。
    * `--model_id_with_origin_paths`: 带原始路径的模型 ID，例如 `"Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors"`。用逗号分隔。
    * `--extra_inputs`: 模型 Pipeline 所需的额外输入参数，例如训练图像编辑模型 Qwen-Image-Edit 时需要额外参数 `edit_image`，以 `,` 分隔。
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
* 图像宽高配置（适用于图像生成模型和视频生成模型）
    * `--height`: 图像或视频的高度。将 `height` 和 `width` 留空以启用动态分辨率。
    * `--width`: 图像或视频的宽度。将 `height` 和 `width` 留空以启用动态分辨率。
    * `--max_pixels`: 图像或视频帧的最大像素面积，当启用动态分辨率时，分辨率大于这个数值的图片都会被缩小，分辨率小于这个数值的图片保持不变。

部分模型的训练脚本还包含额外的参数，详见各模型的文档。

## 准备数据集

`DiffSynth-Studio` 采用通用数据集格式，数据集包含一系列数据文件（图像、视频等），以及标注元数据的文件，我们建议您这样组织数据集文件：

```
data/example_image_dataset/
├── metadata.csv
├── image_1.jpg
└── image_2.jpg
```

其中 `image_1.jpg`、`image_2.jpg` 为训练用图像数据，`metadata.csv` 为元数据列表，例如

```
image,prompt
image_1.jpg,"a dog"
image_2.jpg,"a cat"
```

我们构建了样例数据集，以方便您进行测试。了解通用数据集架构是如何实现的，请参考 [`diffsynth.core.data`](/docs/zh/API_Reference/core/data.md)。

<details>

<summary>样例图像数据集</summary>

> ```shell
> modelscope download --dataset DiffSynth-Studio/example_image_dataset --local_dir ./data/example_image_dataset
> ```
> 
> 适用于 Qwen-Image、FLUX 等图像生成模型的训练。

</details>

<details>

<summary>样例视频数据集</summary>

> ```shell
> modelscope download --dataset DiffSynth-Studio/example_video_dataset --local_dir ./data/example_video_dataset
> ```
> 
> 适用于 Wan 等视频生成模型的训练。

</details>

## 加载模型

类似于[推理时的模型加载](/docs/zh/Pipeline_Usage/Model_Inference.md#加载模型)，我们支持多种方式配置模型路径，两种方式是可以混用的。

<details>

<summary>从远程下载模型并加载</summary>

> 如果在推理时我们通过以下设置加载模型
> 
> ```python
> model_configs=[
>     ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
>     ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
>     ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
> ]
> ```
> 
> 那么在训练时，填入以下参数即可加载对应的模型。
> 
> ```shell
> --model_id_with_origin_paths "Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors"
> ```
> 
> 模型文件默认下载到 `./models` 路径，该路径可通过[环境变量 DIFFSYNTH_MODEL_BASE_PATH](/docs/zh/Pipeline_Usage/Environment_Variables.md#diffsynth_model_base_path) 修改。
> 
> 默认情况下，即使模型已经下载完毕，程序仍会向远程查询是否有遗漏文件，如果要完全关闭远程请求，请将[环境变量 DIFFSYNTH_SKIP_DOWNLOAD](/docs/zh/Pipeline_Usage/Environment_Variables.md#diffsynth_skip_download) 设置为 `True`。

</details>

<details>

<summary>从本地文件路径加载模型</summary>

> 如果从本地文件加载模型，例如推理时
> 
> ```python
> model_configs=[
>     ModelConfig([
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00009-of-00009.safetensors"
>     ]),
>     ModelConfig([
>         "models/Qwen/Qwen-Image/text_encoder/model-00001-of-00004.safetensors",
>         "models/Qwen/Qwen-Image/text_encoder/model-00002-of-00004.safetensors",
>         "models/Qwen/Qwen-Image/text_encoder/model-00003-of-00004.safetensors",
>         "models/Qwen/Qwen-Image/text_encoder/model-00004-of-00004.safetensors"
>     ]),
>     ModelConfig("models/Qwen/Qwen-Image/vae/diffusion_pytorch_model.safetensors")
> ]
> ```
> 
> 那么训练时需设置为
> 
> ```shell
> --model_paths '[
>     [
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
>         "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00009-of-00009.safetensors"
>     ],
>     [
>         "models/Qwen/Qwen-Image/text_encoder/model-00001-of-00004.safetensors",
>         "models/Qwen/Qwen-Image/text_encoder/model-00002-of-00004.safetensors",
>         "models/Qwen/Qwen-Image/text_encoder/model-00003-of-00004.safetensors",
>         "models/Qwen/Qwen-Image/text_encoder/model-00004-of-00004.safetensors"
>     ],
>     "models/Qwen/Qwen-Image/vae/diffusion_pytorch_model.safetensors"
> ]' \
> ```
> 
> 请注意，`--model_paths` 是 json 格式，其中不能出现多余的 `,`，否则无法被正常解析。

</details>

## 设置可训练模块

训练框架支持任意模型的训练，以 Qwen-Image 为例，若全量训练其中的 DiT 模型，则需设置为

```shell
--trainable_models "dit"
```

若训练 DiT 模型的 LoRA，则需设置

```shell
--lora_base_model dit --lora_target_modules "to_q,to_k,to_v" --lora_rank 32
```

我们希望给技术探索留下足够的发挥空间，因此框架支持同时训练任意多个模块，例如同时训练 text encoder、controlnet，以及 DiT 的 LoRA：

```shell
--trainable_models "text_encoder,controlnet" --lora_base_model dit --lora_target_modules "to_q,to_k,to_v" --lora_rank 32
```

此外，由于训练脚本中加载了多个模块（text encoder、dit、vae 等），保存模型文件时需要移除前缀，例如在全量训练 DiT 部分或者训练 DiT 部分的 LoRA 模型时，请设置 `--remove_prefix_in_ckpt pipe.dit.`。如果多个模块同时训练，则需开发者在训练完成后自行编写代码拆分模型文件中的 state dict。

## 启动训练程序

训练框架基于 [`accelerate`](https://huggingface.co/docs/accelerate/index) 构建，训练命令按照如下格式编写：

```shell
accelerate launch xxx/train.py \
  --xxx yyy \
  --xxxx yyyy
```

我们为每个模型编写了预置的训练脚本，详见各模型的文档。

默认情况下，`accelerate` 会按照 `~/.cache/huggingface/accelerate/default_config.yaml` 的配置进行训练，使用 `accelerate config` 可在终端交互式地配置，包括多 GPU 训练、[`DeepSpeed`](https://www.deepspeed.ai/) 等。

我们为部分模型提供了推荐的 `accelerate` 配置文件，可通过 `--config_file` 设置，例如 Qwen-Image 模型的全量训练：

```shell
accelerate launch --config_file examples/qwen_image/model_training/full/accelerate_config_zero2offload.yaml examples/qwen_image/model_training/train.py \
  --dataset_base_path data/example_image_dataset \
  --dataset_metadata_path data/example_image_dataset/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image_full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --find_unused_parameters
```

## 训练注意事项

* 数据集的元数据除 `csv` 格式外，还支持 `json`、`jsonl` 格式，关于如何选择最佳的元数据格式，请参考[](/docs/zh/API_Reference/core/data.md#元数据)
* 通常训练效果与训练步数强相关，与 epoch 数量弱相关，因此我们更推荐使用参数 `--save_steps` 按训练步数间隔来保存模型文件。
* 当数据量 * `dataset_repeat` 超过 $10^9$ 时，我们观测到数据集的速度明显变慢，这似乎是 `PyTorch` 的 bug，我们尚不确定新版本的 `PyTorch` 是否已经修复了这一问题。
* 学习率 `--learning_rate` 在 LoRA 训练中建议设置为 `1e-4`，在全量训练中建议设置为 `1e-5`。
* 训练框架不支持 batch size > 1，原因是复杂的，详见 [Q&A: 为什么训练框架不支持 batch size > 1？](/docs/zh/QA.md#为什么训练框架不支持-batch-size--1)
* 少数模型包含冗余参数，例如 Qwen-Image 的 DiT 部分最后一层的文本编码部分，在训练这些模型时，需设置 `--find_unused_parameters` 避免在多 GPU 训练中报错。出于对开源社区模型兼容性的考虑，我们不打算删除这些冗余参数。
* Diffusion 模型的损失函数值与实际效果的关系不大，因此我们在训练过程中不会记录损失函数值。我们建议把 `--num_epochs` 设置为足够大的数值，边训边测，直至效果收敛后手动关闭训练程序。
* `--use_gradient_checkpointing` 通常是开启的，除非 GPU 显存足够；`--use_gradient_checkpointing_offload` 则按需开启，详见 [`diffsynth.core.gradient`](/docs/zh/API_Reference/core/gradient.md)。
