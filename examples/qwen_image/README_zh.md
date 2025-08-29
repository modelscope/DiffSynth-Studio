# Qwen-Image

[Switch to English](./README.md)

Qwen-Image 是由阿里巴巴通义实验室开源的图像生成模型。

## 安装

在使用本系列模型之前，请通过源码安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

## 快速开始

通过运行以下代码可以快速加载 [Qwen/Qwen-Image](https://www.modelscope.cn/models/Qwen/Qwen-Image) 模型并进行推理

```python
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from PIL import Image
import torch

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
image = pipe(
    prompt, seed=0, num_inference_steps=40,
    # edit_image=Image.open("xxx.jpg").resize((1328, 1328)) # For Qwen-Image-Edit
)
image.save("image.jpg")
```

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
|[Qwen/Qwen-Image](https://www.modelscope.cn/models/Qwen/Qwen-Image)|[code](./model_inference/Qwen-Image.py)|[code](./model_inference_low_vram/Qwen-Image.py)|[code](./model_training/full/Qwen-Image.sh)|[code](./model_training/validate_full/Qwen-Image.py)|[code](./model_training/lora/Qwen-Image.sh)|[code](./model_training/validate_lora/Qwen-Image.py)|
|[Qwen/Qwen-Image-Edit](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit)|[code](./model_inference/Qwen-Image-Edit.py)|[code](./model_inference_low_vram/Qwen-Image-Edit.py)|[code](./model_training/full/Qwen-Image-Edit.sh)|[code](./model_training/validate_full/Qwen-Image-Edit.py)|[code](./model_training/lora/Qwen-Image-Edit.sh)|[code](./model_training/validate_lora/Qwen-Image-Edit.py)|
|[DiffSynth-Studio/Qwen-Image-Distill-Full](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-Full)|[code](./model_inference/Qwen-Image-Distill-Full.py)|[code](./model_inference_low_vram/Qwen-Image-Distill-Full.py)|[code](./model_training/full/Qwen-Image-Distill-Full.sh)|[code](./model_training/validate_full/Qwen-Image-Distill-Full.py)|[code](./model_training/lora/Qwen-Image-Distill-Full.sh)|[code](./model_training/validate_lora/Qwen-Image-Distill-Full.py)|
|[DiffSynth-Studio/Qwen-Image-Distill-LoRA](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-LoRA)|[code](./model_inference/Qwen-Image-Distill-LoRA.py)|[code](./model_inference_low_vram/Qwen-Image-Distill-LoRA.py)|-|-|-|-|
|[DiffSynth-Studio/Qwen-Image-EliGen](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-EliGen)|[code](./model_inference/Qwen-Image-EliGen.py)|[code](./model_inference_low_vram/Qwen-Image-EliGen.py)|-|-|[code](./model_training/lora/Qwen-Image-EliGen.sh)|[code](./model_training/validate_lora/Qwen-Image-EliGen.py)|
|[DiffSynth-Studio/Qwen-Image-EliGen-V2](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-EliGen-V2)|[code](./model_inference/Qwen-Image-EliGen-V2.py)|[code](./model_inference_low_vram/Qwen-Image-EliGen-V2.py)|-|-|[code](./model_training/lora/Qwen-Image-EliGen.sh)|[code](./model_training/validate_lora/Qwen-Image-EliGen.py)|
|[DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny)|[code](./model_inference/Qwen-Image-Blockwise-ControlNet-Canny.py)|[code](./model_inference_low_vram/Qwen-Image-Blockwise-ControlNet-Canny.py)|[code](./model_training/full/Qwen-Image-Blockwise-ControlNet-Canny.sh)|[code](./model_training/validate_full/Qwen-Image-Blockwise-ControlNet-Canny.py)|[code](./model_training/lora/Qwen-Image-Blockwise-ControlNet-Canny.sh)|[code](./model_training/validate_lora/Qwen-Image-Blockwise-ControlNet-Canny.py)|
|[DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth)|[code](./model_inference/Qwen-Image-Blockwise-ControlNet-Depth.py)|[code](./model_inference_low_vram/Qwen-Image-Blockwise-ControlNet-Depth.py)|[code](./model_training/full/Qwen-Image-Blockwise-ControlNet-Depth.sh)|[code](./model_training/validate_full/Qwen-Image-Blockwise-ControlNet-Depth.py)|[code](./model_training/lora/Qwen-Image-Blockwise-ControlNet-Depth.sh)|[code](./model_training/validate_lora/Qwen-Image-Blockwise-ControlNet-Depth.py)|
|[DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint)|[code](./model_inference/Qwen-Image-Blockwise-ControlNet-Inpaint.py)|[code](./model_inference_low_vram/Qwen-Image-Blockwise-ControlNet-Inpaint.py)|[code](./model_training/full/Qwen-Image-Blockwise-ControlNet-Inpaint.sh)|[code](./model_training/validate_full/Qwen-Image-Blockwise-ControlNet-Inpaint.py)|[code](./model_training/lora/Qwen-Image-Blockwise-ControlNet-Inpaint.sh)|[code](./model_training/validate_lora/Qwen-Image-Blockwise-ControlNet-Inpaint.py)|
|[DiffSynth-Studio/Qwen-Image-In-Context-Control-Union](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-In-Context-Control-Union)|[code](./model_inference/Qwen-Image-In-Context-Control-Union.py)|[code](./model_inference_low_vram/Qwen-Image-In-Context-Control-Union.py)|-|-|[code](./model_training/lora/Qwen-Image-In-Context-Control-Union.sh)|[code](./model_training/validate_lora/Qwen-Image-In-Context-Control-Union.py)|
|[DiffSynth-Studio/Qwen-Image-Edit-Lowres-Fix](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Edit-Lowres-Fix)|[code](./model_inference/Qwen-Image-Edit-Lowres-Fix.py)|[code](./model_inference_low_vram/Qwen-Image-Edit-Lowres-Fix.py)|-|-|-|-|

## 模型推理

以下部分将会帮助您理解我们的功能并编写推理代码。

<details>

<summary>加载模型</summary>

模型通过 `from_pretrained` 加载：

```python
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
```

其中 `torch_dtype` 和 `device` 是计算精度和计算设备。`model_configs` 可通过多种方式配置模型路径：

* 从[魔搭社区](https://modelscope.cn/)下载模型并加载。此时需要填写 `model_id` 和 `origin_file_pattern`，例如

```python
ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
```

* 从本地文件路径加载模型。此时需要填写 `path`，例如

```python
ModelConfig(path="models/xxx.safetensors")
```

对于从多个文件加载的单一模型，使用列表即可，例如

```python
ModelConfig(path=[
    "models/Qwen/Qwen-Image/text_encoder/model-00001-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00002-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00003-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00004-of-00004.safetensors",
])
```

`ModelConfig` 提供了额外的参数用于控制模型加载时的行为：

* `local_model_path`: 用于保存下载模型的路径，默认值为 `"./models"`。
* `skip_download`: 是否跳过下载，默认值为 `False`。当您的网络无法访问[魔搭社区](https://modelscope.cn/)时，请手动下载必要的文件，并将其设置为 `True`。

</details>


<details>

<summary>显存管理</summary>

DiffSynth-Studio 为 Qwen-Image 模型提供了细粒度的显存管理，让模型能够在低显存设备上进行推理，可通过以下代码开启 offload 功能，在显存有限的设备上将部分模块 offload 到内存中。

```python
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", offload_device="cpu"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
pipe.enable_vram_management()
```

FP8 量化功能也是支持的：

```python
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", offload_dtype=torch.float8_e4m3fn),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
pipe.enable_vram_management()
```

FP8 量化和 offload 可同时开启：

```python
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
pipe.enable_vram_management()
```

FP8 量化能够大幅度减少显存占用，但不会加速，部分模型在 FP8 量化下会出现精度不足导致的画面模糊、撕裂、失真问题，请谨慎使用 FP8 量化。

开启显存管理后，框架会自动根据设备上的剩余显存确定显存管理策略。`enable_vram_management` 函数提供了以下参数，用于手动控制显存管理策略：

* `vram_limit`: 显存占用量限制（GB），默认占用设备上的剩余显存。注意这不是一个绝对限制，当设置的显存不足以支持模型进行推理，但实际可用显存足够时，将会以最小化显存占用的形式进行推理。将其设置为0时，将会实现理论最小显存占用。
* `vram_buffer`: 显存缓冲区大小（GB），默认为 0.5GB。由于部分较大的神经网络层在 onload 阶段会不可控地占用更多显存，因此一个显存缓冲区是必要的，理论上的最优值为模型中最大的层所占的显存。
* `num_persistent_param_in_dit`: DiT 模型中常驻显存的参数数量（个），默认为无限制。我们将会在未来删除这个参数，请不要依赖这个参数。
* `enable_dit_fp8_computation`: 是否启用 DiT 模型中的 FP8 计算，仅适用于支持 FP8 运算的 GPU（例如 H200 等），默认不启用。

</details>


<details>

<summary>推理加速</summary>

* FP8 量化：根据您的硬件与需求，请选择合适的量化方式
    * GPU 不支持 FP8 计算（例如 A100、4090 等）：FP8 量化仅能降低显存占用，无法加速，代码：[./model_inference_low_vram/Qwen-Image.py](./model_inference_low_vram/Qwen-Image.py)
    * GPU 支持 FP8 运算（例如 H200 等）：请安装 [Flash Attention 3](https://github.com/Dao-AILab/flash-attention)，否则 FP8 加速仅对 Linear 层生效
        * 更快的速度，但更大的显存：请使用 [./accelerate/Qwen-Image-FP8.py](./accelerate/Qwen-Image-FP8.py)
        * 稍慢的速度，但更小的显存：请使用 [./accelerate/Qwen-Image-FP8-offload.py](./accelerate/Qwen-Image-FP8-offload.py)
* 蒸馏加速：我们训练了两个蒸馏加速模型，可以在 `cfg_scale=1` 和 `num_inference_steps=15` 设置下进行快速推理
    * [DiffSynth-Studio/Qwen-Image-Distill-Full](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-Full)：全量蒸馏训练版本，更好的生成效果，稍差的 LoRA 兼容性，请使用 [./model_inference/Qwen-Image-Distill-Full.py](./model_inference/Qwen-Image-Distill-Full.py)
    * [DiffSynth-Studio/Qwen-Image-Distill-LoRA](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-LoRA)：LoRA 蒸馏训练版本，稍差的生成效果，更好的 LoRA 兼容性，请使用 [./model_inference/Qwen-Image-Distill-LoRA.py](./model_inference/Qwen-Image-Distill-LoRA.py)

</details>


<details>

<summary>输入参数</summary>

Pipeline 在推理阶段能够接收以下输入参数：

* `prompt`: 提示词，描述画面中出现的内容。
* `negative_prompt`: 负向提示词，描述画面中不应该出现的内容，默认值为 `""`。
* `cfg_scale`: Classifier-free guidance 的参数，默认值为 1，当设置为大于1的数值时生效。
* `input_image`: 输入图像，用于图生图，该参数与 `denoising_strength` 配合使用。
* `denoising_strength`: 去噪强度，范围是 0～1，默认值为 1，当数值接近 0 时，生成图像与输入图像相似；当数值接近 1 时，生成图像与输入图像相差更大。在不输入 `input_image` 参数时，请不要将其设置为非 1 的数值。
* `height`: 图像高度，需保证高度为 16 的倍数。
* `width`: 图像宽度，需保证宽度为 16 的倍数。
* `seed`: 随机种子。默认为 `None`，即完全随机。
* `rand_device`: 生成随机高斯噪声矩阵的计算设备，默认为 `"cpu"`。当设置为 `cuda` 时，在不同 GPU 上会导致不同的生成结果。
* `num_inference_steps`: 推理次数，默认值为 30。
* `tiled`: 是否启用 VAE 分块推理，默认为 `False`。设置为 `True` 时可显著减少 VAE 编解码阶段的显存占用，会产生少许误差，以及少量推理时间延长。
* `tile_size`: VAE 编解码阶段的分块大小，默认为 128，仅在 `tiled=True` 时生效。
* `tile_stride`: VAE 编解码阶段的分块步长，默认为 64，仅在 `tiled=True` 时生效，需保证其数值小于或等于 `tile_size`。
* `progress_bar_cmd`: 进度条，默认为 `tqdm.tqdm`。可通过设置为 `lambda x:x` 来屏蔽进度条。

</details>


## 模型训练

Qwen-Image 系列模型训练通过统一的 [`./model_training/train.py`](./model_training/train.py) 脚本进行。

<details>

<summary>脚本参数</summary>

脚本包含以下参数：

* 数据集
  * `--dataset_base_path`: 数据集的根路径。
  * `--dataset_metadata_path`: 数据集的元数据文件路径。
  * `--max_pixels`: 最大像素面积，默认为 1024*1024，当启用动态分辨率时，任何分辨率大于这个数值的图片都会被缩小。
  * `--height`: 图像或视频的高度。将 `height` 和 `width` 留空以启用动态分辨率。
  * `--width`: 图像或视频的宽度。将 `height` 和 `width` 留空以启用动态分辨率。
  * `--data_file_keys`: 元数据中的数据文件键。用逗号分隔。
  * `--dataset_repeat`: 每个 epoch 中数据集重复的次数。
  * `--dataset_num_workers`: 每个 Dataloder 的进程数量。
* 模型
  * `--model_paths`: 要加载的模型路径。JSON 格式。
  * `--model_id_with_origin_paths`: 带原始路径的模型 ID，例如 Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors。用逗号分隔。
  * `--tokenizer_path`: tokenizer 路径，留空将会自动下载。
  * `--processor_path`：Qwen-Image-Edit 的 processor 路径。留空则自动下载。
* 训练
  * `--learning_rate`: 学习率。
  * `--weight_decay`：权重衰减大小。
  * `--num_epochs`: 轮数（Epoch）。
  * `--output_path`: 保存路径。
  * `--remove_prefix_in_ckpt`: 在 ckpt 中移除前缀。
  * `--save_steps`: 保存模型的间隔 step 数量，如果设置为 None ，则每个 epoch 保存一次
  * `--find_unused_parameters`: DDP 训练中是否存在未使用的参数
* 可训练模块
  * `--trainable_models`: 可训练的模型，例如 dit、vae、text_encoder。
  * `--lora_base_model`: LoRA 添加到哪个模型上。
  * `--lora_target_modules`: LoRA 添加到哪一层上。
  * `--lora_rank`: LoRA 的秩（Rank）。
  * `--lora_checkpoint`: LoRA 检查点的路径。如果提供此路径，LoRA 将从此检查点加载。
* 额外模型输入
  * `--extra_inputs`: 额外的模型输入，以逗号分隔。
* 显存管理
  * `--use_gradient_checkpointing`: 是否启用 gradient checkpointing。
  * `--use_gradient_checkpointing_offload`: 是否将 gradient checkpointing 卸载到内存中。
  * `--gradient_accumulation_steps`: 梯度累积步数。

此外，训练框架基于 [`accelerate`](https://huggingface.co/docs/accelerate/index) 构建，在开始训练前运行 `accelerate config` 可配置 GPU 的相关参数。对于部分模型训练（例如 20B 模型的全量训练）脚本，我们提供了建议的 `accelerate` 配置文件，可在对应的训练脚本中查看。

</details>


<details>

<summary>Step 1: 准备数据集</summary>

数据集包含一系列文件，我们建议您这样组织数据集文件：

```
data/example_image_dataset/
├── metadata.csv
├── image1.jpg
└── image2.jpg
```

其中 `image1.jpg`、`image2.jpg` 为训练用图像数据，`metadata.csv` 为元数据列表，例如

```
image,prompt
image1.jpg,"a cat is sleeping"
image2.jpg,"a dog is running"
```

我们构建了一个样例图像数据集，以方便您进行测试，通过以下命令可以下载这个数据集：

```shell
modelscope download --dataset DiffSynth-Studio/example_image_dataset --local_dir ./data/example_image_dataset
```

数据集支持多种图片格式，`"jpg", "jpeg", "png", "webp"`。

图片的尺寸可通过脚本参数 `--height`、`--width` 控制。当 `--height` 和 `--width` 为空时将会开启动态分辨率，按照数据集中每个图像的实际宽高训练。

**我们强烈建议使用固定分辨率训练，因为在多卡训练中存在负载均衡问题。**

</details>


<details>

<summary>Step 2: 加载模型</summary>

类似于推理时的模型加载逻辑，可直接通过模型 ID 配置要加载的模型。例如，推理时我们通过以下设置加载模型

```python
model_configs=[
    ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
    ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
    ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
]
```

那么在训练时，填入以下参数即可加载对应的模型。

```shell
--model_id_with_origin_paths "Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors"
```

如果您希望从本地文件加载模型，例如推理时

```python
model_configs=[
    ModelConfig([
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00009-of-00009.safetensors"
    ]),
    ModelConfig([
        "models/Qwen/Qwen-Image/text_encoder/model-00001-of-00004.safetensors",
        "models/Qwen/Qwen-Image/text_encoder/model-00002-of-00004.safetensors",
        "models/Qwen/Qwen-Image/text_encoder/model-00003-of-00004.safetensors",
        "models/Qwen/Qwen-Image/text_encoder/model-00004-of-00004.safetensors"
    ]),
    ModelConfig("models/Qwen/Qwen-Image/vae/diffusion_pytorch_model.safetensors")
]
```

那么训练时需设置为

```shell
--model_paths '[
    [
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
        "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model-00009-of-00009.safetensors"
    ],
    [
        "models/Qwen/Qwen-Image/text_encoder/model-00001-of-00004.safetensors",
        "models/Qwen/Qwen-Image/text_encoder/model-00002-of-00004.safetensors",
        "models/Qwen/Qwen-Image/text_encoder/model-00003-of-00004.safetensors",
        "models/Qwen/Qwen-Image/text_encoder/model-00004-of-00004.safetensors"
    ],
    "models/Qwen/Qwen-Image/vae/diffusion_pytorch_model.safetensors"
]' \
```

</details>


<details>

<summary>Step 3: 设置可训练模块</summary>

训练框架支持训练基础模型，或 LoRA 模型。以下是几个例子：

* 全量训练 DiT 部分：`--trainable_models dit`
* 训练 DiT 部分的 LoRA 模型：`--lora_base_model dit --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" --lora_rank 32`

此外，由于训练脚本中加载了多个模块（text encoder、dit、vae），保存模型文件时需要移除前缀，例如在全量训练 DiT 部分或者训练 DiT 部分的 LoRA 模型时，请设置 `--remove_prefix_in_ckpt pipe.dit.`

</details>


<details>

<summary>Step 4: 启动训练程序</summary>

我们为每一个模型编写了训练命令，请参考本文档开头的表格。

</details>
