# FLUX

[Switch to English](./README.md)

FLUX 是由 Black-Forest-Labs 开源的一系列图像生成模型。

**DiffSynth-Studio 启用了新的推理和训练框架，如需使用旧版本，请点击[这里](https://github.com/modelscope/DiffSynth-Studio/tree/3edf3583b1f08944cee837b94d9f84d669c2729c)。**

## 安装

在使用本系列模型之前，请通过源码安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

## 快速开始

通过运行以下代码可以快速加载 [black-forest-labs/FLUX.1-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-dev) 模型并进行推理。

```python
import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig

pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
    ],
)

image = pipe(prompt="a cat", seed=0)
image.save("image.jpg")
```

## 模型总览

|模型 ID|额外参数|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|-|
|[FLUX.1-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-dev)||[code](./model_inference/FLUX.1-dev.py)|[code](./model_inference_low_vram/FLUX.1-dev.py)|[code](./model_training/full/FLUX.1-dev.sh)|[code](./model_training/validate_full/FLUX.1-dev.py)|[code](./model_training/lora/FLUX.1-dev.sh)|[code](./model_training/validate_lora/FLUX.1-dev.py)|
|[FLUX.1-Krea-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-Krea-dev)||[code](./model_inference/FLUX.1-Krea-dev.py)|[code](./model_inference_low_vram/FLUX.1-Krea-dev.py)|[code](./model_training/full/FLUX.1-Krea-dev.sh)|[code](./model_training/validate_full/FLUX.1-Krea-dev.py)|[code](./model_training/lora/FLUX.1-Krea-dev.sh)|[code](./model_training/validate_lora/FLUX.1-Krea-dev.py)|
|[FLUX.1-Kontext-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-Kontext-dev)|`kontext_images`|[code](./model_inference/FLUX.1-Kontext-dev.py)|[code](./model_inference_low_vram/FLUX.1-Kontext-dev.py)|[code](./model_training/full/FLUX.1-Kontext-dev.sh)|[code](./model_training/validate_full/FLUX.1-Kontext-dev.py)|[code](./model_training/lora/FLUX.1-Kontext-dev.sh)|[code](./model_training/validate_lora/FLUX.1-Kontext-dev.py)|
|[FLUX.1-dev-Controlnet-Inpainting-Beta](https://www.modelscope.cn/models/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta)|`controlnet_inputs`|[code](./model_inference/FLUX.1-dev-Controlnet-Inpainting-Beta.py)|[code](./model_inference_low_vram/FLUX.1-dev-Controlnet-Inpainting-Beta.py)|[code](./model_training/full/FLUX.1-dev-Controlnet-Inpainting-Beta.sh)|[code](./model_training/validate_full/FLUX.1-dev-Controlnet-Inpainting-Beta.py)|[code](./model_training/lora/FLUX.1-dev-Controlnet-Inpainting-Beta.sh)|[code](./model_training/validate_lora/FLUX.1-dev-Controlnet-Inpainting-Beta.py)|
|[FLUX.1-dev-Controlnet-Union-alpha](https://www.modelscope.cn/models/InstantX/FLUX.1-dev-Controlnet-Union-alpha)|`controlnet_inputs`|[code](./model_inference/FLUX.1-dev-Controlnet-Union-alpha.py)|[code](./model_inference_low_vram/FLUX.1-dev-Controlnet-Union-alpha.py)|[code](./model_training/full/FLUX.1-dev-Controlnet-Union-alpha.sh)|[code](./model_training/validate_full/FLUX.1-dev-Controlnet-Union-alpha.py)|[code](./model_training/lora/FLUX.1-dev-Controlnet-Union-alpha.sh)|[code](./model_training/validate_lora/FLUX.1-dev-Controlnet-Union-alpha.py)|
|[FLUX.1-dev-Controlnet-Upscaler](https://www.modelscope.cn/models/jasperai/Flux.1-dev-Controlnet-Upscaler)|`controlnet_inputs`|[code](./model_inference/FLUX.1-dev-Controlnet-Upscaler.py)|[code](./model_inference_low_vram/FLUX.1-dev-Controlnet-Upscaler.py)|[code](./model_training/full/FLUX.1-dev-Controlnet-Upscaler.sh)|[code](./model_training/validate_full/FLUX.1-dev-Controlnet-Upscaler.py)|[code](./model_training/lora/FLUX.1-dev-Controlnet-Upscaler.sh)|[code](./model_training/validate_lora/FLUX.1-dev-Controlnet-Upscaler.py)|
|[FLUX.1-dev-IP-Adapter](https://www.modelscope.cn/models/InstantX/FLUX.1-dev-IP-Adapter)|`ipadapter_images`, `ipadapter_scale`|[code](./model_inference/FLUX.1-dev-IP-Adapter.py)|[code](./model_inference_low_vram/FLUX.1-dev-IP-Adapter.py)|[code](./model_training/full/FLUX.1-dev-IP-Adapter.sh)|[code](./model_training/validate_full/FLUX.1-dev-IP-Adapter.py)|[code](./model_training/lora/FLUX.1-dev-IP-Adapter.sh)|[code](./model_training/validate_lora/FLUX.1-dev-IP-Adapter.py)|
|[FLUX.1-dev-InfiniteYou](https://www.modelscope.cn/models/ByteDance/InfiniteYou)|`infinityou_id_image`, `infinityou_guidance`, `controlnet_inputs`|[code](./model_inference/FLUX.1-dev-InfiniteYou.py)|[code](./model_inference_low_vram/FLUX.1-dev-InfiniteYou.py)|[code](./model_training/full/FLUX.1-dev-InfiniteYou.sh)|[code](./model_training/validate_full/FLUX.1-dev-InfiniteYou.py)|[code](./model_training/lora/FLUX.1-dev-InfiniteYou.sh)|[code](./model_training/validate_lora/FLUX.1-dev-InfiniteYou.py)|
|[FLUX.1-dev-EliGen](https://www.modelscope.cn/models/DiffSynth-Studio/Eligen)|`eligen_entity_prompts`, `eligen_entity_masks`, `eligen_enable_on_negative`, `eligen_enable_inpaint`|[code](./model_inference/FLUX.1-dev-EliGen.py)|[code](./model_inference_low_vram/FLUX.1-dev-EliGen.py)|-|-|[code](./model_training/lora/FLUX.1-dev-EliGen.sh)|[code](./model_training/validate_lora/FLUX.1-dev-EliGen.py)|
|[FLUX.1-dev-LoRA-Encoder](https://www.modelscope.cn/models/DiffSynth-Studio/LoRA-Encoder-FLUX.1-Dev)|`lora_encoder_inputs`, `lora_encoder_scale`|[code](./model_inference/FLUX.1-dev-LoRA-Encoder.py)|[code](./model_inference_low_vram/FLUX.1-dev-LoRA-Encoder.py)|[code](./model_training/full/FLUX.1-dev-LoRA-Encoder.sh)|[code](./model_training/validate_full/FLUX.1-dev-LoRA-Encoder.py)|-|-|
|[FLUX.1-dev-LoRA-Fusion-Preview](https://modelscope.cn/models/DiffSynth-Studio/LoRAFusion-preview-FLUX.1-dev)||[code](./model_inference/FLUX.1-dev-LoRA-Fusion.py)|-|-|-|-|-|
|[Step1X-Edit](https://www.modelscope.cn/models/stepfun-ai/Step1X-Edit)|`step1x_reference_image`|[code](./model_inference/Step1X-Edit.py)|[code](./model_inference_low_vram/Step1X-Edit.py)|[code](./model_training/full/Step1X-Edit.sh)|[code](./model_training/validate_full/Step1X-Edit.py)|[code](./model_training/lora/Step1X-Edit.sh)|[code](./model_training/validate_lora/Step1X-Edit.py)|
|[FLEX.2-preview](https://www.modelscope.cn/models/ostris/Flex.2-preview)|`flex_inpaint_image`, `flex_inpaint_mask`, `flex_control_image`, `flex_control_strength`, `flex_control_stop`|[code](./model_inference/FLEX.2-preview.py)|[code](./model_inference_low_vram/FLEX.2-preview.py)|[code](./model_training/full/FLEX.2-preview.sh)|[code](./model_training/validate_full/FLEX.2-preview.py)|[code](./model_training/lora/FLEX.2-preview.sh)|[code](./model_training/validate_lora/FLEX.2-preview.py)|
|[Nexus-Gen](https://www.modelscope.cn/models/DiffSynth-Studio/Nexus-GenV2)|`nexus_gen_reference_image`|[code](./model_inference/Nexus-Gen-Editing.py)|[code](./model_inference_low_vram/Nexus-Gen-Editing.py)|[code](./model_training/full/Nexus-Gen.sh)|[code](./model_training/validate_full/Nexus-Gen.py)|[code](./model_training/lora/Nexus-Gen.sh)|[code](./model_training/validate_lora/Nexus-Gen.py)|

## 模型推理

以下部分将会帮助您理解我们的功能并编写推理代码。

<details>

<summary>加载模型</summary>

模型通过 `from_pretrained` 加载：

```python
import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig

pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
    ],
)
```

其中 `torch_dtype` 和 `device` 是计算精度和计算设备。`model_configs` 可通过多种方式配置模型路径：

* 从[魔搭社区](https://modelscope.cn/)下载模型并加载。此时需要填写 `model_id` 和 `origin_file_pattern`，例如

```python
ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors")
```

* 从本地文件路径加载模型。此时需要填写 `path`，例如

```python
ModelConfig(path="models/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors")
```

对于从多个文件加载的单一模型，使用列表即可，例如

```python
ModelConfig(path=[
    "models/xxx/diffusion_pytorch_model-00001-of-00003.safetensors",
    "models/xxx/diffusion_pytorch_model-00002-of-00003.safetensors",
    "models/xxx/diffusion_pytorch_model-00003-of-00003.safetensors",
])
```

`ModelConfig` 还提供了额外的参数用于控制模型加载时的行为：

* `local_model_path`: 用于保存下载模型的路径，默认值为 `"./models"`。
* `skip_download`: 是否跳过下载，默认值为 `False`。当您的网络无法访问[魔搭社区](https://modelscope.cn/)时，请手动下载必要的文件，并将其设置为 `True`。

</details>


<details>

<summary>显存管理</summary>

DiffSynth-Studio 为 FLUX 模型提供了细粒度的显存管理，让模型能够在低显存设备上进行推理，可通过以下代码开启 offload 功能，在显存有限的设备上将部分模块 offload 到内存中。

```python
pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors", offload_device="cpu"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors", offload_device="cpu"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/", offload_device="cpu"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()
```

FP8 量化功能也是支持的：

```python
pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors", offload_dtype=torch.float8_e4m3fn),
    ],
)
pipe.enable_vram_management()
```

FP8 量化和 offload 可同时开启：

```python
pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
    ],
)
pipe.enable_vram_management()
```

开启显存管理后，框架会自动根据设备上的剩余显存确定显存管理策略。对于大多数 FLUX 系列模型，最低 8GB 显存即可进行推理。`enable_vram_management` 函数提供了以下参数，用于手动控制显存管理策略：

* `vram_limit`: 显存占用量限制（GB），默认占用设备上的剩余显存。注意这不是一个绝对限制，当设置的显存不足以支持模型进行推理，但实际可用显存足够时，将会以最小化显存占用的形式进行推理。将其设置为0时，将会实现理论最小显存占用。
* `vram_buffer`: 显存缓冲区大小（GB），默认为 0.5GB。由于部分较大的神经网络层在 onload 阶段会不可控地占用更多显存，因此一个显存缓冲区是必要的，理论上的最优值为模型中最大的层所占的显存。
* `num_persistent_param_in_dit`: DiT 模型中常驻显存的参数数量（个），默认为无限制。我们将会在未来删除这个参数，请不要依赖这个参数。

</details>


<details>

<summary>推理加速</summary>

* TeaCache：加速技术 [TeaCache](https://github.com/ali-vilab/TeaCache)，请参考[示例代码](./acceleration/teacache.py)。

</details>

<details>

<summary>输入参数</summary>

Pipeline 在推理阶段能够接收以下输入参数：

* `prompt`: 提示词，描述画面中出现的内容。
* `negative_prompt`: 负向提示词，描述画面中不应该出现的内容，默认值为 `""`。
* `cfg_scale`: Classifier-free guidance 的参数，默认值为 1，当设置为大于1的数值时生效。
* `embedded_guidance`: FLUX-dev 的内嵌引导参数，默认值为 3.5。
* `t5_sequence_length`: T5 模型的文本向量序列长度，默认值为 512。
* `input_image`: 输入图像，用于图生图，该参数与 `denoising_strength` 配合使用。
* `denoising_strength`: 去噪强度，范围是 0～1，默认值为 1，当数值接近 0 时，生成图像与输入图像相似；当数值接近 1 时，生成图像与输入图像相差更大。在不输入 `input_image` 参数时，请不要将其设置为非 1 的数值。
* `height`: 图像高度，需保证高度为 16 的倍数。
* `width`: 图像宽度，需保证宽度为 16 的倍数。
* `seed`: 随机种子。默认为 `None`，即完全随机。
* `rand_device`: 生成随机高斯噪声矩阵的计算设备，默认为 `"cpu"`。当设置为 `cuda` 时，在不同 GPU 上会导致不同的生成结果。
* `sigma_shift`: Rectified Flow 理论中的参数，默认为 3。数值越大，模型在去噪的开始阶段停留的步骤数越多，可适当调大这个参数来提高画面质量，但会因生成过程与训练过程不一致导致生成的图像内容与训练数据存在差异。
* `num_inference_steps`: 推理次数，默认值为 30。
* `kontext_images`: Kontext 模型的输入图像。
* `controlnet_inputs`: ControlNet 模型的输入。
* `ipadapter_images`: IP-Adapter 模型的输入图像。
* `ipadapter_scale`: IP-Adapter 模型的控制强度。
* `eligen_entity_prompts`: EliGen 模型的图像局部提示词。
* `eligen_entity_masks`: EliGen 模型的局部提示词控制区域，与 `eligen_entity_prompts` 一一对应。
* `eligen_enable_on_negative`: 是否在负向提示词一侧启用 EliGen，仅在 `cfg_scale > 1` 时生效。
* `eligen_enable_inpaint`: 是否启用 EliGen 局部重绘。
* `infinityou_id_image`: InfiniteYou 模型的人脸图像。
* `infinityou_guidance`: InfiniteYou 模型的控制强度。
* `flex_inpaint_image`: FLEX 模型用于局部重绘的图像。
* `flex_inpaint_mask`: FLEX 模型用于局部重绘的区域。
* `flex_control_image`: FLEX 模型用于结构控制的图像。
* `flex_control_strength`: FLEX 模型用于结构控制的强度。
* `flex_control_stop`: FLEX 模型结构控制的结束点，1表示全程启用，0.5表示在前半段启用，0表示不启用。
* `step1x_reference_image`: Step1x-Edit 模型用于图像编辑的输入图像。
* `lora_encoder_inputs`: LoRA 编码器的输入，格式为 ModelConfig 或本地路径。
* `lora_encoder_scale`: LoRA 编码器的激活强度，默认值为1，数值越小，LoRA 激活越弱。
* `tea_cache_l1_thresh`: TeaCache 的阈值，数值越大，速度越快，画面质量越差。请注意，开启 TeaCache 后推理速度并非均匀，因此进度条上显示的剩余时间将会变得不准确。
* `tiled`: 是否启用 VAE 分块推理，默认为 `False`。设置为 `True` 时可显著减少 VAE 编解码阶段的显存占用，会产生少许误差，以及少量推理时间延长。
* `tile_size`: VAE 编解码阶段的分块大小，默认为 128，仅在 `tiled=True` 时生效。
* `tile_stride`: VAE 编解码阶段的分块步长，默认为 64，仅在 `tiled=True` 时生效，需保证其数值小于或等于 `tile_size`。
* `progress_bar_cmd`: 进度条，默认为 `tqdm.tqdm`。可通过设置为 `lambda x:x` 来屏蔽进度条。

</details>


## 模型训练

FLUX 系列模型训练通过统一的 [`./model_training/train.py`](./model_training/train.py) 脚本进行。

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
  * `--model_id_with_origin_paths`: 带原始路径的模型 ID，例如 black-forest-labs/FLUX.1-dev:flux1-dev.safetensors。用逗号分隔。
* 训练
  * `--learning_rate`: 学习率。
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
* 额外模型输入
  * `--extra_inputs`: 额外的模型输入，以逗号分隔。
* 显存管理
  * `--use_gradient_checkpointing`: 是否启用 gradient checkpointing。
  * `--use_gradient_checkpointing_offload`: 是否将 gradient checkpointing 卸载到内存中。
  * `--gradient_accumulation_steps`: 梯度累积步数。
* 其他
  * `--align_to_opensource_format`: 是否将 FLUX DiT LoRA 的格式与开源版本对齐，仅对 LoRA 训练生效。


此外，训练框架基于 [`accelerate`](https://huggingface.co/docs/accelerate/index) 构建，在开始训练前运行 `accelerate config` 可配置 GPU 的相关参数。对于部分模型训练（例如模型的全量训练）脚本，我们提供了建议的 `accelerate` 配置文件，可在对应的训练脚本中查看。

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

当模型需要额外输入时，例如具备控制能力的模型 [`black-forest-labs/FLUX.1-Kontext-dev`](https://modelscope.cn/models/black-forest-labs/FLUX.1-Kontext-dev) 所需的 `kontext_images`，请在数据集中补充相应的列，例如：

```
image,prompt,kontext_images
image1.jpg,"a cat is sleeping",image1_reference.jpg
```

额外输入若包含图像文件，则需要在 `--data_file_keys` 参数中指定要解析的列名。可根据额外输入增加相应的列名，例如 `--data_file_keys "image,kontext_images"`，同时启用 `--extra_inputs "kontext_images"`。

</details>


<details>

<summary>Step 2: 加载模型</summary>

类似于推理时的模型加载逻辑，可直接通过模型 ID 配置要加载的模型。例如，推理时我们通过以下设置加载模型

```python
model_configs=[
    ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
    ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
    ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
    ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
]
```

那么在训练时，填入以下参数即可加载对应的模型。

```shell
--model_id_with_origin_paths "black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/,black-forest-labs/FLUX.1-dev:ae.safetensors"
```

如果您希望从本地文件加载模型，例如推理时

```python
model_configs=[
    ModelConfig(path="models/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors"),
    ModelConfig(path="models/black-forest-labs/FLUX.1-dev/text_encoder/model.safetensors"),
    ModelConfig(path="models/black-forest-labs/FLUX.1-dev/text_encoder_2/"),
    ModelConfig(path="models/black-forest-labs/FLUX.1-dev/ae.safetensors"),
]
```

那么训练时需设置为

```shell
--model_paths '[
    "models/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors",
    "models/black-forest-labs/FLUX.1-dev/text_encoder/model.safetensors",
    "models/black-forest-labs/FLUX.1-dev/text_encoder_2/",
    "models/black-forest-labs/FLUX.1-dev/ae.safetensors"
]' \
```

</details>


<details>

<summary>Step 3: 设置可训练模块</summary>

训练框架支持训练基础模型，或 LoRA 模型。以下是几个例子：

* 全量训练 DiT 部分：`--trainable_models dit`
* 训练 DiT 部分的 LoRA 模型：`--lora_base_model dit --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" --lora_rank 32`

此外，由于训练脚本中加载了多个模块（text encoder、dit、vae），保存模型文件时需要移除前缀，例如在全量训练 DiT 部分或者训练 DiT 部分的 LoRA 模型时，请设置 `--remove_prefix_in_ckpt pipe.dit.`

</details>


<details>

<summary>Step 4: 启动训练程序</summary>

我们为每一个模型编写了训练命令，请参考本文档开头的表格。

</details>
