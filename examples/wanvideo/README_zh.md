# 通义万相（Wan）

[Switch to English](./README.md)

Wan 是由阿里巴巴通义实验室开源的一系列视频生成模型。

**DiffSynth-Studio 启用了新的推理和训练框架，如需使用旧版本，请点击[这里](https://github.com/modelscope/DiffSynth-Studio/tree/3edf3583b1f08944cee837b94d9f84d669c2729c)。**

## 安装

在使用本系列模型之前，请通过源码安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

## 快速开始

通过运行以下代码可以快速加载 [Wan-AI/Wan2.1-T2V-1.3B](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B) 模型并进行推理

```python
import torch
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()

video = pipe(
    prompt="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
)
save_video(video, "video1.mp4", fps=15, quality=5)
```

## 模型总览

|模型 ID|额外参数|推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
|[Wan-AI/Wan2.2-I2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)|`input_image`|[code](./model_inference/Wan2.2-I2V-A14B.py)|[code](./model_training/full/Wan2.2-I2V-A14B.sh)|[code](./model_training/validate_full/Wan2.2-I2V-A14B.py)|[code](./model_training/lora/Wan2.2-I2V-A14B.sh)|[code](./model_training/validate_lora/Wan2.2-I2V-A14B.py)|
|[Wan-AI/Wan2.2-T2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B)||[code](./model_inference/Wan2.2-T2V-A14B.py)|[code](./model_training/full/Wan2.2-T2V-A14B.sh)|[code](./model_training/validate_full/Wan2.2-T2V-A14B.py)|[code](./model_training/lora/Wan2.2-T2V-A14B.sh)|[code](./model_training/validate_lora/Wan2.2-T2V-A14B.py)|
|[Wan-AI/Wan2.2-TI2V-5B](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B)|`input_image`|[code](./model_inference/Wan2.2-TI2V-5B.py)|[code](./model_training/full/Wan2.2-TI2V-5B.sh)|[code](./model_training/validate_full/Wan2.2-TI2V-5B.py)|[code](./model_training/lora/Wan2.2-TI2V-5B.sh)|[code](./model_training/validate_lora/Wan2.2-TI2V-5B.py)|
|[PAI/Wan2.2-Fun-A14B-InP](https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-InP)|`input_image`, `end_image`|[code](./model_inference/Wan2.2-Fun-A14B-InP.py)|[code](./model_training/full/Wan2.2-Fun-A14B-InP.sh)|[code](./model_training/validate_full/Wan2.2-Fun-A14B-InP.py)|[code](./model_training/lora/Wan2.2-Fun-A14B-InP.sh)|[code](./model_training/validate_lora/Wan2.2-Fun-A14B-InP.py)|
|[PAI/Wan2.2-Fun-A14B-Control](https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-Control)|`control_video`, `reference_image`|[code](./model_inference/Wan2.2-Fun-A14B-Control.py)|[code](./model_training/full/Wan2.2-Fun-A14B-Control.sh)|[code](./model_training/validate_full/Wan2.2-Fun-A14B-Control.py)|[code](./model_training/lora/Wan2.2-Fun-A14B-Control.sh)|[code](./model_training/validate_lora/Wan2.2-Fun-A14B-Control.py)|
|[PAI/Wan2.2-Fun-A14B-Control-Camera](https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-Control-Camera)|`control_camera_video`, `input_image`|[code](./model_inference/Wan2.2-Fun-A14B-Control-Camera.py)|[code](./model_training/full/Wan2.2-Fun-A14B-Control-Camera.sh)|[code](./model_training/validate_full/Wan2.2-Fun-A14B-Control-Camera.py)|[code](./model_training/lora/Wan2.2-Fun-A14B-Control-Camera.sh)|[code](./model_training/validate_lora/Wan2.2-Fun-A14B-Control-Camera.py)|
|[Wan-AI/Wan2.1-T2V-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B)||[code](./model_inference/Wan2.1-T2V-1.3B.py)|[code](./model_training/full/Wan2.1-T2V-1.3B.sh)|[code](./model_training/validate_full/Wan2.1-T2V-1.3B.py)|[code](./model_training/lora/Wan2.1-T2V-1.3B.sh)|[code](./model_training/validate_lora/Wan2.1-T2V-1.3B.py)|
|[Wan-AI/Wan2.1-T2V-14B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B)||[code](./model_inference/Wan2.1-T2V-14B.py)|[code](./model_training/full/Wan2.1-T2V-14B.sh)|[code](./model_training/validate_full/Wan2.1-T2V-14B.py)|[code](./model_training/lora/Wan2.1-T2V-14B.sh)|[code](./model_training/validate_lora/Wan2.1-T2V-14B.py)|
|[Wan-AI/Wan2.1-I2V-14B-480P](https://modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P)|`input_image`|[code](./model_inference/Wan2.1-I2V-14B-480P.py)|[code](./model_training/full/Wan2.1-I2V-14B-480P.sh)|[code](./model_training/validate_full/Wan2.1-I2V-14B-480P.py)|[code](./model_training/lora/Wan2.1-I2V-14B-480P.sh)|[code](./model_training/validate_lora/Wan2.1-I2V-14B-480P.py)|
|[Wan-AI/Wan2.1-I2V-14B-720P](https://modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P)|`input_image`|[code](./model_inference/Wan2.1-I2V-14B-720P.py)|[code](./model_training/full/Wan2.1-I2V-14B-720P.sh)|[code](./model_training/validate_full/Wan2.1-I2V-14B-720P.py)|[code](./model_training/lora/Wan2.1-I2V-14B-720P.sh)|[code](./model_training/validate_lora/Wan2.1-I2V-14B-720P.py)|
|[Wan-AI/Wan2.1-FLF2V-14B-720P](https://modelscope.cn/models/Wan-AI/Wan2.1-FLF2V-14B-720P)|`input_image`, `end_image`|[code](./model_inference/Wan2.1-FLF2V-14B-720P.py)|[code](./model_training/full/Wan2.1-FLF2V-14B-720P.sh)|[code](./model_training/validate_full/Wan2.1-FLF2V-14B-720P.py)|[code](./model_training/lora/Wan2.1-FLF2V-14B-720P.sh)|[code](./model_training/validate_lora/Wan2.1-FLF2V-14B-720P.py)|
|[PAI/Wan2.1-Fun-1.3B-InP](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-InP)|`input_image`, `end_image`|[code](./model_inference/Wan2.1-Fun-1.3B-InP.py)|[code](./model_training/full/Wan2.1-Fun-1.3B-InP.sh)|[code](./model_training/validate_full/Wan2.1-Fun-1.3B-InP.py)|[code](./model_training/lora/Wan2.1-Fun-1.3B-InP.sh)|[code](./model_training/validate_lora/Wan2.1-Fun-1.3B-InP.py)|
|[PAI/Wan2.1-Fun-1.3B-Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-Control)|`control_video`|[code](./model_inference/Wan2.1-Fun-1.3B-Control.py)|[code](./model_training/full/Wan2.1-Fun-1.3B-Control.sh)|[code](./model_training/validate_full/Wan2.1-Fun-1.3B-Control.py)|[code](./model_training/lora/Wan2.1-Fun-1.3B-Control.sh)|[code](./model_training/validate_lora/Wan2.1-Fun-1.3B-Control.py)|
|[PAI/Wan2.1-Fun-14B-InP](https://modelscope.cn/models/PAI/Wan2.1-Fun-14B-InP)|`input_image`, `end_image`|[code](./model_inference/Wan2.1-Fun-14B-InP.py)|[code](./model_training/full/Wan2.1-Fun-14B-InP.sh)|[code](./model_training/validate_full/Wan2.1-Fun-14B-InP.py)|[code](./model_training/lora/Wan2.1-Fun-14B-InP.sh)|[code](./model_training/validate_lora/Wan2.1-Fun-14B-InP.py)|
|[PAI/Wan2.1-Fun-14B-Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-14B-Control)|`control_video`|[code](./model_inference/Wan2.1-Fun-14B-Control.py)|[code](./model_training/full/Wan2.1-Fun-14B-Control.sh)|[code](./model_training/validate_full/Wan2.1-Fun-14B-Control.py)|[code](./model_training/lora/Wan2.1-Fun-14B-Control.sh)|[code](./model_training/validate_lora/Wan2.1-Fun-14B-Control.py)|
|[PAI/Wan2.1-Fun-V1.1-1.3B-Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control)|`control_video`, `reference_image`|[code](./model_inference/Wan2.1-Fun-V1.1-1.3B-Control.py)|[code](./model_training/full/Wan2.1-Fun-V1.1-1.3B-Control.sh)|[code](./model_training/validate_full/Wan2.1-Fun-V1.1-1.3B-Control.py)|[code](./model_training/lora/Wan2.1-Fun-V1.1-1.3B-Control.sh)|[code](./model_training/validate_lora/Wan2.1-Fun-V1.1-1.3B-Control.py)|
|[PAI/Wan2.1-Fun-V1.1-14B-Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control)|`control_video`, `reference_image`|[code](./model_inference/Wan2.1-Fun-V1.1-14B-Control.py)|[code](./model_training/full/Wan2.1-Fun-V1.1-14B-Control.sh)|[code](./model_training/validate_full/Wan2.1-Fun-V1.1-14B-Control.py)|[code](./model_training/lora/Wan2.1-Fun-V1.1-14B-Control.sh)|[code](./model_training/validate_lora/Wan2.1-Fun-V1.1-14B-Control.py)|
|[PAI/Wan2.1-Fun-V1.1-1.3B-InP](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-InP)|`input_image`, `end_image`|[code](./model_inference/Wan2.1-Fun-V1.1-1.3B-InP.py)|[code](./model_training/full/Wan2.1-Fun-V1.1-1.3B-InP.sh)|[code](./model_training/validate_full/Wan2.1-Fun-V1.1-1.3B-InP.py)|[code](./model_training/lora/Wan2.1-Fun-V1.1-1.3B-InP.sh)|[code](./model_training/validate_lora/Wan2.1-Fun-V1.1-1.3B-InP.py)|
|[PAI/Wan2.1-Fun-V1.1-14B-InP](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-InP)|`input_image`, `end_image`|[code](./model_inference/Wan2.1-Fun-V1.1-14B-InP.py)|[code](./model_training/full/Wan2.1-Fun-V1.1-14B-InP.sh)|[code](./model_training/validate_full/Wan2.1-Fun-V1.1-14B-InP.py)|[code](./model_training/lora/Wan2.1-Fun-V1.1-14B-InP.sh)|[code](./model_training/validate_lora/Wan2.1-Fun-V1.1-14B-InP.py)|
|[PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera)|`control_camera_video`, `input_image`|[code](./model_inference/Wan2.1-Fun-V1.1-1.3B-Control-Camera.py)|[code](./model_training/full/Wan2.1-Fun-V1.1-1.3B-Control-Camera.sh)|[code](./model_training/validate_full/Wan2.1-Fun-V1.1-1.3B-Control-Camera.py)|[code](./model_training/lora/Wan2.1-Fun-V1.1-1.3B-Control-Camera.sh)|[code](./model_training/validate_lora/Wan2.1-Fun-V1.1-1.3B-Control-Camera.py)|
|[PAI/Wan2.1-Fun-V1.1-14B-Control-Camera](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control-Camera)|`control_camera_video`, `input_image`|[code](./model_inference/Wan2.1-Fun-V1.1-14B-Control-Camera.py)|[code](./model_training/full/Wan2.1-Fun-V1.1-14B-Control-Camera.sh)|[code](./model_training/validate_full/Wan2.1-Fun-V1.1-14B-Control-Camera.py)|[code](./model_training/lora/Wan2.1-Fun-V1.1-14B-Control-Camera.sh)|[code](./model_training/validate_lora/Wan2.1-Fun-V1.1-14B-Control-Camera.py)|
|[iic/VACE-Wan2.1-1.3B-Preview](https://modelscope.cn/models/iic/VACE-Wan2.1-1.3B-Preview)|`vace_control_video`, `vace_reference_image`|[code](./model_inference/Wan2.1-VACE-1.3B-Preview.py)|[code](./model_training/full/Wan2.1-VACE-1.3B-Preview.sh)|[code](./model_training/validate_full/Wan2.1-VACE-1.3B-Preview.py)|[code](./model_training/lora/Wan2.1-VACE-1.3B-Preview.sh)|[code](./model_training/validate_lora/Wan2.1-VACE-1.3B-Preview.py)|
|[Wan-AI/Wan2.1-VACE-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-VACE-1.3B)|`vace_control_video`, `vace_reference_image`|[code](./model_inference/Wan2.1-VACE-1.3B.py)|[code](./model_training/full/Wan2.1-VACE-1.3B.sh)|[code](./model_training/validate_full/Wan2.1-VACE-1.3B.py)|[code](./model_training/lora/Wan2.1-VACE-1.3B.sh)|[code](./model_training/validate_lora/Wan2.1-VACE-1.3B.py)|
|[Wan-AI/Wan2.1-VACE-14B](https://modelscope.cn/models/Wan-AI/Wan2.1-VACE-14B)|`vace_control_video`, `vace_reference_image`|[code](./model_inference/Wan2.1-VACE-14B.py)|[code](./model_training/full/Wan2.1-VACE-14B.sh)|[code](./model_training/validate_full/Wan2.1-VACE-14B.py)|[code](./model_training/lora/Wan2.1-VACE-14B.sh)|[code](./model_training/validate_lora/Wan2.1-VACE-14B.py)|
|[DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1](https://modelscope.cn/models/DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1)|`motion_bucket_id`|[code](./model_inference/Wan2.1-1.3b-speedcontrol-v1.py)|[code](./model_training/full/Wan2.1-1.3b-speedcontrol-v1.sh)|[code](./model_training/validate_full/Wan2.1-1.3b-speedcontrol-v1.py)|[code](./model_training/lora/Wan2.1-1.3b-speedcontrol-v1.sh)|[code](./model_training/validate_lora/Wan2.1-1.3b-speedcontrol-v1.py)|

## 模型推理

以下部分将会帮助您理解我们的功能并编写推理代码。

<details>

<summary>加载模型</summary>

模型通过 `from_pretrained` 加载：

```python
import torch
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
)
```

其中 `torch_dtype` 和 `device` 是计算精度和计算设备。`model_configs` 可通过多种方式配置模型路径：

* 从[魔搭社区](https://modelscope.cn/)下载模型并加载。此时需要填写 `model_id` 和 `origin_file_pattern`，例如

```python
ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
```

* 从本地文件路径加载模型。此时需要填写 `path`，例如

```python
ModelConfig(path="models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
```

对于从多个文件加载的单一模型，使用列表即可，例如

```python
ModelConfig(path=[
    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors",
])
```

`ModelConfig` 提供了额外的参数用于控制模型加载时的行为：

* `local_model_path`: 用于保存下载模型的路径，默认值为 `"./models"`。
* `skip_download`: 是否跳过下载，默认值为 `False`。当您的网络无法访问[魔搭社区](https://modelscope.cn/)时，请手动下载必要的文件，并将其设置为 `True`。

`from_pretrained` 提供了额外的参数用于控制模型加载时的行为：

* `tokenizer_config`: Wan 模型的 tokenizer 路径，默认值为 `ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*")`。
* `redirect_common_files`: 是否重定向重复模型文件，默认值为 `True`。由于 Wan 系列模型包括多个基础模型，每个基础模型的 text encoder 等模块都是相同的，为避免重复下载，我们会对模型路径进行重定向。
* `use_usp`: 是否启用 Unified Sequence Parallel，默认值为 `False`。用于多 GPU 并行推理。

</details>


<details>

<summary>显存管理</summary>

DiffSynth-Studio 为 Wan 模型提供了细粒度的显存管理，让模型能够在低显存设备上进行推理，可通过以下代码开启 offload 功能，在显存有限的设备上将部分模块 offload 到内存中。

```python
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()
```

FP8 量化功能也是支持的：

```python
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_dtype=torch.float8_e4m3fn),
    ],
)
pipe.enable_vram_management()
```

FP8 量化和 offload 可同时开启：

```python
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu", offload_dtype=torch.float8_e4m3fn),
    ],
)
pipe.enable_vram_management()
```

FP8 量化能够大幅度减少显存占用，但不会加速，部分模型在 FP8 量化下会出现精度不足导致的画面模糊、撕裂、失真问题，请谨慎使用 FP8 量化。

开启显存管理后，框架会自动根据设备上的剩余显存确定显存管理策略。`enable_vram_management` 函数提供了以下参数，用于手动控制显存管理策略：

* `vram_limit`: 显存占用量限制（GB），默认占用设备上的剩余显存。注意这不是一个绝对限制，当设置的显存不足以支持模型进行推理，但实际可用显存足够时，将会以最小化显存占用的形式进行推理。将其设置为0时，将会实现理论最小显存占用。
* `vram_buffer`: 显存缓冲区大小（GB），默认为 0.5GB。由于部分较大的神经网络层在 onload 阶段会不可控地占用更多显存，因此一个显存缓冲区是必要的，理论上的最优值为模型中最大的层所占的显存。
* `num_persistent_param_in_dit`: DiT 模型中常驻显存的参数数量（个），默认为无限制。我们将会在未来删除这个参数，请不要依赖这个参数。

</details>


<details>

<summary>推理加速</summary>

Wan 支持多种加速方案，包括

* 高效注意力机制实现：当您的 Python 环境中安装过这些注意力机制实现方案时，我们将会按照以下优先级自动启用。
    * [Flash Attention 3](https://github.com/Dao-AILab/flash-attention)
    * [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)
    * [Sage Attention](https://github.com/thu-ml/SageAttention)
    * [torch SDPA](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) (默认设置，建议安装 `torch>=2.5.0`)
* 统一序列并行：基于 [xDiT](https://github.com/xdit-project/xDiT) 实现的序列并行，请参考[示例代码](./acceleration/unified_sequence_parallel.py)，使用以下命令运行：

```shell
pip install "xfuser[flash-attn]>=0.4.3"
torchrun --standalone --nproc_per_node=8 examples/wanvideo/acceleration/unified_sequence_parallel.py
```

* TeaCache：加速技术 [TeaCache](https://github.com/ali-vilab/TeaCache)，请参考[示例代码](./acceleration/teacache.py)。

</details>


<details>

<summary>输入参数</summary>

Pipeline 在推理阶段能够接收以下输入参数：

* `prompt`: 提示词，描述画面中出现的内容。
* `negative_prompt`: 负向提示词，描述画面中不应该出现的内容，默认值为 `""`。
* `input_image`: 输入图片，适用于图生视频模型，例如 [`Wan-AI/Wan2.1-I2V-14B-480P`](https://modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P)、[`PAI/Wan2.1-Fun-1.3B-InP`](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-InP)，以及首尾帧模型，例如 [`Wan-AI/Wan2.1-FLF2V-14B-720P`](Wan-AI/Wan2.1-FLF2V-14B-720P)。
* `end_image`: 结尾帧，适用于首尾帧模型，例如 [`Wan-AI/Wan2.1-FLF2V-14B-720P`](Wan-AI/Wan2.1-FLF2V-14B-720P)。
* `input_video`: 输入视频，用于视频生视频，适用于任意 Wan 系列模型，需与参数 `denoising_strength` 配合使用。
* `denoising_strength`: 去噪强度，范围为 [0, 1]。数值越小，生成的视频越接近 `input_video`。
* `control_video`: 控制视频，适用于带控制能力的 Wan 系列模型，例如 [`PAI/Wan2.1-Fun-1.3B-Control`](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-Control)。
* `reference_image`: 参考图片，适用于带参考图能力的 Wan 系列模型，例如 [`PAI/Wan2.1-Fun-V1.1-1.3B-Control`](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control)。
* `camera_control_direction`: 镜头控制方向，可选 "Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown" 之一，适用于 Camera-Control 模型，例如 [PAI/Wan2.1-Fun-V1.1-14B-Control-Camera](https://www.modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control-Camera)。
* `camera_control_speed`: 镜头控制速度，适用于 Camera-Control 模型，例如 [PAI/Wan2.1-Fun-V1.1-14B-Control-Camera](https://www.modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control-Camera)。
* `camera_control_origin`: 镜头控制序列的原点坐标，请参考[原论文](https://arxiv.org/pdf/2404.02101)进行设置，适用于 Camera-Control 模型，例如 [PAI/Wan2.1-Fun-V1.1-14B-Control-Camera](https://www.modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control-Camera)。
* `vace_video`: VACE 模型的输入视频，适用于 VACE 系列模型，例如 [`iic/VACE-Wan2.1-1.3B-Preview`](https://modelscope.cn/models/iic/VACE-Wan2.1-1.3B-Preview)。
* `vace_video_mask`: VACE 模型的 mask 视频，适用于 VACE 系列模型，例如 [`iic/VACE-Wan2.1-1.3B-Preview`](https://modelscope.cn/models/iic/VACE-Wan2.1-1.3B-Preview)。
* `vace_reference_image`: VACE 模型的参考图片，适用于 VACE 系列模型，例如 [`iic/VACE-Wan2.1-1.3B-Preview`](https://modelscope.cn/models/iic/VACE-Wan2.1-1.3B-Preview)。
* `vace_scale`: VACE 模型对基础模型的影响程度，默认为1。数值越大，控制强度越高，但画面崩坏概率越大。
* `seed`: 随机种子。默认为 `None`，即完全随机。
* `rand_device`: 生成随机高斯噪声矩阵的计算设备，默认为 `"cpu"`。当设置为 `cuda` 时，在不同 GPU 上会导致不同的生成结果。
* `height`: 帧高度，默认为 480。需设置为 16 的倍数，不满足时向上取整。
* `width`: 帧宽度，默认为 832。需设置为 16 的倍数，不满足时向上取整。
* `num_frames`: 帧数，默认为 81。需设置为 4 的倍数 + 1，不满足时向上取整，最小值为 1。
* `cfg_scale`: Classifier-free guidance 机制的数值，默认为 5。数值越大，提示词的控制效果越强，但画面崩坏的概率越大。
* `cfg_merge`: 是否合并 Classifier-free guidance 的两侧进行统一推理，默认为 `False`。该参数目前仅在基础的文生视频和图生视频模型上生效。
* `switch_DiT_boundary`: 切换 DiT 模型的时间点，默认值为 0.875，仅对多 DiT 的混合模型生效，例如 [Wan-AI/Wan2.2-I2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)。
* `num_inference_steps`: 推理次数，默认值为 50。
* `sigma_shift`: Rectified Flow 理论中的参数，默认为 5。数值越大，模型在去噪的开始阶段停留的步骤数越多，可适当调大这个参数来提高画面质量，但会因生成过程与训练过程不一致导致生成的视频内容与训练数据存在差异。
* `motion_bucket_id`: 运动幅度，范围为 [0, 100]。适用于速度控制模块，例如 [`DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1`](https://modelscope.cn/models/DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1)，数值越大，运动幅度越大。
* `tiled`: 是否启用 VAE 分块推理，默认为 `False`。设置为 `True` 时可显著减少 VAE 编解码阶段的显存占用，会产生少许误差，以及少量推理时间延长。
* `tile_size`: VAE 编解码阶段的分块大小，默认为 (30, 52)，仅在 `tiled=True` 时生效。
* `tile_stride`: VAE 编解码阶段的分块步长，默认为 (15, 26)，仅在 `tiled=True` 时生效，需保证其数值小于或等于 `tile_size`。
* `sliding_window_size`: DiT 部分的滑动窗口大小。实验性功能，效果不稳定。
* `sliding_window_stride`: DiT 部分的滑动窗口步长。实验性功能，效果不稳定。
* `tea_cache_l1_thresh`: TeaCache 的阈值，数值越大，速度越快，画面质量越差。请注意，开启 TeaCache 后推理速度并非均匀，因此进度条上显示的剩余时间将会变得不准确。
* `tea_cache_model_id`: TeaCache 的参数模板，可选 `"Wan2.1-T2V-1.3B"`、`Wan2.1-T2V-14B`、`Wan2.1-I2V-14B-480P`、`Wan2.1-I2V-14B-720P` 之一。
* `progress_bar_cmd`: 进度条，默认为 `tqdm.tqdm`。可通过设置为 `lambda x:x` 来屏蔽进度条。

</details>


## 模型训练

Wan 系列模型训练通过统一的 [`./model_training/train.py`](./model_training/train.py) 脚本进行。

<details>

<summary>脚本参数</summary>

脚本包含以下参数：

* 数据集
  * `--dataset_base_path`: 数据集的根路径。
  * `--dataset_metadata_path`: 数据集的元数据文件路径。
  * `--height`: 图像或视频的高度。将 `height` 和 `width` 留空以启用动态分辨率。
  * `--width`: 图像或视频的宽度。将 `height` 和 `width` 留空以启用动态分辨率。
  * `--num_frames`: 每个视频中的帧数。帧从视频前缀中采样。
  * `--data_file_keys`: 元数据中的数据文件键。用逗号分隔。
  * `--dataset_repeat`: 每个 epoch 中数据集重复的次数。
  * `--dataset_num_workers`: 每个 Dataloder 的进程数量。
* 模型
  * `--model_paths`: 要加载的模型路径。JSON 格式。
  * `--model_id_with_origin_paths`: 带原始路径的模型 ID，例如 Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors。用逗号分隔。
  * `--max_timestep_boundary`: Timestep 区间最大值，范围为 0～1，默认为 1，仅在多 DiT 的混合模型训练中需要手动设置，例如 [Wan-AI/Wan2.2-I2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)。
  * `--min_timestep_boundary`: Timestep 区间最小值，范围为 0～1，默认为 1，仅在多 DiT 的混合模型训练中需要手动设置，例如 [Wan-AI/Wan2.2-I2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)。
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
  * `--use_gradient_checkpointing_offload`: 是否将 gradient checkpointing 卸载到内存中。

此外，训练框架基于 [`accelerate`](https://huggingface.co/docs/accelerate/index) 构建，在开始训练前运行 `accelerate config` 可配置 GPU 的相关参数。对于部分模型训练（例如 14B 模型的全量训练）脚本，我们提供了建议的 `accelerate` 配置文件，可在对应的训练脚本中查看。

</details>


<details>

<summary>Step 1: 准备数据集</summary>

数据集包含一系列文件，我们建议您这样组织数据集文件：

```
data/example_video_dataset/
├── metadata.csv
├── video1.mp4
└── video2.mp4
```

其中 `video1.mp4`、`video2.mp4` 为训练用视频数据，`metadata.csv` 为元数据列表，例如

```
video,prompt
video1.mp4,"from sunset to night, a small town, light, house, river"
video2.mp4,"a dog is running"
```

我们构建了一个样例视频数据集，以方便您进行测试，通过以下命令可以下载这个数据集：

```shell
modelscope download --dataset DiffSynth-Studio/example_video_dataset --local_dir ./data/example_video_dataset
```

数据集支持视频和图片混合训练，支持的视频文件格式包括 `"mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"`，支持的图片格式包括 `"jpg", "jpeg", "png", "webp"`。

视频的尺寸可通过脚本参数 `--height`、`--width`、`--num_frames` 控制。在每个视频中，前 `num_frames` 帧会被用于训练，因此当视频长度不足 `num_frames` 帧时会报错，图片文件会被视为单帧视频。当 `--height` 和 `--width` 为空时将会开启动态分辨率，按照数据集中每个视频或图片的实际宽高训练。

**我们强烈建议使用固定分辨率训练，并避免图像和视频混合训练，因为在多卡训练中存在负载均衡问题。**

当模型需要额外输入时，例如具备控制能力的模型 [`PAI/Wan2.1-Fun-1.3B-Control`](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-Control) 所需的 `control_video`，请在数据集中补充相应的列，例如：

```
video,prompt,control_video
video1.mp4,"from sunset to night, a small town, light, house, river",video1_softedge.mp4
```

额外输入若包含视频和图像文件，则需要在 `--data_file_keys` 参数中指定要解析的列名。该参数的默认值为 `"image,video"`，即解析列名为 `image` 和 `video` 的列。可根据额外输入增加相应的列名，例如 `--data_file_keys "image,video,control_video"`，同时启用 `--input_contains_control_video`。

</details>


<details>

<summary>Step 2: 加载模型</summary>

类似于推理时的模型加载逻辑，可直接通过模型 ID 配置要加载的模型。例如，推理时我们通过以下设置加载模型

```python
model_configs=[
    ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
    ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
    ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth"),
]
```

那么在训练时，填入以下参数即可加载对应的模型。

```shell
--model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth"
```

如果您希望从本地文件加载模型，例如推理时

```python
model_configs=[
    ModelConfig(path=[
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors",
    ]),
    ModelConfig(path="models/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth"),
    ModelConfig(path="models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth"),
]
```

那么训练时需设置为

```shell
--model_paths '[
    [
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
        "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors"
    ],
    "models/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
    "models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth"
]' \
```

</details>


<details>

<summary>Step 3: 设置可训练模块</summary>

训练框架支持训练基础模型，或 LoRA 模型。以下是几个例子：

* 全量训练 DiT 部分：`--trainable_models dit`
* 训练 DiT 部分的 LoRA 模型：`--lora_base_model dit --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank 32`
* 训练 DiT 部分的 LoRA 和 Motion Controller 部分（是的，可以训练这种花里胡哨的结构）：`--trainable_models motion_controller --lora_base_model dit --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank 32`

此外，由于训练脚本中加载了多个模块（text encoder、dit、vae），保存模型文件时需要移除前缀，例如在全量训练 DiT 部分或者训练 DiT 部分的 LoRA 模型时，请设置 `--remove_prefix_in_ckpt pipe.dit.`

</details>


<details>

<summary>Step 4: 启动训练程序</summary>

我们为每一个模型编写了训练命令，请参考本文档开头的表格。

请注意，14B 模型全量训练需要8个GPU，每个GPU的显存至少为80G。全量训练这些14B模型时需要安装 `deepspeed`（`pip install deepspeed`），我们编写了建议的[配置文件](./model_training/full/accelerate_config_14B.yaml)，这个配置文件会在对应的训练脚本中被加载，这些脚本已在 8*A100 上测试过。

训练脚本的默认视频尺寸为 `480*832*81`，提升分辨率将可能导致显存不足，请添加参数 `--use_gradient_checkpointing_offload` 降低显存占用。

</details>

## 案例展示

1.3B 文生视频：

https://github.com/user-attachments/assets/124397be-cd6a-4f29-a87c-e4c695aaabb8

给狗狗戴上墨镜（1.3B 视频生视频）：

https://github.com/user-attachments/assets/272808d7-fbeb-4747-a6df-14a0860c75fb

14B 文生视频：

https://github.com/user-attachments/assets/3908bc64-d451-485a-8b61-28f6d32dd92f

14B 图生视频：

https://github.com/user-attachments/assets/c0bdd5ca-292f-45ed-b9bc-afe193156e75

LoRA 训练：

https://github.com/user-attachments/assets/9bd8e30b-97e8-44f9-bb6f-da004ba376a9
