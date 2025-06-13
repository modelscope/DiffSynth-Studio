# Wan 2.1

[切换到中文](./README_zh.md)

Wan 2.1 is a collection of video synthesis models open-sourced by Alibaba.

**DiffSynth-Studio has adopted a new inference and training framework. To use the previous version, please click [here](https://github.com/modelscope/DiffSynth-Studio/tree/3edf3583b1f08944cee837b94d9f84d669c2729c).**

## Installation

Before using this model, please install DiffSynth-Studio from **source code**.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

## Overview

| Model ID | Extra Parameters | Inference | Full Training | Full Training Validation | LoRA Training | LoRA Training Validation |
|-|-|-|-|-|-|-|
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


## Model Inference

The following sections will help you understand our functionalities and write inference code.

<details>

<summary>Loading the Model</summary>

The model is loaded using `from_pretrained`:

```python
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

Here, `torch_dtype` and `device` specify the computation precision and device respectively. The `model_configs` can be used to configure model paths in various ways:

* Downloading the model from [ModelScope](https://modelscope.cn/) and loading it. In this case, both `model_id` and `origin_file_pattern` need to be specified, for example:

```python
ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
```

* Loading the model from a local file path. In this case, the `path` parameter needs to be specified, for example:

```python
ModelConfig(path="models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
```

For models that are loaded from multiple files, simply use a list, for example:

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

The `from_pretrained` function also provides additional parameters to control the behavior during model loading:

* `tokenizer_config`: Path to the tokenizer of the Wan model. Default value is `ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*")`.
* `local_model_path`: Path where downloaded models are saved. Default value is `"./models"`.
* `skip_download`: Whether to skip downloading models. Default value is `False`. When your network cannot access [ModelScope](https://modelscope.cn/), manually download the necessary files and set this to `True`.
* `redirect_common_files`: Whether to redirect duplicate model files. Default value is `True`. Since the Wan series models include multiple base models, some modules like text encoder are shared across these models. To avoid redundant downloads, we redirect the model paths.
* `use_usp`: Whether to enable Unified Sequence Parallel. Default value is `False`. Used for multi-GPU parallel inference.

</details>

<details>

<summary>VRAM Management</summary>

DiffSynth-Studio provides fine-grained VRAM management for the Wan model, allowing it to run on devices with limited VRAM. You can enable offloading functionality via the following code, which moves parts of the model to system memory on devices with limited VRAM:

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

FP8 quantization is also supported:

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

Both FP8 quantization and offloading can be enabled simultaneously:

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

FP8 quantization significantly reduces VRAM usage but does not accelerate computations. Some models may experience issues such as blurry, torn, or distorted outputs due to insufficient precision when using FP8 quantization. Use FP8 quantization with caution.

The `enable_vram_management` function provides the following parameters to control VRAM usage:

* `vram_limit`: VRAM usage limit (in GB). By default, it uses all available VRAM on the device. Note that this is not an absolute limit; if the specified VRAM is insufficient but more VRAM is actually available, inference will proceed using the minimum required VRAM.
* `vram_buffer`: Size of the VRAM buffer (in GB). Default is 0.5GB. Since certain large neural network layers may consume more VRAM unpredictably during their execution phase, a VRAM buffer is necessary. Ideally, this should match the maximum VRAM consumed by any single layer in the model.
* `num_persistent_param_in_dit`: Number of persistent parameters in DiT models. By default, there is no limit. We plan to remove this parameter in the future, so please avoid relying on it.

</details>

<details>

<summary>Inference Acceleration</summary>

Wan supports multiple acceleration techniques, including:

* **Efficient attention implementations**: If any of these attention implementations are installed in your Python environment, they will be automatically enabled in the following priority:
    * [Flash Attention 3](https://github.com/Dao-AILab/flash-attention)  
    * [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)  
    * [Sage Attention](https://github.com/thu-ml/SageAttention)  
    * [torch SDPA](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)   (default setting; we recommend installing `torch>=2.5.0`)
* **Unified Sequence Parallel**: Sequence parallelism based on [xDiT](https://github.com/xdit-project/xDiT). Please refer to [this example](./acceleration/unified_sequence_parallel.py), and run it using the command: 

```shell
pip install xfuser>=0.4.3
torchrun --standalone --nproc_per_node=8 examples/wanvideo/acceleration/unified_sequence_parallel.py
```

* **TeaCache**: Acceleration technique [TeaCache](https://github.com/ali-vilab/TeaCache). Please refer to [this example](./acceleration/teacache.py).

</details>


<details>

<summary>Input Parameters</summary>

The pipeline accepts the following input parameters during inference:

* `prompt`: Prompt describing the content to appear in the video.
* `negative_prompt`: Negative prompt describing content that should not appear in the video. Default is `""`.
* `input_image`: Input image, applicable for image-to-video models such as [`Wan-AI/Wan2.1-I2V-14B-480P`](https://modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P) and [`PAI/Wan2.1-Fun-1.3B-InP`](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-InP), as well as first-and-last-frame models like [`Wan-AI/Wan2.1-FLF2V-14B-720P`](Wan-AI/Wan2.1-FLF2V-14B-720P).
* `end_image`: End frame, applicable for first-and-last-frame models such as [`Wan-AI/Wan2.1-FLF2V-14B-720P`](Wan-AI/Wan2.1-FLF2V-14B-720P).
* `input_video`: Input video used for video-to-video generation. Applicable to any Wan series model and must be used together with `denoising_strength`.
* `denoising_strength`: Denoising strength in range [0, 1]. A smaller value results in a video closer to `input_video`.
* `control_video`: Control video, applicable to Wan models with control capabilities such as [`PAI/Wan2.1-Fun-1.3B-Control`](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-Control).  
* `reference_image`: Reference image, applicable to Wan models supporting reference images such as [`PAI/Wan2.1-Fun-V1.1-1.3B-Control`](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control).  
* `camera_control_direction`: Camera control direction, optional values are "Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown". Applicable to Camera-Control models, such as [PAI/Wan2.1-Fun-V1.1-14B-Control-Camera](https://www.modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control-Camera).  
* `camera_control_speed`: Camera control speed. Applicable to Camera-Control models, such as [PAI/Wan2.1-Fun-V1.1-14B-Control-Camera](https://www.modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control-Camera).  
* `camera_control_origin`: Origin coordinate of the camera control sequence. Please refer to the [original paper](https://arxiv.org/pdf/2404.02101) for proper configuration. Applicable to Camera-Control models, such as [PAI/Wan2.1-Fun-V1.1-14B-Control-Camera](https://www.modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control-Camera).
* `vace_video`: Input video for VACE models, applicable to the VACE series such as [`iic/VACE-Wan2.1-1.3B-Preview`](https://modelscope.cn/models/iic/VACE-Wan2.1-1.3B-Preview).  
* `vace_video_mask`: Mask video for VACE models, applicable to the VACE series such as [`iic/VACE-Wan2.1-1.3B-Preview`](https://modelscope.cn/models/iic/VACE-Wan2.1-1.3B-Preview).  
* `vace_reference_image`: Reference image for VACE models, applicable to the VACE series such as [`iic/VACE-Wan2.1-1.3B-Preview`](https://modelscope.cn/models/iic/VACE-Wan2.1-1.3B-Preview).  
* `vace_scale`: Influence of the VACE model on the base model, default is 1. Higher values increase control strength but may lead to visual artifacts or breakdowns.
* `seed`: Random seed. Default is `None`, meaning fully random.
* `rand_device`: Device used to generate random Gaussian noise matrix. Default is `"cpu"`. When set to `"cuda"`, different GPUs may produce different generation results.
* `height`: Frame height, default is 480. Must be a multiple of 16; if not, it will be rounded up.
* `width`: Frame width, default is 832. Must be a multiple of 16; if not, it will be rounded up.
* `num_frames`: Number of frames, default is 81. Must be a multiple of 4 plus 1; if not, it will be rounded up, minimum is 1.
* `cfg_scale`: Classifier-free guidance scale, default is 5. Higher values increase adherence to the prompt but may cause visual artifacts.
* `cfg_merge`: Whether to merge both sides of classifier-free guidance for unified inference. Default is `False`. This parameter currently only works for basic text-to-video and image-to-video models.
* `num_inference_steps`: Number of inference steps, default is 50.
* `sigma_shift`: Parameter from Rectified Flow theory, default is 5. Higher values make the model stay longer at the initial denoising stage. Increasing this may improve video quality but may also cause inconsistency between generated videos and training data due to deviation from training behavior.
* `motion_bucket_id`: Motion intensity, range [0, 100], applicable to motion control modules such as [`DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1`](https://modelscope.cn/models/DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1). Larger values indicate more intense motion.  
* `tiled`: Whether to enable tiled VAE inference, default is `False`. Setting to `True` significantly reduces VRAM usage during VAE encoding/decoding but introduces small errors and slightly increases inference time.
* `tile_size`: Tile size during VAE encoding/decoding, default is (30, 52), only effective when `tiled=True`.
* `tile_stride`: Stride of tiles during VAE encoding/decoding, default is (15, 26), only effective when `tiled=True`. Must be less than or equal to `tile_size`.
* `sliding_window_size`: Sliding window size for DiT part. Experimental feature, effects are unstable.
* `sliding_window_stride`: Sliding window stride for DiT part. Experimental feature, effects are unstable.
* `tea_cache_l1_thresh`: Threshold for TeaCache. Larger values result in faster speed but lower quality. Note that after enabling TeaCache, the inference speed is not uniform, so the remaining time shown on the progress bar becomes inaccurate.
* `tea_cache_model_id`: TeaCache parameter template, options include `"Wan2.1-T2V-1.3B"`, `"Wan2.1-T2V-14B"`, `"Wan2.1-I2V-14B-480P"`, `"Wan2.1-I2V-14B-720P"`.
* `progress_bar_cmd`: Progress bar implementation, default is `tqdm.tqdm`. You can set it to `lambda x:x` to disable the progress bar.

</details>

## Model Training

Wan series models are trained using a unified script located at [`./model_training/train.py`](./model_training/train.py).

<details>

<summary>Script Parameters</summary>

The script includes the following parameters:

* Dataset
  * `--dataset_base_path`: Base path of the dataset.
  * `--dataset_metadata_path`: Path to the metadata file of the dataset.
  * `--height`: Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.
  * `--width`: Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.
  * `--num_frames`: Number of frames per video. Frames are sampled from the video prefix.
  * `--data_file_keys`: Data file keys in the metadata. Comma-separated.
  * `--dataset_repeat`: Number of times to repeat the dataset per epoch.
* Models
  * `--model_paths`: Paths to load models. In JSON format.
  * `--model_id_with_origin_paths`: Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.
* Training
  * `--learning_rate`: Learning rate.
  * `--num_epochs`: Number of epochs.
  * `--output_path`: Output save path.
  * `--remove_prefix_in_ckpt`: Remove prefix in ckpt.
* Trainable Modules
  * `--trainable_models`: Models to train, e.g., dit, vae, text_encoder.
  * `--lora_base_model`: Which model LoRA is added to.
  * `--lora_target_modules`: Which layers LoRA is added to.
  * `--lora_rank`: Rank of LoRA.
* Extra Inputs
  * `--extra_inputs`: Additional model inputs, comma-separated.
* VRAM Management
  * `--use_gradient_checkpointing_offload`: Whether to offload gradient checkpointing to CPU memory.

Additionally, the training framework is built upon [`accelerate`](https://huggingface.co/docs/accelerate/index). Before starting training, run `accelerate config` to configure GPU-related parameters. For certain training scripts (e.g., full fine-tuning of 14B models), we provide recommended `accelerate` configuration files, which can be found in the corresponding training scripts.

</details>


<details>

<summary>Step 1: Prepare the Dataset</summary>

The dataset consists of a series of files. We recommend organizing your dataset as follows:

```
data/example_video_dataset/
├── metadata.csv
├── video1.mp4
└── video2.mp4
```

Here, `video1.mp4` and `video2.mp4` are training video files, and `metadata.csv` is the metadata list, for example:

```
video,prompt
video1.mp4,"from sunset to night, a small town, light, house, river"
video2.mp4,"a dog is running"
```

We have prepared a sample video dataset to help you test. You can download it using the following command:

```shell
modelscope download --dataset DiffSynth-Studio/example_video_dataset --local_dir ./data/example_video_dataset
```

The dataset supports mixed training of videos and images. Supported video formats include `"mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"`, and supported image formats include `"jpg", "jpeg", "png", "webp"`.

The resolution of videos can be controlled via script parameters `--height`, `--width`, and `--num_frames`. For each video, the first `num_frames` frames will be used for training; therefore, an error will occur if the video length is less than `num_frames`. Image files will be treated as single-frame videos. When both `--height` and `--width` are left empty, dynamic resolution will be enabled, meaning training will use the actual resolution of each video or image in the dataset.

**We strongly recommend using fixed-resolution training and avoiding mixing images and videos in the same dataset due to load balancing issues in multi-GPU training.**

When the model requires additional inputs, such as the `control_video` needed by control-capable models like [`PAI/Wan2.1-Fun-1.3B-Control`](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-Control), please add corresponding columns in the metadata file, for example:

```
video,prompt,control_video
video1.mp4,"from sunset to night, a small town, light, house, river",video1_softedge.mp4
```

If additional inputs contain video or image files, their column names need to be specified in the `--data_file_keys` parameter. The default value of this parameter is `"image,video"`, meaning it parses columns named `image` and `video`. You can extend this list based on the additional input requirements, for example: `--data_file_keys "image,video,control_video"`, and also enable `--input_contains_control_video`.

</details>


<details>

<summary>Step 2: Load the Model</summary>

Similar to the model loading logic during inference, you can configure the model to be loaded directly via its model ID. For instance, during inference we load the model using:

```python
model_configs=[
    ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
    ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
    ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth"),
]
```

During training, simply use the following parameter to load the corresponding model:

```shell
--model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth"
```

If you want to load the model from local files, for example during inference:

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

Then during training, set the parameter as:

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

<summary>Step 3: Configure Trainable Modules</summary>

The training framework supports full fine-tuning of base models or LoRA-based training. Here are some examples:

* Full fine-tuning of the DiT module: `--trainable_models dit`
* Training a LoRA model for the DiT module: `--lora_base_model dit --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank 32`
* Training both a LoRA model for DiT and the Motion Controller (yes, you can train such advanced structures): `--trainable_models motion_controller --lora_base_model dit --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank 32`

Additionally, since multiple modules (text encoder, dit, vae) are loaded in the training script, you need to remove prefixes when saving model files. For example, when fully fine-tuning the DiT module or training a LoRA version of DiT, please set `--remove_prefix_in_ckpt pipe.dit.`

</details>


<details>

<summary>Step 4: Launch the Training Process</summary>

We have prepared training commands for each model. Please refer to the table at the beginning of this document.

Note that full fine-tuning of the 14B model requires 8 GPUs, each with at least 80GB VRAM. During full fine-tuning of these 14B models, you must install `deepspeed` (`pip install deepspeed`). We have provided recommended [configuration files](./model_training/full/accelerate_config_14B.yaml), which will be loaded automatically in the corresponding training scripts. These scripts have been tested on 8*A100.

The default video resolution in the training script is `480*832*81`. Increasing the resolution may cause out-of-memory errors. To reduce VRAM usage, add the parameter `--use_gradient_checkpointing_offload`.

</details>

## Gallery

1.3B text-to-video:

https://github.com/user-attachments/assets/124397be-cd6a-4f29-a87c-e4c695aaabb8

Put sunglasses on the dog (1.3B video-to-video):

https://github.com/user-attachments/assets/272808d7-fbeb-4747-a6df-14a0860c75fb

14B text-to-video:

https://github.com/user-attachments/assets/3908bc64-d451-485a-8b61-28f6d32dd92f

14B image-to-video:

https://github.com/user-attachments/assets/c0bdd5ca-292f-45ed-b9bc-afe193156e75

LoRA training:

https://github.com/user-attachments/assets/9bd8e30b-97e8-44f9-bb6f-da004ba376a9
