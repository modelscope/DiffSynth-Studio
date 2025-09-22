# Qwen-Image

[Switch to English](./README.md)

Qwen-Image is an open-source image generation model developed by Tongyi Lab, Alibaba.

## Installation

Before using this model series, install DiffSynth-Studio from source code.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git  
cd DiffSynth-Studio
pip install -e .
```

## Quick Start

Run the following code to quickly load the [Qwen/Qwen-Image](https://www.modelscope.cn/models/Qwen/Qwen-Image  ) model and perform inference.

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
prompt = "A detailed portrait of a girl underwater, wearing a blue flowing dress, hair gently floating, clear light and shadow, surrounded by bubbles, calm expression, fine details, dreamy and beautiful."
image = pipe(
    prompt, seed=0, num_inference_steps=40,
    # edit_image=Image.open("xxx.jpg").resize((1328, 1328)) # For Qwen-Image-Edit
)
image.save("image.jpg")
```

## Model Overview

|Model ID|Inference|Low VRAM Inference|Full Training|Validation after Full Training|LoRA Training|Validation after LoRA Training|
|-|-|-|-|-|-|-|
|[Qwen/Qwen-Image](https://www.modelscope.cn/models/Qwen/Qwen-Image)|[code](./model_inference/Qwen-Image.py)|[code](./model_inference_low_vram/Qwen-Image.py)|[code](./model_training/full/Qwen-Image.sh)|[code](./model_training/validate_full/Qwen-Image.py)|[code](./model_training/lora/Qwen-Image.sh)|[code](./model_training/validate_lora/Qwen-Image.py)|
|[Qwen/Qwen-Image-Edit](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit)|[code](./model_inference/Qwen-Image-Edit.py)|[code](./model_inference_low_vram/Qwen-Image-Edit.py)|[code](./model_training/full/Qwen-Image-Edit.sh)|[code](./model_training/validate_full/Qwen-Image-Edit.py)|[code](./model_training/lora/Qwen-Image-Edit.sh)|[code](./model_training/validate_lora/Qwen-Image-Edit.py)|
|[Qwen/Qwen-Image-Edit-2509](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit-2509)|[code](./model_inference/Qwen-Image-Edit-2509.py)|[code](./model_inference_low_vram/Qwen-Image-Edit-2509.py)|[code](./model_training/full/Qwen-Image-Edit-2509.sh)|[code](./model_training/validate_full/Qwen-Image-Edit-2509.py)|[code](./model_training/lora/Qwen-Image-Edit-2509.sh)|[code](./model_training/validate_lora/Qwen-Image-Edit-2509.py)|
|[DiffSynth-Studio/Qwen-Image-Distill-Full](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-Full)|[code](./model_inference/Qwen-Image-Distill-Full.py)|[code](./model_inference_low_vram/Qwen-Image-Distill-Full.py)|[code](./model_training/full/Qwen-Image-Distill-Full.sh)|[code](./model_training/validate_full/Qwen-Image-Distill-Full.py)|[code](./model_training/lora/Qwen-Image-Distill-Full.sh)|[code](./model_training/validate_lora/Qwen-Image-Distill-Full.py)|
|[DiffSynth-Studio/Qwen-Image-Distill-LoRA](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-LoRA)|[code](./model_inference/Qwen-Image-Distill-LoRA.py)|[code](./model_inference_low_vram/Qwen-Image-Distill-LoRA.py)|-|-|[code](./model_training/lora/Qwen-Image-Distill-LoRA.sh)|[code](./model_training/validate_lora/Qwen-Image-Distill-LoRA.py)|
|[DiffSynth-Studio/Qwen-Image-EliGen](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-EliGen)|[code](./model_inference/Qwen-Image-EliGen.py)|[code](./model_inference_low_vram/Qwen-Image-EliGen.py)|-|-|[code](./model_training/lora/Qwen-Image-EliGen.sh)|[code](./model_training/validate_lora/Qwen-Image-EliGen.py)|
|[DiffSynth-Studio/Qwen-Image-EliGen-V2](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-EliGen-V2)|[code](./model_inference/Qwen-Image-EliGen-V2.py)|[code](./model_inference_low_vram/Qwen-Image-EliGen-V2.py)|-|-|[code](./model_training/lora/Qwen-Image-EliGen.sh)|[code](./model_training/validate_lora/Qwen-Image-EliGen.py)|
|[DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny)|[code](./model_inference/Qwen-Image-Blockwise-ControlNet-Canny.py)|[code](./model_inference_low_vram/Qwen-Image-Blockwise-ControlNet-Canny.py)|[code](./model_training/full/Qwen-Image-Blockwise-ControlNet-Canny.sh)|[code](./model_training/validate_full/Qwen-Image-Blockwise-ControlNet-Canny.py)|[code](./model_training/lora/Qwen-Image-Blockwise-ControlNet-Canny.sh)|[code](./model_training/validate_lora/Qwen-Image-Blockwise-ControlNet-Canny.py)|
|[DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth)|[code](./model_inference/Qwen-Image-Blockwise-ControlNet-Depth.py)|[code](./model_inference_low_vram/Qwen-Image-Blockwise-ControlNet-Depth.py)|[code](./model_training/full/Qwen-Image-Blockwise-ControlNet-Depth.sh)|[code](./model_training/validate_full/Qwen-Image-Blockwise-ControlNet-Depth.py)|[code](./model_training/lora/Qwen-Image-Blockwise-ControlNet-Depth.sh)|[code](./model_training/validate_lora/Qwen-Image-Blockwise-ControlNet-Depth.py)|
|[DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint)|[code](./model_inference/Qwen-Image-Blockwise-ControlNet-Inpaint.py)|[code](./model_inference_low_vram/Qwen-Image-Blockwise-ControlNet-Inpaint.py)|[code](./model_training/full/Qwen-Image-Blockwise-ControlNet-Inpaint.sh)|[code](./model_training/validate_full/Qwen-Image-Blockwise-ControlNet-Inpaint.py)|[code](./model_training/lora/Qwen-Image-Blockwise-ControlNet-Inpaint.sh)|[code](./model_training/validate_lora/Qwen-Image-Blockwise-ControlNet-Inpaint.py)|
|[DiffSynth-Studio/Qwen-Image-In-Context-Control-Union](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-In-Context-Control-Union)|[code](./model_inference/Qwen-Image-In-Context-Control-Union.py)|[code](./model_inference_low_vram/Qwen-Image-In-Context-Control-Union.py)|-|-|[code](./model_training/lora/Qwen-Image-In-Context-Control-Union.sh)|[code](./model_training/validate_lora/Qwen-Image-In-Context-Control-Union.py)|
|[DiffSynth-Studio/Qwen-Image-Edit-Lowres-Fix](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Edit-Lowres-Fix)|[code](./model_inference/Qwen-Image-Edit-Lowres-Fix.py)|[code](./model_inference_low_vram/Qwen-Image-Edit-Lowres-Fix.py)|-|-|-|-|

## Model Inference

The following section helps you understand our features and write inference code.

<details>

<summary>Load Model</summary>

Use `from_pretrained` to load the model:

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

Here, `torch_dtype` and `device` set the computation precision and device. `model_configs` can be used in different ways to specify model paths:

* Download the model from [ModelScope](https://modelscope.cn/  ) and load it. In this case, fill in `model_id` and `origin_file_pattern`, for example:

```python
ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
```

* Load the model from a local file path. In this case, fill in `path`, for example:

```python
ModelConfig(path="models/xxx.safetensors")
```

For a single model loaded from multiple files, use a list, for example:

```python
ModelConfig(path=[
    "models/Qwen/Qwen-Image/text_encoder/model-00001-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00002-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00003-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00004-of-00004.safetensors",
])
```

`ModelConfig` provides extra options to control model loading behavior:

* `local_model_path`: Path to save downloaded models. Default is `"./models"`.
* `skip_download`: Whether to skip downloading. Default is `False`. If your network cannot access [ModelScope](https://modelscope.cn/  ), download the required files manually and set this to `True`.

</details>


<details>

<summary>VRAM Management</summary>

DiffSynth-Studio provides fine-grained VRAM management for the Qwen-Image model. This allows the model to run on devices with low VRAM. You can enable the offload feature using the code below. It moves some model parts to CPU memory when GPU memory is limited.

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

FP8 quantization is also supported:

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

You can use FP8 quantization and offload at the same time:

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

FP8 quantization can greatly reduce VRAM use, but it does not speed up inference. Some models may have quality issues like blur, tearing, or distortion when using FP8. Use FP8 with care.

After enabling VRAM management, the framework will automatically choose a memory strategy based on free VRAM. The `enable_vram_management` function has the following options to control this strategy:

* `vram_limit`: VRAM usage limit in GB. By default, it uses all free VRAM on the device. Note that this is not a strict limit. If the set limit is too low but actual free VRAM is enough, the model will run with minimal VRAM use. Set it to 0 for the smallest possible VRAM use.
* `vram_buffer`: VRAM buffer size in GB. Default is 0.5GB. A buffer is needed because large network layers may use more VRAM than expected during loading. The best value is the VRAM size of the largest model layer.
* `num_persistent_param_in_dit`: Number of parameters to keep in VRAM in the DiT model. Default is no limit. This option will be removed in the future. Do not rely on it.
* `enable_dit_fp8_computation`: Whether to enable FP8 computation in the DiT model. This is only applicable to GPUs that support FP8 operations (e.g., H200, etc.). Disabled by default.

</details>


<details>

<summary>Inference Acceleration</summary>

* FP8 Quantization: Choose the appropriate quantization method based on your hardware and requirements.
    * GPUs that do not support FP8 computation (e.g., A100, 4090, etc.): FP8 quantization will only reduce VRAM usage without speeding up inference. Code: [./model_inference_low_vram/Qwen-Image.py](./model_inference_low_vram/Qwen-Image.py)
    * GPUs that support FP8 operations (e.g., H200, etc.): Please install [Flash Attention 3](https://github.com/Dao-AILab/flash-attention). Otherwise, FP8 acceleration will only apply to Linear layers.
        * Faster inference but higher VRAM usage: Use [./accelerate/Qwen-Image-FP8.py](./accelerate/Qwen-Image-FP8.py)
        * Slightly slower inference but lower VRAM usage: Use [./accelerate/Qwen-Image-FP8-offload.py](./accelerate/Qwen-Image-FP8-offload.py)
* Distillation acceleration: We trained two distillation models for fast inference at `cfg_scale=1` and `num_inference_steps=15`.
    * [DiffSynth-Studio/Qwen-Image-Distill-Full](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-Full): Full distillation version. Better image quality but lower LoRA compatibility. Use [./model_inference/Qwen-Image-Distill-Full.py](./model_inference/Qwen-Image-Distill-Full.py).
    * [DiffSynth-Studio/Qwen-Image-Distill-LoRA](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Distill-LoRA): LoRA distillation version. Slightly lower image quality but better LoRA compatibility. Use [./model_inference/Qwen-Image-Distill-LoRA.py](./model_inference/Qwen-Image-Distill-LoRA.py).

</details>


<details>

<summary>Input Parameters</summary>

The pipeline supports the following input parameters during inference:

* `prompt`: Text prompt that describes what should appear in the image.
* `negative_prompt`: Negative prompt that describes what should not appear in the image. Default is `""`.
* `cfg_scale`: Parameter for classifier-free guidance. Default is 1. It takes effect when set to a value greater than 1.
* `input_image`: Input image for image-to-image generation. Used with `denoising_strength`.
* `denoising_strength`: Denoising strength, range from 0 to 1. Default is 1. When close to 0, the output image is similar to the input. When close to 1, the output is more different. Do not set this to a non-1 value if `input_image` is not given.
* `height`: Image height. Must be a multiple of 16.
* `width`: Image width. Must be a multiple of 16.
* `seed`: Random seed. Default is `None`, meaning fully random.
* `rand_device`: Device for generating random noise. Default is `"cpu"`. Setting it to `"cuda"` may lead to different results on different GPUs.
* `num_inference_steps`: Number of inference steps. Default is 30.
* `tiled`: Whether to enable tiled VAE inference. Default is `False`. Set to `True` to reduce VRAM use in VAE encoding/decoding. This causes small errors and slightly longer inference time.
* `tile_size`: Tile size for VAE encoding/decoding. Default is 128. Only works when `tiled=True`.
* `tile_stride`: Tile stride for VAE encoding/decoding. Default is 64. Only works when `tiled=True`. Must be less than or equal to `tile_size`.
* `progress_bar_cmd`: Progress bar display. Default is `tqdm.tqdm`. Set to `lambda x: x` to hide the progress bar.

</details>


## Model Training

The Qwen-Image series models are trained using a unified script [`./model_training/train.py`](./model_training/train.py).

<details>

<summary>Script Parameters</summary>

The script includes the following parameters:

* Dataset
  * `--dataset_base_path`: Root path of the dataset.
  * `--dataset_metadata_path`: Path to the dataset metadata file.
  * `--max_pixels`: Maximum pixel area. Default is 1024*1024. When dynamic resolution is enabled, any image with resolution higher than this will be resized down.
  * `--height`: Height of image or video. Leave `height` and `width` empty to enable dynamic resolution.
  * `--width`: Width of image or video. Leave `height` and `width` empty to enable dynamic resolution.
  * `--data_file_keys`: Data file keys in metadata. Separate with commas.
  * `--dataset_repeat`: Number of times the dataset repeats per epoch.
  * `--dataset_num_workers`: Number of workers for data loading.
* Model
  * `--model_paths`: Model paths to load. In JSON format.
  * `--model_id_with_origin_paths`: Model ID with original paths, e.g., Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors. Separate with commas.
  * `--tokenizer_path`: Tokenizer path. Leave empty to auto-download.
  * `--processor_path`: Path to the processor of Qwen-Image-Edit. Leave empty to auto-download.
* Training
  * `--learning_rate`: Learning rate.
  * `--weight_decay`: Weight decay.
  * `--num_epochs`: Number of epochs.
  * `--output_path`: Save path.
  * `--remove_prefix_in_ckpt`: Remove prefix in checkpoint.
  * `--save_steps`: Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.
  * `--find_unused_parameters`: Whether to find unused parameters in DDP.
* Trainable Modules
  * `--trainable_models`: Models to train, e.g., dit, vae, text_encoder.
  * `--lora_base_model`: Which model to add LoRA to.
  * `--lora_target_modules`: Which layers to add LoRA to.
  * `--lora_rank`: Rank of LoRA.
  * `--lora_checkpoint`: Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.
* Extra Model Inputs
  * `--extra_inputs`: Extra model inputs, separated by commas.
* VRAM Management
  * `--use_gradient_checkpointing`: Whether to enable gradient checkpointing.
  * `--use_gradient_checkpointing_offload`: Whether to offload gradient checkpointing to CPU memory.
  * `--gradient_accumulation_steps`: Number of gradient accumulation steps.

In addition, the training framework is built on [`accelerate`](https://huggingface.co/docs/accelerate/index). Run `accelerate config` before training to set GPU-related settings. For some training tasks (e.g., full training of 20B model), we provide suggested `accelerate` config files. Check the corresponding training script for details.

</details>


<details>

<summary>Step 1: Prepare Dataset</summary>

The dataset contains a set of files. We suggest organizing your dataset like this:

```
data/example_image_dataset/
├── metadata.csv
├── image1.jpg
└── image2.jpg
```

Here, `image1.jpg` and `image2.jpg` are image files for training, and `metadata.csv` is a metadata list, for example:

```
image,prompt
image1.jpg,"a cat is sleeping"
image2.jpg,"a dog is running"
```

We have built a sample image dataset for your testing. Use the following command to download it:

```shell
modelscope download --dataset DiffSynth-Studio/example_image_dataset --local_dir ./data/example_image_dataset
```

The dataset supports multiple image formats: `"jpg", "jpeg", "png", "webp"`.

Image size can be controlled by script parameters `--height` and `--width`. When `--height` and `--width` are empty, dynamic resolution is enabled. Images will be trained using their original sizes.

**We strongly recommend using fixed resolution for training, as multi-GPU training may have load balancing issues with dynamic resolution.**

</details>


<details>

<summary>Step 2: Load Model</summary>

Similar to model loading during inference, you can set the model to load directly by model ID. For example, during inference we load the model like this:

```python
model_configs=[
    ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
    ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
    ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
]
```

Then during training, use the following parameter to load the same models:

```shell
--model_id_with_origin_paths "Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors"
```

If you want to load the model from local files, for example, during inference:

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

Then during training, set it as:

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

<summary>Step 3: Set Trainable Modules</summary>

The training framework supports training base models or LoRA models. Here are some examples:

* Full training of DiT part: `--trainable_models dit`
* Train LoRA on DiT part: `--lora_base_model dit --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" --lora_rank 32`

Also, since the training script loads multiple modules (text encoder, dit, vae), you need to remove prefixes when saving model files. For example, when fully training the DiT part or training LoRA on DiT, set `--remove_prefix_in_ckpt pipe.dit.`

</details>


<details>

<summary>Step 4: Start Training</summary>

We have written training commands for each model. Please refer to the table at the start of this document.

</details>
