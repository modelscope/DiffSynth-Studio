# FLUX

[Switch to Chinese](./README_zh.md)

FLUX is a series of image generation models open-sourced by Black-Forest-Labs.

**DiffSynth-Studio has introduced a new inference and training framework. If you need to use the old version, please click [here](https://github.com/modelscope/DiffSynth-Studio/tree/3edf3583b1f08944cee837b94d9f84d669c2729c).**

## Installation

Before using these models, please install DiffSynth-Studio from source code:

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git  
cd DiffSynth-Studio
pip install -e .
```

## Quick Start

You can quickly load the FLUX.1-dev model and perform inference by running the following code:

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

## Model Overview

**Support for the new framework of the FLUX series models is under active development. Stay tuned!**

| Model ID | Additional Parameters | Inference | Full Training | Validation After Full Training | LoRA Training | Validation After LoRA Training |
|---------|------------------------|-----------|---------------|-------------------------------|---------------|--------------------------------|
| [black-forest-labs/FLUX.1-dev](https://modelscope.cn/models/black-forest-labs/FLUX.1-dev) |  | [code](./model_inference/FLUX.1-dev.py) | [code](./model_training/full/FLUX.1-dev.sh) | [code](./model_training/validate_full/FLUX.1-dev.py) | [code](./model_training/lora/FLUX.1-dev.sh) | [code](./model_training/validate_lora/FLUX.1-dev.py) |
| [black-forest-labs/FLUX.1-Kontext-dev](https://modelscope.cn/models/black-forest-labs/FLUX.1-Kontext-dev) | `kontext_images` | [code](./model_inference/FLUX.1-Kontext-dev.py) |  |  | [code](./model_training/lora/FLUX.1-Kontext-dev.sh) | [code](./model_training/validate_lora/FLUX.1-Kontext-dev.py) |

## Model Inference

The following sections will help you understand our features and write inference code.

<details>

<summary>Loading Models</summary>

Models are loaded using `from_pretrained`:

```python
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

Here, `torch_dtype` and `device` refer to the computation precision and device, respectively. The `model_configs` can be configured in various ways to specify model paths:

* Download the model from [ModelScope Community](https://modelscope.cn/) and load it. In this case, provide `model_id` and `origin_file_pattern`, for example:

```python
ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors")
```

* Load the model from a local file path. In this case, provide the `path`, for example:

```python
ModelConfig(path="models/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors")
```

For models that consist of multiple files, use a list as follows:

```python
ModelConfig(path=[
    "models/xxx/diffusion_pytorch_model-00001-of-00003.safetensors",
    "models/xxx/diffusion_pytorch_model-00002-of-00003.safetensors",
    "models/xxx/diffusion_pytorch_model-00003-of-00003.safetensors",
])
```

The `from_pretrained` method also provides additional parameters to control model loading behavior:

* `local_model_path`: Path for saving downloaded models. The default is `"./models"`.
* `skip_download`: Whether to skip downloading models. The default is `False`. If your network cannot access [ModelScope Community](https://modelscope.cn/), manually download the required files and set this to `True`.

</details>


<details>

<summary>VRAM Management</summary>

DiffSynth-Studio provides fine-grained VRAM management for FLUX models, enabling inference on devices with limited VRAM. You can enable offloading functionality via the following code, which moves certain modules to system memory on devices with limited GPU memory.

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

The `enable_vram_management` function provides the following parameters to control VRAM usage:

* `vram_limit`: VRAM usage limit in GB. By default, it uses the remaining VRAM available on the device. Note that this is not an absolute limit; if the set VRAM is insufficient but more VRAM is actually available, the model will run with minimal VRAM consumption. Setting it to 0 achieves the theoretical minimum VRAM usage.
* `vram_buffer`: VRAM buffer size in GB. The default is 0.5GB. Since some large neural network layers may consume extra VRAM during onload phases, a VRAM buffer is necessary. Ideally, the optimal value should match the VRAM occupied by the largest layer in the model.
* `num_persistent_param_in_dit`: Number of persistent parameters in the DiT model (default: no limit). We plan to remove this parameter in the future, so please avoid relying on it.

</details>

<details>

<summary>Inference Acceleration</summary>

* TeaCache: Acceleration technique [TeaCache](https://github.com/ali-vilab/TeaCache), please refer to the [sample code](./acceleration/teacache.py).

</details>

<details>

<summary>Input Parameters</summary>

The pipeline accepts the following input parameters during inference:

* `prompt`: Prompt describing what should appear in the image.
* `negative_prompt`: Negative prompt describing what should **not** appear in the image. Default is `""`.
* `cfg_scale`: Classifier-free guidance scale. Default is 1. It becomes effective when set to a value greater than 1.
* `embedded_guidance`: Embedded guidance parameter for FLUX-dev. Default is 3.5.
* `t5_sequence_length`: Sequence length of T5 text embeddings. Default is 512.
* `input_image`: Input image used for image-to-image generation. This works together with `denoising_strength`.
* `denoising_strength`: Denoising strength, ranging from 0 to 1. Default is 1. When close to 0, the generated image will be similar to the input image; when close to 1, the generated image will differ significantly from the input. Do not set this to a non-1 value if no `input_image` is provided.
* `height`: Height of the generated image. Must be a multiple of 16.
* `width`: Width of the generated image. Must be a multiple of 16.
* `seed`: Random seed. Default is `None`, meaning completely random.
* `rand_device`: Device for generating random Gaussian noise. Default is `"cpu"`. Setting it to `"cuda"` may lead to different results across GPUs.
* `sigma_shift`: Parameter from Rectified Flow theory. Default is 3. A larger value increases the number of steps spent at the beginning of denoising and can improve image quality. However, it may cause inconsistencies between the generation process and training data.
* `num_inference_steps`: Number of inference steps. Default is 30.
* `kontext_images`: Input images for the Kontext model.
* `controlnet_inputs`: Inputs for the ControlNet model.
* `ipadapter_images`: Input images for the IP-Adapter model.
* `ipadapter_scale`: Control strength of the IP-Adapter model.

</details>

## Model Training

FLUX series models are trained using a unified script [`./model_training/train.py`](./model_training/train.py).

<details>

<summary>Script Parameters</summary>

The script supports the following parameters:

* Dataset
  * `--dataset_base_path`: Root path to the dataset.
  * `--dataset_metadata_path`: Path to the metadata file of the dataset.
  * `--height`: Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.
  * `--width`: Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.
  * `--data_file_keys`: Keys in metadata for data files. Comma-separated.
  * `--dataset_repeat`: Number of times the dataset repeats per epoch.
* Models
  * `--model_paths`: Paths to load models. JSON format.
  * `--model_id_with_origin_paths`: Model IDs with original paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.
* Training
  * `--learning_rate`: Learning rate.
  * `--num_epochs`: Number of training epochs.
  * `--output_path`: Output path for saving checkpoints.
  * `--remove_prefix_in_ckpt`: Remove prefix in checkpoint filenames.
* Trainable Modules
  * `--trainable_models`: Models that can be trained, e.g., dit, vae, text_encoder.
  * `--lora_base_model`: Which base model to apply LoRA on.
  * `--lora_target_modules`: Which layers to apply LoRA on.
  * `--lora_rank`: Rank of LoRA.
* Extra Inputs
  * `--extra_inputs`: Additional model inputs. Comma-separated.
* VRAM Management
  * `use_gradient_checkpointing`: Whether to use gradient checkpointing.
  * `--use_gradient_checkpointing_offload`: Whether to offload gradient checkpointing to CPU memory.
  * `gradient_accumulation_steps`: Number of steps for gradient accumulation.
* Miscellaneous
  * `--align_to_opensource_format`: Whether to align the FLUX DiT LoRA format with the open-source version. Only applicable to LoRA training for FLUX.1-dev and FLUX.1-Kontext-dev.

</details>

<details>

<summary>Step 1: Prepare Dataset</summary>

The dataset contains a series of files. We recommend organizing your dataset files as follows:

```
data/example_video_dataset/
├── metadata.csv
├── image1.jpg
└── image2.jpg
```

Here, `image1.jpg`, `image2.jpg` are training video/image data, and `metadata.csv` is the metadata list, for example:

```
video,prompt
image1.jpg,"a cat is sleeping"
image2.jpg,"a dog is running"
```

We have built a sample image dataset to help you test more conveniently. You can download this dataset using the following command:

```shell
modelscope download --dataset DiffSynth-Studio/example_image_dataset --local_dir ./data/example_image_dataset
```

The dataset supports multiple image formats: `"jpg", "jpeg", "png", "webp"`.

The image resolution can be controlled via script parameters `--height` and `--width`. When both `--height` and `--width` are left empty, dynamic resolution will be enabled, allowing training with the actual width and height of each video or image in the dataset.

**We strongly recommend using fixed-resolution training, as there may be load-balancing issues in multi-GPU training with dynamic resolution.**

When the model requires additional inputs—for instance, `kontext_images` required by the controllable model [`black-forest-labs/FLUX.1-Kontext-dev`](https://modelscope.cn/models/black-forest-labs/FLUX.1-Kontext-dev)—please add corresponding columns in the dataset, for example:

```
video,prompt,kontext_images
image1.jpg,"a cat is sleeping",image1_reference.jpg
```

If additional inputs include video or image files, you need to specify the column names to parse using the `--data_file_keys` parameter. You can add more column names accordingly, e.g., `--data_file_keys "image,kontext_images"`.

</details>

<details>

<summary>Step 2: Load Model</summary>

Similar to the model loading logic during inference, you can directly configure the model to be loaded using its model ID. For example, during inference we load the model with the following configuration:

```python
model_configs=[
    ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
    ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
    ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
    ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
]
```

Then during training, simply provide the following parameter to load the corresponding model:

```shell
--model_id_with_origin_paths "black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/,black-forest-labs/FLUX.1-dev:ae.safetensors"
```

If you prefer to load the model from local files, as in the inference example:

```python
model_configs=[
    ModelConfig(path="models/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors"),
    ModelConfig(path="models/black-forest-labs/FLUX.1-dev/text_encoder/model.safetensors"),
    ModelConfig(path="models/black-forest-labs/FLUX.1-dev/text_encoder_2/"),
    ModelConfig(path="models/black-forest-labs/FLUX.1-dev/ae.safetensors"),
]
```

Then during training, set it up as follows:

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

<summary>Step 3: Configure Trainable Modules</summary>

The training framework supports both full-model training and LoRA-based fine-tuning. Below are some examples:

* Full training of the DiT module: `--trainable_models dit`
* Training a LoRA model on the DiT module: `--lora_base_model dit --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" --lora_rank 32`

Additionally, since the training script loads multiple modules (text encoder, DiT, VAE), you need to remove prefixes when saving the model files. For example, when performing full DiT training or LoRA training on the DiT module, please set `--remove_prefix_in_ckpt pipe.dit.`

</details>

<details>

<summary>Step 4: Launch the Training Script</summary>

We have written specific training commands for each model. Please refer to the table at the beginning of this document for details.

</details>
