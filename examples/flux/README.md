# FLUX

[切换到中文](./README_zh.md)

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

You can quickly load the [black-forest-labs/FLUX.1-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-dev  ) model and run inference by executing the code below.

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

|Model ID|Extra Args|Inference|Low VRAM Inference|Full Training|Validation after Full Training|LoRA Training|Validation after LoRA Training|
|-|-|-|-|-|-|-|-|
|[FLUX.1-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-dev)||[code](./model_inference/FLUX.1-dev.py)|[code](./model_inference_low_vram/FLUX.1-dev.py)|[code](./model_training/full/FLUX.1-dev.sh)|[code](./model_training/validate_full/FLUX.1-dev.py)|[code](./model_training/lora/FLUX.1-dev.sh)|[code](./model_training/validate_lora/FLUX.1-dev.py)|
|[FLUX.1-Kera-dev](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-Krea-dev)||[code](./model_inference/FLUX.1-Kera-dev.py)|[code](./model_inference_low_vram/FLUX.1-Kera-dev.py)|[code](./model_training/full/FLUX.1-Kera-dev.sh)|[code](./model_training/validate_full/FLUX.1-Kera-dev.py)|[code](./model_training/lora/FLUX.1-Kera-dev.sh)|[code](./model_training/validate_lora/FLUX.1-Kera-dev.py)|
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

## Model Inference

The following sections will help you understand our features and write inference code.

<details>

<summary>Load Model</summary>

The model is loaded using `from_pretrained`:

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

Here, `torch_dtype` and `device` set the computation precision and device. The `model_configs` can be used in different ways to specify model paths:

* Download the model from [ModelScope](https://modelscope.cn/  ) and load it. In this case, fill in `model_id` and `origin_file_pattern`, for example:

```python
ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors")
```

* Load the model from a local file path. In this case, fill in `path`, for example:

```python
ModelConfig(path="models/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors")
```

For a single model that loads from multiple files, use a list, for example:

```python
ModelConfig(path=[
    "models/xxx/diffusion_pytorch_model-00001-of-00003.safetensors",
    "models/xxx/diffusion_pytorch_model-00002-of-00003.safetensors",
    "models/xxx/diffusion_pytorch_model-00003-of-00003.safetensors",
])
```

The `ModelConfig` method also provides extra arguments to control model loading behavior:

* `local_model_path`: Path to save downloaded models. Default is `"./models"`.
* `skip_download`: Whether to skip downloading. Default is `False`. If your network cannot access [ModelScope](https://modelscope.cn/  ), download the required files manually and set this to `True`.

</details>


<details>

<summary>VRAM Management</summary>

DiffSynth-Studio provides fine-grained VRAM management for the FLUX model. This allows the model to run on devices with low VRAM. You can enable the offload feature using the code below. It moves some modules to CPU memory when GPU memory is limited.

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

FP8 quantization is also supported:

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

You can use FP8 quantization and offload at the same time:

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

After enabling VRAM management, the framework will automatically decide the VRAM strategy based on available GPU memory. For most FLUX models, inference can run with as little as 8GB of VRAM. The `enable_vram_management` function has the following parameters to manually control the VRAM strategy:

* `vram_limit`: VRAM usage limit in GB. By default, it uses all free VRAM on the device. Note that this is not an absolute limit. If the set VRAM is not enough but more VRAM is actually available, the model will run with minimal VRAM usage. Setting it to 0 achieves the theoretical minimum VRAM usage.
* `vram_buffer`: VRAM buffer size in GB. Default is 0.5GB. A buffer is needed because larger neural network layers may use more VRAM than expected during loading. The optimal value is the VRAM used by the largest layer in the model.
* `num_persistent_param_in_dit`: Number of parameters in the DiT model that stay in VRAM. Default is no limit. We plan to remove this parameter in the future. Do not rely on it.

</details>


<details>

<summary>Inference Acceleration</summary>

* TeaCache: Acceleration technique [TeaCache](https://github.com/ali-vilab/TeaCache  ). Please refer to the [example code](./acceleration/teacache.py).

</details>

<details>

<summary>Input Parameters</summary>

The pipeline supports the following input parameters during inference:

* `prompt`: Text prompt describing what should appear in the image.
* `negative_prompt`: Negative prompt describing what should not appear in the image. Default is `""`.
* `cfg_scale`: Parameter for classifier-free guidance. Default is 1. Takes effect when set to a value greater than 1.
* `embedded_guidance`: Built-in guidance parameter for FLUX-dev. Default is 3.5.
* `t5_sequence_length`: Sequence length of text embeddings from the T5 model. Default is 512.
* `input_image`: Input image used for image-to-image generation. Used together with `denoising_strength`.
* `denoising_strength`: Denoising strength, range from 0 to 1. Default is 1. When close to 0, the output image is similar to the input. When close to 1, the output differs more from the input. Do not set it to values other than 1 if `input_image` is not provided.
* `height`: Image height. Must be a multiple of 16.
* `width`: Image width. Must be a multiple of 16.
* `seed`: Random seed. Default is `None`, meaning fully random.
* `rand_device`: Device for generating random Gaussian noise. Default is `"cpu"`. Setting it to `"cuda"` may lead to different results on different GPUs.
* `sigma_shift`: Parameter from Rectified Flow theory. Default is 3. A larger value means the model spends more steps at the start of denoising. Increasing this can improve image quality, but may cause differences between generated images and training data due to inconsistency with training.
* `num_inference_steps`: Number of inference steps. Default is 30.
* `kontext_images`: Input images for the Kontext model.
* `controlnet_inputs`: Inputs for the ControlNet model.
* `ipadapter_images`: Input images for the IP-Adapter model.
* `ipadapter_scale`: Control strength for the IP-Adapter model.
* `eligen_entity_prompts`: Local prompts for the EliGen model.
* `eligen_entity_masks`: Mask regions for local prompts in the EliGen model. Matches one-to-one with `eligen_entity_prompts`.
* `eligen_enable_on_negative`: Whether to enable EliGen on the negative prompt side. Only works when `cfg_scale > 1`.
* `eligen_enable_inpaint`: Whether to enable EliGen for local inpainting.
* `infinityou_id_image`: Face image for the InfiniteYou model.
* `infinityou_guidance`: Control strength for the InfiniteYou model.
* `flex_inpaint_image`: Image for FLEX model's inpainting.
* `flex_inpaint_mask`: Mask region for FLEX model's inpainting.
* `flex_control_image`: Image for FLEX model's structural control.
* `flex_control_strength`: Strength for FLEX model's structural control.
* `flex_control_stop`: End point for FLEX model's structural control. 1 means enabled throughout, 0.5 means enabled in the first half, 0 means disabled.
* `step1x_reference_image`: Input image for Step1x-Edit model's image editing.
* `lora_encoder_inputs`: Inputs for LoRA encoder. Can be ModelConfig or local path.
* `lora_encoder_scale`: Activation strength for LoRA encoder. Default is 1. Smaller values mean weaker LoRA activation.
* `tea_cache_l1_thresh`: Threshold for TeaCache. Larger values mean faster speed but lower image quality. Note that after enabling TeaCache, inference speed is not uniform, so the remaining time shown in the progress bar will be inaccurate.
* `tiled`: Whether to enable tiled VAE inference. Default is `False`. Setting to `True` reduces VRAM usage during VAE encoding/decoding, with slight error and slightly longer inference time.
* `tile_size`: Tile size during VAE encoding/decoding. Default is 128. Only takes effect when `tiled=True`.
* `tile_stride`: Tile stride during VAE encoding/decoding. Default is 64. Only takes effect when `tiled=True`. Must be less than or equal to `tile_size`.
* `progress_bar_cmd`: Progress bar display. Default is `tqdm.tqdm`. Set to `lambda x:x` to disable the progress bar.

</details>


## Model Training

Training for the FLUX series models is done using a unified script [`./model_training/train.py`](./model_training/train.py).

<details>

<summary>Script Parameters</summary>

The script includes the following parameters:

* Dataset
  * `--dataset_base_path`: Root path of the dataset.
  * `--dataset_metadata_path`: Path to the dataset metadata file.
  * `--max_pixels`: Maximum pixel area. Default is 1024*1024. When dynamic resolution is enabled, any image with resolution higher than this will be downscaled.
  * `--height`: Height of the image or video. Leave `height` and `width` empty to enable dynamic resolution.
  * `--width`: Width of the image or video. Leave `height` and `width` empty to enable dynamic resolution.
  * `--data_file_keys`: Data file keys in the metadata. Separate with commas.
  * `--dataset_repeat`: Number of times the dataset repeats per epoch.
* Model
  * `--model_paths`: Paths to load models. In JSON format.
  * `--model_id_with_origin_paths`: Model ID with original paths, e.g., black-forest-labs/FLUX.1-dev:flux1-dev.safetensors. Separate with commas.
* Training
  * `--learning_rate`: Learning rate.
  * `--num_epochs`: Number of epochs.
  * `--output_path`: Save path.
  * `--remove_prefix_in_ckpt`: Remove prefix in checkpoint.
* Trainable Modules
  * `--trainable_models`: Models that can be trained, e.g., dit, vae, text_encoder.
  * `--lora_base_model`: Which model to add LoRA to.
  * `--lora_target_modules`: Which layers to add LoRA to.
  * `--lora_rank`: Rank of LoRA.
* Extra Model Inputs
  * `--extra_inputs`: Extra model inputs, separated by commas.
* VRAM Management
  * `--use_gradient_checkpointing`: Whether to enable gradient checkpointing.
  * `--use_gradient_checkpointing_offload`: Whether to offload gradient checkpointing to CPU memory.
  * `--gradient_accumulation_steps`: Number of gradient accumulation steps.
* Others
  * `--align_to_opensource_format`: Whether to align the FLUX DiT LoRA format with the open-source version. Only works for LoRA training.

In addition, the training framework is built on [`accelerate`](https://huggingface.co/docs/accelerate/index  ). Run `accelerate config` before training to set GPU-related parameters. For some training scripts (e.g., full model training), we provide suggested `accelerate` config files. You can find them in the corresponding training scripts.

</details>


<details>

<summary>Step 1: Prepare Dataset</summary>

A dataset contains a series of files. We suggest organizing your dataset like this:

```
data/example_image_dataset/
├── metadata.csv
├── image1.jpg
└── image2.jpg
```

Here, `image1.jpg` and `image2.jpg` are training images, and `metadata.csv` is the metadata list, for example:

```
image,prompt
image1.jpg,"a cat is sleeping"
image2.jpg,"a dog is running"
```

We have built a sample image dataset to help you test. You can download it with the following command:

```shell
modelscope download --dataset DiffSynth-Studio/example_image_dataset --local_dir ./data/example_image_dataset
```

The dataset supports multiple image formats: `"jpg", "jpeg", "png", "webp"`.

Image size can be controlled by script arguments `--height` and `--width`. When `--height` and `--width` are left empty, dynamic resolution is enabled. The model will train using each image's actual width and height from the dataset.

**We strongly recommend using fixed resolution for training, because there can be load balancing issues in multi-GPU training.**

When the model needs extra inputs, for example, `kontext_images` required by controllable models like [`black-forest-labs/FLUX.1-Kontext-dev`](https://modelscope.cn/models/black-forest-labs/FLUX.1-Kontext-dev  ), add the corresponding column to your dataset, for example:

```
image,prompt,kontext_images
image1.jpg,"a cat is sleeping",image1_reference.jpg
```

If an extra input includes image files, you must specify the column name in the `--data_file_keys` argument. Add column names as needed, for example `--data_file_keys "image,kontext_images"`, and also enable `--extra_inputs "kontext_images"`.

</details>


<details>

<summary>Step 2: Load Model</summary>

Similar to model loading during inference, you can configure which models to load directly using model IDs. For example, during inference we load the model with this setting:

```python
model_configs=[
    ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
    ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
    ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
    ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
]
```

Then, during training, use the following parameter to load the same models:

```shell
--model_id_with_origin_paths "black-forest-labs/FLUX.1-dev:flux1-dev.safetensors,black-forest-labs/FLUX.1-dev:text_encoder/model.safetensors,black-forest-labs/FLUX.1-dev:text_encoder_2/,black-forest-labs/FLUX.1-dev:ae.safetensors"
```

If you want to load models from local files, for example, during inference:

```python
model_configs=[
    ModelConfig(path="models/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors"),
    ModelConfig(path="models/black-forest-labs/FLUX.1-dev/text_encoder/model.safetensors"),
    ModelConfig(path="models/black-forest-labs/FLUX.1-dev/text_encoder_2/"),
    ModelConfig(path="models/black-forest-labs/FLUX.1-dev/ae.safetensors"),
]
```

Then during training, set it as:

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

<summary>Step 3: Set Trainable Modules</summary>

The training framework supports training base models or LoRA models. Here are some examples:

* Full training of the DiT part: `--trainable_models dit`
* Training a LoRA model on the DiT part: `--lora_base_model dit --lora_target_modules "a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp" --lora_rank 32`

Also, because the training script loads multiple modules (text encoder, dit, vae), you need to remove prefixes when saving model files. For example, when fully training the DiT part or training a LoRA model on the DiT part, set `--remove_prefix_in_ckpt pipe.dit.`

</details>


<details>

<summary>Step 4: Start Training</summary>

We have written training commands for each model. Please refer to the table at the beginning of this document.

</details>
