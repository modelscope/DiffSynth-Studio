# Stable Diffusion XL

Stable Diffusion XL (SDXL) is an open-source diffusion-based text-to-image generation model developed by Stability AI, supporting 1024x1024 resolution high-quality text-to-image generation with a dual text encoder (CLIP-L + CLIP-bigG) architecture.

## Installation

Before performing model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information on installation, please refer to [Setup Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Running the following code will quickly load the [stabilityai/stable-diffusion-xl-base-1.0](https://www.modelscope.cn/models/stabilityai/stable-diffusion-xl-base-1.0) model for inference. VRAM management is enabled, the framework automatically controls parameter loading based on available VRAM, requiring a minimum of 6GB VRAM.

```python
import torch
from diffsynth.core import ModelConfig
from diffsynth.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline

vram_config = {
    "offload_dtype": torch.float32,
    "offload_device": "cpu",
    "onload_dtype": torch.float32,
    "onload_device": "cpu",
    "preparing_dtype": torch.float32,
    "preparing_device": "cuda",
    "computation_dtype": torch.float32,
    "computation_device": "cuda",
}
pipe = StableDiffusionXLPipeline.from_pretrained(
    torch_dtype=torch.float32,
    model_configs=[
        ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="text_encoder/model.safetensors", **vram_config),
        ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="text_encoder_2/model.safetensors", **vram_config),
        ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="unet/diffusion_pytorch_model.safetensors", **vram_config),
        ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="tokenizer/"),
    tokenizer_2_config=ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="tokenizer_2/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

image = pipe(
    prompt="a photo of an astronaut riding a horse on mars",
    negative_prompt="",
    cfg_scale=5.0,
    height=1024,
    width=1024,
    seed=42,
    num_inference_steps=50,
)
image.save("image.jpg")
```

## Model Overview

|Model ID|Inference|Low VRAM Inference|Full Training|Full Training Validation|LoRA Training|LoRA Training Validation|
|-|-|-|-|-|-|-|
|[stabilityai/stable-diffusion-xl-base-1.0](https://www.modelscope.cn/models/stabilityai/stable-diffusion-xl-base-1.0)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/stable_diffusion_xl/model_inference/stable-diffusion-xl-base-1.0.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/stable_diffusion_xl/model_inference_low_vram/stable-diffusion-xl-base-1.0.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/stable_diffusion_xl/model_training/full/stable-diffusion-xl-base-1.0.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/stable_diffusion_xl/model_training/validate_full/stable-diffusion-xl-base-1.0.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/stable_diffusion_xl/model_training/lora/stable-diffusion-xl-base-1.0.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/stable_diffusion_xl/model_training/validate_lora/stable-diffusion-xl-base-1.0.py)|

## Model Inference

The model is loaded via `StableDiffusionXLPipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details.

The input parameters for `StableDiffusionXLPipeline` inference include:

* `prompt`: Text prompt.
* `negative_prompt`: Negative prompt, defaults to an empty string.
* `cfg_scale`: Classifier-Free Guidance scale factor, default 5.0.
* `height`: Output image height, default 1024.
* `width`: Output image width, default 1024.
* `seed`: Random seed, defaults to a random value if not set.
* `rand_device`: Noise generation device, defaults to "cpu".
* `num_inference_steps`: Number of inference steps, default 50.
* `guidance_rescale`: Guidance rescale factor, default 0.0.
* `progress_bar_cmd`: Progress bar callback function.

> `StableDiffusionXLPipeline` requires dual tokenizer configurations (`tokenizer_config` and `tokenizer_2_config`), corresponding to the CLIP-L and CLIP-bigG text encoders.

## Model Training

Models in the stable_diffusion_xl series are trained via `examples/stable_diffusion_xl/model_training/train.py`. The script parameters include:

* General Training Parameters
    * Dataset Configuration
        * `--dataset_base_path`: Root directory of the dataset.
        * `--dataset_metadata_path`: Path to the dataset metadata file.
        * `--dataset_repeat`: Number of dataset repeats per epoch.
        * `--dataset_num_workers`: Number of processes per DataLoader.
        * `--data_file_keys`: Field names to load from metadata, typically paths to image or video files, separated by `,`.
    * Model Loading Configuration
        * `--model_paths`: Paths to load models from, in JSON format.
        * `--model_id_with_origin_paths`: Model IDs with original paths, separated by commas.
        * `--extra_inputs`: Additional input parameters required by the model Pipeline, separated by `,`.
        * `--fp8_models`: Models to load in FP8 format, currently only supported for models whose parameters are not updated by gradients.
    * Basic Training Configuration
        * `--learning_rate`: Learning rate.
        * `--num_epochs`: Number of epochs.
        * `--trainable_models`: Trainable models, e.g., `dit`, `vae`, `text_encoder`.
        * `--find_unused_parameters`: Whether unused parameters exist in DDP training.
        * `--weight_decay`: Weight decay magnitude.
        * `--task`: Training task, defaults to `sft`.
    * Output Configuration
        * `--output_path`: Path to save the model.
        * `--remove_prefix_in_ckpt`: Remove prefix in the model's state dict.
        * `--save_steps`: Interval in training steps to save the model.
    * LoRA Configuration
        * `--lora_base_model`: Which model to add LoRA to.
        * `--lora_target_modules`: Which layers to add LoRA to.
        * `--lora_rank`: Rank of LoRA.
        * `--lora_checkpoint`: Path to LoRA checkpoint.
        * `--preset_lora_path`: Path to preset LoRA checkpoint for LoRA differential training.
        * `--preset_lora_model`: Which model to integrate preset LoRA into, e.g., `dit`.
    * Gradient Configuration
        * `--use_gradient_checkpointing`: Whether to enable gradient checkpointing.
        * `--use_gradient_checkpointing_offload`: Whether to offload gradient checkpointing to CPU memory.
        * `--gradient_accumulation_steps`: Number of gradient accumulation steps.
    * Resolution Configuration
        * `--height`: Height of the image/video. Leave empty to enable dynamic resolution.
        * `--width`: Width of the image/video. Leave empty to enable dynamic resolution.
        * `--max_pixels`: Maximum pixel area, images larger than this will be scaled down during dynamic resolution.
        * `--num_frames`: Number of frames for video (video generation models only).
* Stable Diffusion XL Specific Parameters
    * `--tokenizer_path`: Path to the first tokenizer.
    * `--tokenizer_2_path`: Path to the second tokenizer, defaults to `stabilityai/stable-diffusion-xl-base-1.0:tokenizer_2/`.

Example dataset download:

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "stable_diffusion_xl/*" --local_dir ./data/diffsynth_example_dataset
```

[stable-diffusion-xl-base-1.0 training scripts](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/stable_diffusion_xl/model_training/lora/stable-diffusion-xl-base-1.0.sh)

We provide recommended training scripts for each model, please refer to the table in "Model Overview" above. For guidance on writing model training scripts, see [Model Training](../Pipeline_Usage/Model_Training.md); for more advanced training algorithms, see [Training Framework Overview](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/en/Training/).
