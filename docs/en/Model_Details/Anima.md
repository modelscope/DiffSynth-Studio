# Anima

Anima is an image generation model trained and open-sourced by CircleStone Labs and Comfy Org.

## Installation

Before using this project for model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more installation information, please refer to [Install Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

The following code demonstrates how to quickly load the [circlestone-labs/Anima](https://www.modelscope.cn/models/circlestone-labs/Anima) model for inference. VRAM management is enabled by default, allowing the framework to automatically control model parameter loading based on available VRAM. Minimum 8GB VRAM required.

```python
from diffsynth.pipelines.anima_image import AnimaImagePipeline, ModelConfig
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
pipe = AnimaImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="circlestone-labs/Anima", origin_file_pattern="split_files/diffusion_models/anima-preview.safetensors", **vram_config),
        ModelConfig(model_id="circlestone-labs/Anima", origin_file_pattern="split_files/text_encoders/qwen_3_06b_base.safetensors", **vram_config),
        ModelConfig(model_id="circlestone-labs/Anima", origin_file_pattern="split_files/vae/qwen_image_vae.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen3-0.6B", origin_file_pattern="./"),
    tokenizer_t5xxl_config=ModelConfig(model_id="stabilityai/stable-diffusion-3.5-large", origin_file_pattern="tokenizer_3/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
prompt = "Masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait."
negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"
image = pipe(prompt, seed=0, num_inference_steps=50)
image.save("image.jpg")
```

## Model Overview

|Model ID|Inference|Low VRAM Inference|Full Training|Validation after Full Training|LoRA Training|Validation after LoRA Training|
|-|-|-|-|-|-|-|
|[circlestone-labs/Anima](https://www.modelscope.cn/models/circlestone-labs/Anima)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/anima/model_inference/anima-preview.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/anima/model_inference_low_vram/anima-preview.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/anima/model_training/full/anima-preview.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/anima/model_training/validate_full/anima-preview.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/anima/model_training/lora/anima-preview.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/anima/model_training/validate_lora/anima-preview.py)|

Special training scripts:

* Differential LoRA Training: [doc](../Training/Differential_LoRA.md)
* FP8 Precision Training: [doc](../Training/FP8_Precision.md)
* Two-Stage Split Training: [doc](../Training/Split_Training.md)
* End-to-End Direct Distillation: [doc](../Training/Direct_Distill.md)

## Model Inference

Models are loaded through `AnimaImagePipeline.from_pretrained`, see [Model Inference](../Pipeline_Usage/Model_Inference.md#loading-models) for details.

Input parameters for `AnimaImagePipeline` inference include:

* `prompt`: Text description of the desired image content.
* `negative_prompt`: Content to exclude from the generated image (default: `""`).
* `cfg_scale`: Classifier-free guidance parameter (default: 4.0).
* `input_image`: Input image for image-to-image generation (default: `None`).
* `denoising_strength`: Controls similarity to input image (default: 1.0).
* `height`: Image height (must be multiple of 16, default: 1024).
* `width`: Image width (must be multiple of 16, default: 1024).
* `seed`: Random seed (default: `None`).
* `rand_device`: Device for random noise generation (default: `"cpu"`).
* `num_inference_steps`: Inference steps (default: 30).
* `sigma_shift`: Scheduler sigma offset (default: `None`).
* `progress_bar_cmd`: Progress bar implementation (default: `tqdm.tqdm`).

For VRAM constraints, enable [VRAM Management](../Pipeline_Usage/VRAM_management.md). Recommended low-VRAM configurations are provided in the "Model Overview" table above.

## Model Training

Anima models are trained through [`examples/anima/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/anima/model_training/train.py) with parameters including:

* General Training Parameters
    * Dataset Configuration
        * `--dataset_base_path`: Dataset root directory.
        * `--dataset_metadata_path`: Metadata file path.
        * `--dataset_repeat`: Dataset repetition per epoch.
        * `--dataset_num_workers`: Dataloader worker count.
        * `--data_file_keys`: Metadata fields to load (comma-separated).
    * Model Loading
        * `--model_paths`: Model paths (JSON format).
        * `--model_id_with_origin_paths`: Model IDs with origin paths (e.g., `"anima-team/anima-1B:text_encoder/*.safetensors"`).
        * `--extra_inputs`: Additional pipeline inputs (e.g., `controlnet_inputs` for ControlNet).
        * `--fp8_models`: FP8-formatted models (same format as `--model_paths`).
    * Training Configuration
        * `--learning_rate`: Learning rate.
        * `--num_epochs`: Training epochs.
        * `--trainable_models`: Trainable components (e.g., `dit`, `vae`, `text_encoder`).
        * `--find_unused_parameters`: Handle unused parameters in DDP training.
        * `--weight_decay`: Weight decay value.
        * `--task`: Training task (default: `sft`).
    * Output Configuration
        * `--output_path`: Model output directory.
        * `--remove_prefix_in_ckpt`: Remove state dict prefixes.
        * `--save_steps`: Model saving interval.
    * LoRA Configuration
        * `--lora_base_model`: Target model for LoRA.
        * `--lora_target_modules`: Target modules for LoRA.
        * `--lora_rank`: LoRA rank.
        * `--lora_checkpoint`: LoRA checkpoint path.
        * `--preset_lora_path`: Preloaded LoRA checkpoint path.
        * `--preset_lora_model`: Model to merge LoRA with (e.g., `dit`).
    * Gradient Configuration
        * `--use_gradient_checkpointing`: Enable gradient checkpointing.
        * `--use_gradient_checkpointing_offload`: Offload checkpointing to CPU.
        * `--gradient_accumulation_steps`: Gradient accumulation steps.
    * Image Resolution
        * `--height`: Image height (empty for dynamic resolution).
        * `--width`: Image width (empty for dynamic resolution).
        * `--max_pixels`: Maximum pixel area for dynamic resolution.
* Anima-Specific Parameters
    * `--tokenizer_path`: Tokenizer path for text-to-image models.
    * `--tokenizer_t5xxl_path`: T5-XXL tokenizer path.

We provide a sample image dataset for testing:

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

For training script details, refer to [Model Training](../Pipeline_Usage/Model_Training.md). For advanced training techniques, see [Training Framework Documentation](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/zh/Training/).