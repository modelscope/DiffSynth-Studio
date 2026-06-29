# Boogu-Image

Boogu-Image supports text-to-image, image-to-image, and instruction-guided image editing.

## Installation

Before performing model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information on installation, please refer to [Setup Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Running the following code will load the [Boogu/Boogu-Image-0.1-Base](https://modelscope.cn/models/Boogu/Boogu-Image-0.1-Base) model for inference. VRAM management is enabled, the framework automatically controls parameter loading based on available VRAM, requiring a minimum of 8GB VRAM.

```python
from diffsynth.pipelines.boogu_image import BooguImagePipeline, ModelConfig
import torch


vram_config = {
    "offload_dtype": torch.float8_e4m3fn,
    "offload_device": "cpu",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = BooguImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Boogu/Boogu-Image-0.1-Base", origin_file_pattern="transformer/*.safetensors", **vram_config),
        ModelConfig(model_id="Boogu/Boogu-Image-0.1-Base", origin_file_pattern="mllm/*.safetensors", **vram_config),
        ModelConfig(model_id="Boogu/Boogu-Image-0.1-Base", origin_file_pattern="vae/*.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="Boogu/Boogu-Image-0.1-Base", origin_file_pattern="mllm/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

output = pipe(
    prompt="a cat",
    negative_prompt="",
    height=1024,
    width=1024,
    seed=42,
    num_inference_steps=50,
    cfg_scale=4.0,
)
output.save("image_Boogu-Image-0.1-Base.jpg")
```

## Model Overview

|Model ID|Inference|Low VRAM Inference|Full Training|Full Training Validation|LoRA Training|LoRA Training Validation|
|-|-|-|-|-|-|-|
|[Boogu/Boogu-Image-0.1-Base](https://modelscope.cn/models/Boogu/Boogu-Image-0.1-Base)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_inference/Boogu-Image-0.1-Base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_inference_low_vram/Boogu-Image-0.1-Base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_training/full/Boogu-Image-0.1-Base.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_training/validate_full/Boogu-Image-0.1-Base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_training/lora/Boogu-Image-0.1-Base.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_training/validate_lora/Boogu-Image-0.1-Base.py)|
|[Boogu/Boogu-Image-0.1-Turbo](https://modelscope.cn/models/Boogu/Boogu-Image-0.1-Turbo)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_inference/Boogu-Image-0.1-Turbo.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_inference_low_vram/Boogu-Image-0.1-Turbo.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_training/full/Boogu-Image-0.1-Turbo.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_training/validate_full/Boogu-Image-0.1-Turbo.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_training/lora/Boogu-Image-0.1-Turbo.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_training/validate_lora/Boogu-Image-0.1-Turbo.py)|
|[Boogu/Boogu-Image-0.1-Edit](https://modelscope.cn/models/Boogu/Boogu-Image-0.1-Edit)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_inference/Boogu-Image-0.1-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_inference_low_vram/Boogu-Image-0.1-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_training/full/Boogu-Image-0.1-Edit.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_training/validate_full/Boogu-Image-0.1-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_training/lora/Boogu-Image-0.1-Edit.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/boogu_image/model_training/validate_lora/Boogu-Image-0.1-Edit.py)|

## Model Inference

The model is loaded via `BooguImagePipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details.

The input parameters for `BooguImagePipeline` inference include:

* `prompt`: Text prompt describing the desired content or editing instruction.
* `negative_prompt`: Negative prompt specifying what should not appear in the result, defaults to empty string.
* `cfg_scale`: Classifier-free guidance scale factor, defaults to 4.0. Higher values make the output more closely follow the prompt.
* `input_image`: Input image for image-to-image (img2img). When provided, the input image is noised and denoised according to `denoising_strength`.
* `edit_image`: Image to be edited for instruction-guided editing. When provided, the model modifies the image according to the `prompt` instruction.
* `height`: Height of the output image, defaults to 1024. Must be divisible by 16.
* `width`: Width of the output image, defaults to 1024. Must be divisible by 16.
* `seed`: Random seed for reproducibility. Set to `None` for random seed.
* `denoising_strength`: Denoising strength controlling how much the input image is repainted, defaults to 1.0. Only effective when `input_image` is provided.
* `sigmas`: Custom sigma scheduling sequence to override the default scheduling strategy. Required for Turbo models.
* `num_inference_steps`: Number of inference steps, defaults to 20. More steps typically yield better quality.
* `max_sequence_length`: Maximum sequence length for the text encoder, defaults to 1280.
* `max_input_image_pixels`: Maximum pixel area for input images, defaults to 4194304. Images larger than this will be scaled down.
* `max_input_image_side_length`: Maximum side length for input images, defaults to 4096.
* `max_vlm_input_pil_pixels`: Maximum pixel area for VLM input images, defaults to 147456. Only effective in image editing mode.
* `max_vlm_input_pil_side_length`: Maximum side length for VLM input images, defaults to 768. Only effective in image editing mode.
* `rand_device`: Device for generating initial noise, defaults to "cpu".
* `progress_bar_cmd`: Progress bar display mode, defaults to tqdm.

When running low on VRAM, please refer to [VRAM Management](../Pipeline_Usage/VRAM_management.md) to enable VRAM management features.

## Model Training

Models in the boogu_image series are trained uniformly via `examples/boogu_image/model_training/train.py`. The script parameters include:

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
* Boogu-Image Specific Parameters
    * `--processor_path`: Path to the processor for processing text and image encoder inputs.
    * `--initialize_model_on_cpu`: Whether to initialize models on CPU. By default, models are initialized on the accelerator device.

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

We provide recommended training scripts for each model, please refer to the table in "Model Overview" above. For guidance on writing model training scripts, see [Model Training](../Pipeline_Usage/Model_Training.md); for more advanced training algorithms, see [Training Framework Overview](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/en/Training/).
