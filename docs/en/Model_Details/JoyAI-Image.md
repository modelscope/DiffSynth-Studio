# JoyAI-Image

JoyAI-Image is a unified multi-modal foundation model open-sourced by JD.com, supporting image understanding, text-to-image generation, and instruction-guided image editing.

## Installation

Before performing model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information on installation, please refer to [Setup Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Running the following code will load the [jd-opensource/JoyAI-Image-Edit](https://modelscope.cn/models/jd-opensource/JoyAI-Image-Edit) model for inference. VRAM management is enabled, the framework automatically controls parameter loading based on available VRAM, requiring a minimum of 4GB VRAM.

```python
from diffsynth.pipelines.joyai_image import JoyAIImagePipeline, ModelConfig
import torch
from PIL import Image
from modelscope import dataset_snapshot_download

# Download dataset
dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/diffsynth_example_dataset",
    local_dir="data/diffsynth_example_dataset",
    allow_file_pattern="joyai_image/JoyAI-Image-Edit/*"
)

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

pipe = JoyAIImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="jd-opensource/JoyAI-Image-Edit", origin_file_pattern="transformer/transformer.pth", **vram_config),
        ModelConfig(model_id="jd-opensource/JoyAI-Image-Edit", origin_file_pattern="JoyAI-Image-Und/model*.safetensors", **vram_config),
        ModelConfig(model_id="jd-opensource/JoyAI-Image-Edit", origin_file_pattern="vae/Wan2.1_VAE.pth", **vram_config),
    ],
    processor_config=ModelConfig(model_id="jd-opensource/JoyAI-Image-Edit", origin_file_pattern="JoyAI-Image-Und/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

# Use first sample from dataset
dataset_base_path = "data/diffsynth_example_dataset/joyai_image/JoyAI-Image-Edit"
prompt = "将裙子改为粉色"
edit_image = Image.open(f"{dataset_base_path}/edit/image1.jpg").convert("RGB")

output = pipe(
    prompt=prompt,
    edit_image=edit_image,
    height=1024,
    width=1024,
    seed=0,
    num_inference_steps=30,
    cfg_scale=5.0,
)

output.save("output_joyai_edit_low_vram.png")
```

## Model Overview

|Model ID|Inference|Low VRAM Inference|Full Training|Full Training Validation|LoRA Training|LoRA Training Validation|
|-|-|-|-|-|-|-|
|[jd-opensource/JoyAI-Image-Edit](https://modelscope.cn/models/jd-opensource/JoyAI-Image-Edit)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/joyai_image/model_inference/JoyAI-Image-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/joyai_image/model_inference_low_vram/JoyAI-Image-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/joyai_image/model_training/full/JoyAI-Image-Edit.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/joyai_image/model_training/validate_full/JoyAI-Image-Edit.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/joyai_image/model_training/lora/JoyAI-Image-Edit.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/joyai_image/model_training/validate_lora/JoyAI-Image-Edit.py)|

## Model Inference

The model is loaded via `JoyAIImagePipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details.

The input parameters for `JoyAIImagePipeline` inference include:

* `prompt`: Text prompt describing the desired image editing effect.
* `negative_prompt`: Negative prompt specifying what should not appear in the result, defaults to empty string.
* `cfg_scale`: Classifier-free guidance scale factor, defaults to 5.0. Higher values make the output more closely follow the prompt.
* `edit_image`: Image to be edited.
* `denoising_strength`: Denoising strength controlling how much the input image is repainted, defaults to 1.0.
* `height`: Height of the output image, defaults to 1024. Must be divisible by 16.
* `width`: Width of the output image, defaults to 1024. Must be divisible by 16.
* `seed`: Random seed for reproducibility. Set to `None` for random seed.
* `max_sequence_length`: Maximum sequence length for the text encoder, defaults to 4096.
* `num_inference_steps`: Number of inference steps, defaults to 30. More steps typically yield better quality.
* `tiled`: Whether to enable tiling for reduced VRAM usage, defaults to False.
* `tile_size`: Tile size, defaults to (30, 52).
* `tile_stride`: Tile stride, defaults to (15, 26).
* `shift`: Shift parameter for the scheduler, controlling the Flow Match scheduling curve, defaults to 4.0.
* `progress_bar_cmd`: Progress bar display mode, defaults to tqdm.

## Model Training

Models in the joyai_image series are trained uniformly via `examples/joyai_image/model_training/train.py`. The script parameters include:

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
* JoyAI-Image Specific Parameters
    * `--processor_path`: Path to the processor for processing text and image encoder inputs.
    * `--initialize_model_on_cpu`: Whether to initialize models on CPU. By default, models are initialized on the accelerator device.

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

We provide recommended training scripts for each model, please refer to the table in "Model Overview" above. For guidance on writing model training scripts, see [Model Training](../Pipeline_Usage/Model_Training.md); for more advanced training algorithms, see [Training Framework Overview](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/en/Training/).
