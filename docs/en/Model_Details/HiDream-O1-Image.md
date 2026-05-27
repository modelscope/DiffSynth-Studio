# HiDream-O1-Image

HiDream-O1-Image is an image generation model open-sourced by HiDream.ai, based on the Pixel-Level Unified Transformer (UiT) architecture. This model unifies VAE, DiT, and TextEncoder within a single Qwen3VLModel, performing diffusion denoising directly in pixel patch space without requiring a separate VAE component.

## Installation

Before performing model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information on installation, please refer to [Setup Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Running the following code will quickly load the [HiDream-ai/HiDream-O1-Image](https://modelscope.cn/HiDream-ai/HiDream-O1-Image) model for inference. VRAM management is enabled, the framework automatically controls parameter loading based on available VRAM, requiring a minimum of 3GB VRAM.

```python
from diffsynth.pipelines.hidream_o1_image import HiDreamO1ImagePipeline
from diffsynth.core.loader.config import ModelConfig
import torch


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


pipe = HiDreamO1ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="HiDream-ai/HiDream-O1-Image", origin_file_pattern="model-*.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="HiDream-ai/HiDream-O1-Image", origin_file_pattern="./"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
image = pipe(
    prompt="medium shot, eye-level, front view. A woman is seated in an ornate bedroom, illuminated by candlelight, with a calm and composed expression. The subject is a young woman with fair skin, light brown hair styled in an updo with loose tendrils framing her face, and blue eyes. She wears a cream-colored satin robe with delicate floral embroidery and lace trim along the neckline. Her ears are adorned with pearl drop earrings. She is seated on a bed with a dark, intricately carved wooden headboard. To her left, a wooden nightstand holds three lit white candles and a candelabra with multiple lit candles in the background. The bed is covered with patterned pillows and a dark, textured blanket. The walls are paneled with dark wood and feature a large, ornate tapestry with muted earth tones. The lighting creates soft highlights on her face and robe, with warm shadows cast across the room.",
    negative_prompt=" ",
    cfg_scale=4.0,
    height=2048,
    width=2048,
    seed=42,
    num_inference_steps=50,
)
image.save("image.jpg")
```

## Model Overview

|Model ID|Inference|Low VRAM Inference|Full Training|Full Training Validation|LoRA Training|LoRA Training Validation|
|-|-|-|-|-|-|-|
|[HiDream-ai/HiDream-O1-Image](https://modelscope.cn/HiDream-ai/HiDream-O1-Image)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hidream_o1_image/model_inference/HiDream-O1-Image.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hidream_o1_image/model_inference_low_vram/HiDream-O1-Image.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hidream_o1_image/model_training/full/HiDream-O1-Image.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hidream_o1_image/model_training/validate_full/HiDream-O1-Image.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hidream_o1_image/model_training/lora/HiDream-O1-Image.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hidream_o1_image/model_training/validate_lora/HiDream-O1-Image.py)|
|[HiDream-ai/HiDream-O1-Image-Dev](https://modelscope.cn/HiDream-ai/HiDream-O1-Image-Dev)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hidream_o1_image/model_inference/HiDream-O1-Image-Dev.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hidream_o1_image/model_inference_low_vram/HiDream-O1-Image-Dev.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hidream_o1_image/model_training/full/HiDream-O1-Image-Dev.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hidream_o1_image/model_training/validate_full/HiDream-O1-Image-Dev.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hidream_o1_image/model_training/lora/HiDream-O1-Image-Dev.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hidream_o1_image/model_training/validate_lora/HiDream-O1-Image-Dev.py)|

## Model Inference

The model is loaded via `HiDreamO1ImagePipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details.

The input parameters for `HiDreamO1ImagePipeline` inference include:

* `prompt`: Text prompt.
* `negative_prompt`: Negative prompt, defaults to `" "`.
* `cfg_scale`: Classifier-Free Guidance scale, defaults to 4.0. For the Dev model, it is recommended to set to 1.0.
* `height`: Output image height, defaults to 2048.
* `width`: Output image width, defaults to 2048.
* `seed`: Random seed, defaults to random.
* `rand_device`: Noise generation device, defaults to `"cpu"`.
* `num_inference_steps`: Number of inference steps, defaults to 50 for Full model and 28 for Dev model.
* `model_type`: Model type, `"full"` for Full model, `"dev"` for distilled Dev model.
* `shift`: Timestep shift parameter affecting sigma computation, defaults to 3.0.
* `noise_scale`: Noise scaling factor, defaults to 8.0. For the Dev model, it is recommended to set to 7.5.
* `edit_image`: List of reference images for image editing. Defaults to None (text-to-image mode).
* `keep_original_aspect`: Whether to preserve the original aspect ratio of reference images, defaults to True.

> **VRAM Note**: HiDream-O1-Image has a large parameter count (~8B). When generating 2048x2048 images, it is recommended to enable VRAM management (vram_config) or use the low VRAM inference scripts.

## Model Training

Models in the hidream_o1_image series are trained uniformly via `examples/hidream_o1_image/model_training/train.py`. The script parameters include:

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
* HiDream-O1-Image Specific Parameters
    * `--processor_config`: Path to the processor configuration file, used for loading AutoProcessor for text tokenization.
    * `--noise_scale`: Noise scaling factor, defaults to 8.0.
    * `--initialize_model_on_cpu`: Whether to initialize the model on CPU, which can help reduce peak GPU VRAM usage.

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

We provide recommended training scripts for each model, please refer to the table in "Model Overview" above. For guidance on writing model training scripts, see [Model Training](../Pipeline_Usage/Model_Training.md); for more advanced training algorithms, see [Training Framework Overview](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/en/Training/).
