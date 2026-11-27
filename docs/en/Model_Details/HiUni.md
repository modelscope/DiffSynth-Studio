# HiUni

HiUni is a unified multimodal model open-sourced by ERNIE Research. One checkpoint handles image understanding, text-to-image generation and instruction-based image editing. It is a small model that adds a lightweight pixel-space diffusion head to a vision-language backbone, so images are generated directly as pixels without a VAE or a separate text encoder.

## Installation

Before performing model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information on installation, please refer to [Setup Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Running the following code will load the [ernie-research/HiUni](https://modelscope.cn/models/ernie-research/HiUni) model for inference. VRAM management is enabled, the framework automatically controls parameter loading based on available VRAM, requiring a minimum of 12GB VRAM.

```python
from diffsynth.pipelines.hiuni import HiUniPipeline, ModelConfig
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

pipe = HiUniPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="ernie-research/HiUni", origin_file_pattern="model-*.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="ernie-research/HiUni", origin_file_pattern="tokenizer*.json"),
    processor_config=ModelConfig(model_id="ernie-research/HiUni", origin_file_pattern="preprocessor_config.json"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

prompt = "A cozy reading nook by a rain-streaked window, a ginger cat asleep on a knitted blanket, warm lamp light, film photography."
image = pipe(prompt, height=2048, width=2048, seed=42, num_inference_steps=25, cfg_scale=4.0, shift=4.0)
image.save("image_HiUni.jpg")
```

## Model Overview

|Model ID|Inference|Low VRAM Inference|Parallel Inference|Full Training|Full Training Validation|LoRA Training|LoRA Training Validation|
|-|-|-|-|-|-|-|-|
|[ernie-research/HiUni](https://modelscope.cn/models/ernie-research/HiUni)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hiuni/model_inference/HiUni.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hiuni/model_inference_low_vram/HiUni.py)|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hiuni/model_training/full/HiUni.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hiuni/model_training/validate_full/HiUni.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hiuni/model_training/lora/HiUni.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hiuni/model_training/validate_lora/HiUni.py)|

## Model Inference

The model is loaded via `HiUniPipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details. `tokenizer_config` and `processor_config` load the tokenizer and the image processor from the repository root and only download their JSON files.

The input parameters for `HiUniPipeline` inference include:

* `prompt`: Text prompt. Acts as the editing instruction in image editing mode.
* `cfg_scale`: Text guidance scale, defaults to 4.0.
* `height`: Output image height, defaults to None: 1024 for text-to-image, or the reference resolution when editing. Must be a multiple of 32.
* `width`: Output image width, defaults to None: same rule as `height`. Must be a multiple of 32.
* `seed`: Random seed, defaults to 42.
* `num_inference_steps`: Number of inference steps, defaults to 25.
* `shift`: Timestep shift affecting sigma computation, defaults to 4.0.
* `edit_image`: Reference image. Passing it switches to image editing mode, defaults to None (text-to-image mode).
* `image_cfg_scale`: Image guidance scale for editing, defaults to 1.0. Values other than 1.0 add a third, unconditional branch.
* `resize_to_edit_image`: Whether to resize the edited image back to the size of `edit_image`, defaults to True.
* `min_pixels`, `max_pixels`: Pixel budget for the reference image in editing mode, both default to 262144 (512x512).

### Image Editing

Passing `edit_image` switches to image editing mode. The reference image is encoded by the vision encoder and placed in the user turn before the instruction; its pixels also feed the diffusion head at every step. Unless `height`/`width` are given, the output uses the reference's processed resolution, which keeps its aspect ratio within the `min_pixels`/`max_pixels` budget. With `resize_to_edit_image=True` the result is resized back to the original reference size.

```python
from PIL import Image

edit_image = Image.open("input.jpg").convert("RGB")
image = pipe("Change the background to a snowy mountain.", edit_image=edit_image, seed=42)
```

Guidance combines three predictions when `image_cfg_scale != 1`: `full + cfg_scale * (posi - nega) + image_cfg_scale * (nega - full)`, where `posi` sees the reference and the instruction, `nega` sees only the reference, and `full` sees neither. With the default `image_cfg_scale=1.0` the `full` branch cancels out and is skipped.

### Image Understanding and Text Chat

`pipe.chat` runs autoregressive text generation with the same checkpoint. Pass `images` (a single image or a list) for image understanding, or omit it for plain text chat:

```python
answer = pipe.chat("Describe this image in detail.", images=edit_image)
answer = pipe.chat("Write a short poem about autumn.", think_mode="empty")
```

* `images`: Input images, a single image or a list, defaults to None (plain text chat).
* `system`: System prompt, defaults to an empty string.
* `think_mode`: `"think"` (default) lets the model write a `<think>` block before answering, `"empty"` prefills an empty think block, and `"none"` adds no prefill.
* `max_new_tokens`: Maximum number of generated tokens, defaults to 1024.
* `max_length`: Maximum length of the whole sequence, defaults to 9216.
* `temperature`, `top_k`, `top_p`, `seed`: Sampling options. The default `temperature=0.0` is greedy decoding.
* `min_pixels`, `max_pixels`: Pixel budget for input images, defaulting to 262144 and 4194304.
* `return_raw`: Return the full decoded text including the think block. By default only the answer after the last `</think>` is returned.

Decoding runs without a KV cache, as in training: every new token recomputes the whole sequence.

### Architecture Notes

* **Shared backbone**: understanding, generation and editing all run through the same language model. The diffusion head reads the backbone's hidden states for the noisy image tokens, so DiffSynth registers a single `hiuni_dit` component.
* **Pixel-space denoising without a VAE**: `latents` holds raster 16x16 RGB patches of shape `(4 * H/32 * W/32, 768)` in `[-1, 1]`. The model predicts the clean image `x0`, `model_fn` converts it to a flow velocity, and guidance is combined in velocity space before the scheduler step.
* **Resolution-dependent noise**: the initial noise is scaled by `sqrt(4 * H/32 * W/32 / 256)`, matching training.

## Model Training

Models in the HiUni series are trained uniformly via `examples/hiuni/model_training/train.py`. The script parameters include:

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
        * `--quant_options`: Dynamically quantize loaded models. Semicolon-separated entries, each `<model_string>:<method>[/<exclude_modules>]`, where `<model_string>` matches an entry in `--model_paths`/`--model_id_with_origin_paths`, `method` is a registered method (e.g. `bitsandbytes_nf4`), and `exclude_modules` optionally lists layers kept in full precision.
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
        * `--preset_lora_model`: Which model to integrate preset LoRA into, e.g. `dit`.
    * Gradient Configuration
        * `--use_gradient_checkpointing`: Whether to enable gradient checkpointing.
        * `--use_gradient_checkpointing_offload`: Whether to offload gradient checkpointing to CPU memory.
        * `--gradient_accumulation_steps`: Number of gradient accumulation steps.
    * Resolution Configuration
        * `--height`: Height of the image. Leave empty to enable dynamic resolution.
        * `--width`: Width of the image. Leave empty to enable dynamic resolution.
        * `--max_pixels`: Maximum pixel area, images larger than this will be scaled down during dynamic resolution.
* HiUni Specific Parameters
    * `--tokenizer_path`: Path to the tokenizer. Left empty, it is loaded from `ernie-research/HiUni`.
    * `--processor_path`: Path to the image processor. Left empty, it is loaded from `ernie-research/HiUni`.
    * `--initialize_model_on_cpu`: Whether to initialize the models on CPU.

Example dataset:

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "hiuni/HiUni/*" --local_dir ./data/diffsynth_example_dataset
```

We provide recommended training scripts for each model, please refer to the table in "Model Overview" above. For guidance on writing model training scripts, see [Model Training](../Pipeline_Usage/Model_Training.md); for more advanced training algorithms, see [Training Framework Overview](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/en/Training/).
