# Fibo 2.0

Fibo 2.0 is an 8B unified image generation and editing model developed by Bria AI. It runs in the latent space of the FLUX.2 VAE, encodes prompts and reference images with Qwen3-VL-4B, and uses a single-stream DiT with 44 blocks. Text-to-image and image editing share one set of weights.

Fibo 2.0 Dev is the full, non-distilled base model, trained with classifier-free guidance and intended for fine-tuning. Fibo 2.0 Edit Turbo is a distilled editing-only variant for 4-step, guidance-free inference.

## Installation

Before performing model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information on installation, please refer to [Setup Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Run the following code to quickly load the [briaai/Fibo-2.0-dev](https://www.modelscope.cn/models/briaai/Fibo-2.0-dev) model and perform text-to-image inference. VRAM management is enabled, and the framework will automatically control model parameter loading based on remaining VRAM.

```python
import json
import torch
from diffsynth.pipelines.fibo import FiboPipeline, ModelConfig

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = FiboPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="transformer/*.safetensors", **vram_config),
        ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="text_encoder/model.safetensors", **vram_config),
        ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="processor/"),
    font_config=ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="fonts/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
caption = {
    "short_description": "An origami whale swimming through paper clouds.",
    "objects": [{"description": "a blue origami whale, folded from glossy paper", "location": "center"}],
    "background_setting": "a bright sky made of layered white paper clouds",
    "style_medium": "paper craft photograph",
}
prompt = json.dumps(caption, separators=(",", ":"), ensure_ascii=False)
image = pipe(prompt=prompt, height=1024, width=1024, seed=42, num_inference_steps=50, cfg_scale=5.0)
image.save("image.jpg")
```

## Model Overview

|Model ID|Inference|Low VRAM Inference|Full Training|Full Training Validation|LoRA Training|LoRA Training Validation|
|-|-|-|-|-|-|-|
|[briaai/Fibo-2.0-dev](https://www.modelscope.cn/models/briaai/Fibo-2.0-dev)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_inference/Fibo-2.0-dev.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_inference_low_vram/Fibo-2.0-dev.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/full/Fibo-2.0-dev-T2I.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/validate_full/Fibo-2.0-dev-T2I.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/lora/Fibo-2.0-dev-T2I.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/validate_lora/Fibo-2.0-dev-T2I.py)|
|[briaai/Fibo-2.0-edit-turbo](https://www.modelscope.cn/models/briaai/Fibo-2.0-edit-turbo)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_inference/Fibo-2.0-edit-turbo.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_inference_low_vram/Fibo-2.0-edit-turbo.py)|-|-|-|-|

The training columns above link to the text-to-image scripts. Fibo 2.0 Dev also supports image-editing training, see [Model Training](#model-training).

## Model Inference

The model is loaded via `FiboPipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details.

`from_pretrained` also accepts:

* `processor_config`: `ModelConfig` of the Qwen3-VL processor.
* `font_config`: `ModelConfig` of the fonts used to draw `text_render` images. It can point to a font directory, a font file, or a list of font files.

The input parameters for `FiboPipeline` inference include:

* `prompt`: A structured JSON caption, a plain description, or an edit instruction. Edit prompts may also be a JSON caption of the result with an `edit_instruction` field.
* `negative_prompt`: Used when `cfg_scale > 1`. Defaults to the null caption Fibo 2.0 was trained with.
* `edit_image`: The image or list of reference images to edit. Leave empty for text-to-image.
* `mask`: For single-image editing, white marks the region to regenerate.
* `text_images`: Images of the text in a caption's `text_render` entries. When omitted, they are drawn from the caption.
* `fonts`: Font paths used to draw characters that Pillow's built-in font lacks.
* `render_text_images`: Whether to draw text images from the caption, default is `True`.
* `height`, `width`: Output size, multiples of 16. When editing and left empty, the trained size for the first image's aspect ratio is used.
* `cfg_scale`: Classifier-free guidance scale, default is 5.0.
* `num_inference_steps`: Number of inference steps, default is 30.
* `denoising_strength`: Denoising strength, range is 0~1, default is 1. Lower values start from a lower noise level.
* `seed`, `rand_device`: Random seed and the device used to generate noise.
* `progress_bar_cmd`: Progress bar, default is `tqdm.tqdm`. Set to `lambda x:x` to disable it.

If VRAM is insufficient, please enable [VRAM Management](../Pipeline_Usage/VRAM_management.md). We provide recommended low VRAM configurations for each model in the example code, see the table in the "Model Overview" section above.

### Image Editing

`edit_image` accepts one image, or a list of up to 5 images for multi-reference editing. The example below uses the distilled `Fibo-2.0-edit-turbo` model:

```python
from PIL import Image
from diffsynth.pipelines.fibo import FiboPipeline, ModelConfig

pipe = FiboPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="briaai/Fibo-2.0-edit-turbo", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="briaai/Fibo-2.0-edit-turbo", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="briaai/Fibo-2.0-edit-turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="briaai/Fibo-2.0-edit-turbo", origin_file_pattern="processor/"),
    font_config=ModelConfig(model_id="briaai/Fibo-2.0-edit-turbo", origin_file_pattern="fonts/"),
)
source = Image.open("photo.jpg").convert("RGB")
image = pipe(prompt="replace the jacket with a red leather one", edit_image=source, seed=42, num_inference_steps=4, cfg_scale=1.0)
image.save("edited.jpg")
```

`Fibo-2.0-dev` uses 50 steps with `cfg_scale=5.0`; `Fibo-2.0-edit-turbo` uses 4 steps with `cfg_scale=1.0`.

### Masked Editing

For single-image editing, pass `mask` together with `edit_image` (white marks the region to regenerate):

```python
mask = Image.open("mask.png").convert("L")  # white: regenerate, black: keep
image = pipe(
    prompt="replace the suitcase with a vintage brown leather one",
    edit_image=source, mask=mask, seed=42, num_inference_steps=4, cfg_scale=1.0,
)
```

### Text Rendering

Fibo renders text from a caption's `text_render` entries. Each entry's text is drawn as a small black-on-white image and shown to Qwen3-VL together with the prompt:

```python
caption = {
    "short_description": "A young woman holding a white sign that reads \"ModelScope\".",
    "objects": [{"description": "a white sign with bold black text", "location": "center"}],
    "style_medium": "photograph",
    "text_render": [{"text": "ModelScope"}],
}
image = pipe(
    prompt=json.dumps(caption, separators=(",", ":"), ensure_ascii=False),
    height=1024, width=1024, seed=42, num_inference_steps=50, cfg_scale=5.0,
)
```

`font_config` loads the fonts shipped with the model. To render characters that those fonts lack, pass additional font paths through `fonts`:

```python
image = pipe(prompt=prompt, fonts=["/path/to/a/cjk-font.ttf"], ...)
```

Characters not covered by any font are drawn with Pillow's built-in font. Text rendering accuracy is model-dependent, especially for non-Latin scripts.

## Model Training

Fibo 2.0 Dev is trained via [`examples/fibo/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/train.py). The script parameters include:

* General Training Parameters
    * Dataset Basic Configuration
        * `--dataset_base_path`: Root directory of the dataset.
        * `--dataset_metadata_path`: Metadata file path of the dataset.
        * `--dataset_repeat`: Number of times the dataset is repeated in each epoch.
        * `--dataset_num_workers`: Number of processes for each DataLoader.
        * `--data_file_keys`: Field names to be loaded from metadata, usually image or video file paths, separated by `,`.
    * Model Loading Configuration
        * `--model_paths`: Paths of models to be loaded. JSON format.
        * `--model_id_with_origin_paths`: Model IDs with original paths, e.g., `"briaai/Fibo-2.0-dev:transformer/*.safetensors"`. Separated by commas.
        * `--extra_inputs`: Extra input parameters required by the model Pipeline, e.g., `edit_image` and `mask` when training the image editing model, separated by `,`.
        * `--fp8_models`: Models loaded in FP8 format, consistent with `--model_paths` or `--model_id_with_origin_paths` format. Currently only supports models whose parameters are not updated by gradients (no gradient backpropagation, or gradients only update their LoRA).
        * `--quant_options`: Dynamically quantize loaded models. Semicolon-separated entries, each `<model_string>:<method>[/<exclude_modules>]`, where `<model_string>` matches an entry in `--model_paths`/`--model_id_with_origin_paths`, `method` is a registered method (e.g. `bitsandbytes_nf4`), and `exclude_modules` optionally lists layers kept in full precision.
    * Training Basic Configuration
        * `--learning_rate`: Learning rate.
        * `--num_epochs`: Number of epochs.
        * `--trainable_models`: Trainable models, e.g., `dit`, `vae`, `text_encoder`.
        * `--find_unused_parameters`: Whether there are unused parameters in DDP training. Some models contain redundant parameters that do not participate in gradient calculation, and this setting needs to be enabled to avoid errors in multi-GPU training.
        * `--weight_decay`: Weight decay size, see [torch.optim.AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).
        * `--task`: Training task, default is `sft`. Some models support more training modes, please refer to the documentation of each specific model.
    * Output Configuration
        * `--output_path`: Model saving path.
        * `--remove_prefix_in_ckpt`: Remove prefix in the state dict of the model file.
        * `--save_steps`: Interval of training steps to save the model. If this parameter is left blank, the model is saved once per epoch.
    * LoRA Configuration
        * `--lora_base_model`: Which model to add LoRA to.
        * `--lora_target_modules`: Which layers to add LoRA to.
        * `--lora_rank`: Rank of LoRA.
        * `--lora_checkpoint`: Path of the LoRA checkpoint. If this path is provided, LoRA will be loaded from this checkpoint.
        * `--preset_lora_path`: Preset LoRA checkpoint path. If this path is provided, this LoRA will be loaded in the form of being merged into the base model. This parameter is used for LoRA differential training.
        * `--preset_lora_model`: Model that the preset LoRA is merged into, e.g., `dit`.
    * Gradient Configuration
        * `--use_gradient_checkpointing`: Whether to enable gradient checkpointing.
        * `--use_gradient_checkpointing_offload`: Whether to offload gradient checkpointing to memory.
        * `--gradient_accumulation_steps`: Number of gradient accumulation steps.
    * Image Width/Height Configuration (Applicable to Image Generation and Video Generation Models)
        * `--height`: Height of image or video. Leave `height` and `width` blank to enable dynamic resolution.
        * `--width`: Width of image or video. Leave `height` and `width` blank to enable dynamic resolution.
        * `--max_pixels`: Maximum pixel area of image or video frames. When dynamic resolution is enabled, images with resolution larger than this value will be downscaled, and images with resolution smaller than this value will remain unchanged.
* Fibo 2.0 Specific Parameters
    * `--processor_path`: Path or `model_id:origin_file_pattern` of the Qwen3-VL processor, leave blank to automatically download from remote.
    * `--font_path`: Path or `model_id:origin_file_pattern` of the fonts used for text rendering, leave blank to automatically download from remote.
    * `--initialize_model_on_cpu`: Whether to initialize the model on CPU.

We have built sample datasets, including a text-to-image dataset and an image-editing dataset. You can download them with the following command:

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "fibo/*" --local_dir ./data/diffsynth_example_dataset
```

* `data/diffsynth_example_dataset/fibo/Fibo-2.0-dev`: text-to-image, with `metadata.csv` containing `image` and `prompt`.
* `data/diffsynth_example_dataset/fibo/Fibo-2.0-dev-Edit`: image editing, with `metadata.json` containing `image`, `prompt`, `edit_image`, and an optional `mask`.

Fibo 2.0 Dev supports both text-to-image and image-editing training. Training reuses the inference Pipeline: the VAE and text encoder are frozen, and the DiT is trained with flow matching. Full training uses `--trainable_models "dit"`, while LoRA training uses `--lora_base_model "dit"`. For image editing, `edit_image` and `mask` are loaded with `UnifiedDataset` and injected through `--extra_inputs`.

| Mode | Full Training | LoRA Training | Full Validation | LoRA Validation |
|-|-|-|-|-|
| Text-to-image | [script](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/full/Fibo-2.0-dev-T2I.sh) | [script](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/lora/Fibo-2.0-dev-T2I.sh) | [script](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/validate_full/Fibo-2.0-dev-T2I.py) | [script](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/validate_lora/Fibo-2.0-dev-T2I.py) |
| Image editing | [script](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/full/Fibo-2.0-dev-Edit.sh) | [script](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/lora/Fibo-2.0-dev-Edit.sh) | [script](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/validate_full/Fibo-2.0-dev-Edit.py) | [script](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/fibo/model_training/validate_lora/Fibo-2.0-dev-Edit.py) |

The full training scripts use `--config_file examples/fibo/model_training/full/accelerate_config.yaml`; the LoRA scripts run on a single GPU by default. `mask` is optional per sample, and rows without `mask` are trained as ordinary editing samples. Fibo 2.0 Edit Turbo is not intended for fine-tuning; train on Fibo 2.0 Dev instead.

We have written recommended training scripts for each model, please refer to the tables above. For how to write model training scripts, please refer to [Model Training](../Pipeline_Usage/Model_Training.md); for more advanced training algorithms, please refer to [Training Framework Detailed Explanation](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/en/Training/).

## License

Fibo 2.0 is released under CC BY-NC 4.0 for non-commercial use. Commercial use requires a separate agreement with Bria AI.
