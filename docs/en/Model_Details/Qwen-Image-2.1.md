# Qwen-Image-2.1

Qwen-Image-2.1 is a unified text-to-image and image editing model open-sourced by the Qwen team, built on a single-stream block-causal DiT, a 64-channel RGBA VAE and a Qwen3-VL text encoder. It can directly generate RGBA images with an alpha channel.

## Installation

Before performing model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information on installation, please refer to [Setup Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Running the following code will load the [Qwen/Qwen-Image-2.1](https://www.modelscope.cn/models/Qwen/Qwen-Image-2.1) model for inference. VRAM management is enabled, the framework automatically controls parameter loading based on available VRAM, requiring a minimum of 7GB VRAM.

```python
from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline, ModelConfig
import torch
from PIL import Image

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
pipe = QwenImage21Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="vae/diffusion_pytorch_model*.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

# Text-to-Image, the output is an RGBA image
prompt = "Flat anime-style illustration, a girl with long black hair, wearing a JK uniform."
image = pipe(prompt, seed=0)
image.save("image1.png")

prompt = "Flat anime-style illustration, a sunny and cheerful high school girl."
image_2 = pipe(prompt=prompt, seed=0)
image_2.save("image2.png")

# Image Editing, the generated RGBA image is fed back as the condition
prompt = "Generate a group photo of these two characters."
edit_image = [Image.open("image1.png"), Image.open("image2.png")]
image_3 = pipe(prompt, edit_image=edit_image, seed=1)
image_3.save("image3.png")
```

## Model Overview

|Model ID|Inference|Low VRAM Inference|Full Training|Full Training Validation|LoRA Training|LoRA Training Validation|
|-|-|-|-|-|-|-|
|[Qwen/Qwen-Image-2.1](https://www.modelscope.cn/models/Qwen/Qwen-Image-2.1)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_inference/Qwen-Image-2.1.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_inference_low_vram/Qwen-Image-2.1.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/full/Qwen-Image-2.1.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/validate_full/Qwen-Image-2.1.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/lora/Qwen-Image-2.1.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/validate_lora/Qwen-Image-2.1.py)|

## Model Inference

The model is loaded via `QwenImage21Pipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details.

The input parameters for `QwenImage21Pipeline` inference include:

* `prompt`: Prompt describing the content of the image. Defaults to `" "`; an empty string is treated as a single space, because Qwen has no bos token and an empty string would leave the text encoder with nothing to read.
* `negative_prompt`: Negative prompt, defaults to `" "`. It conditions the negative branch when `cfg_scale` is greater than 1.
* `cfg_scale`: CFG strength, defaults to 1.0. Whether CFG is enabled is decided solely by whether this value is greater than 1.
* `edit_image`: Image(s) to edit, only a `PIL.Image` or a list of `PIL.Image` is accepted. Text-to-image runs when it is left empty, image editing runs when it is provided.
* `height`: Image height, defaults to 1024 and is aligned to a multiple of 32; in editing mode `edit_image` is resized to the `height * width` area following its own aspect ratio.
* `width`: Image width, defaults to 1024, same rule as `height`.
* `seed`: Random seed. Defaults to `None`, i.e. fully random.
* `rand_device`: Device used to generate the Gaussian noise, defaults to `"cpu"`.
* `num_inference_steps`: Number of inference steps, defaults to 40.
* `use_kv_cache`: Whether to enable the per-layer KV cache under the block-causal condition, defaults to `True`.
* `tiled`: Whether to enable tiled VAE inference, defaults to `False`.
* `tile_size`: Tile size for VAE encoding/decoding, defaults to 256, effective only when `tiled=True`.
* `tile_stride`: Tile stride for VAE encoding/decoding, defaults to 192, effective only when `tiled=True`.

The pipeline returns a PIL image in RGBA mode.

If VRAM is insufficient, please enable [VRAM Management](../Pipeline_Usage/VRAM_management.md). We provide recommended low VRAM configurations for each model in the example code, see the table in the "Model Overview" section above.

### Transparent Backgrounds

The VAE works in a 4-channel RGBA pixel space and the 4th decoded channel is the alpha, so transparency is produced by the model from the prompt and the pipeline exposes no alpha-related switch. To obtain a transparent background:

* State explicitly in the prompt that the output is an asset with a transparent channel, e.g. `isolated on a fully transparent background`, `alpha matte`, `die-cut sticker`, `PNG with transparency`, `no background`.
* Avoid describing an environment or ambient light (e.g. "underwater", "indoor"), otherwise the model fills the canvas. Rewrite such atmosphere as attributes of the subject itself and add "nothing else in the scene".
* Save to a format that supports transparency, e.g. `image.save("image.png")`; `image.convert("RGB")` or saving as jpg discards the alpha.

Subjects with hair, ribbons or water-like lighting tend to produce wide semi-transparent transitions (intermediate alpha values). For harder edges, add descriptions like "clean cut-out edges" or "crisp silhouette" to the prompt.

## Model Training

Models in the Qwen-Image-2.1 series are trained uniformly via [`examples/qwen_image_21/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image_21/model_training/train.py). The script parameters include:

* General Training Parameters
    * Dataset Basic Configuration
        * `--dataset_base_path`: Root directory of the dataset.
        * `--dataset_metadata_path`: Metadata file path of the dataset.
        * `--dataset_repeat`: Number of times the dataset is repeated in each epoch.
        * `--dataset_num_workers`: Number of processes for each DataLoader.
        * `--data_file_keys`: Field names to be loaded from metadata, usually image or video file paths, separated by `,`.
    * Model Loading Configuration
        * `--model_paths`: Paths of models to be loaded. JSON format.
        * `--model_id_with_origin_paths`: Model IDs with original paths, e.g., `"Qwen/Qwen-Image-2.1:transformer/diffusion_pytorch_model*.safetensors"`. Separated by commas.
        * `--extra_inputs`: Extra input parameters required by the model Pipeline, e.g., the extra parameter `edit_image` when training image editing, separated by `,`.
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
* Qwen-Image-2.1 Specific Parameters
    * `--processor_path`: Path of the processor, leave blank to automatically download from remote.
    * `--initialize_model_on_cpu`: Whether to initialize the models on CPU, which lowers the peak VRAM usage when launching multi-GPU training.

Training images are loaded as RGBA by default, so datasets with an alpha channel can be used directly to train transparent generation. For image editing training, add `edit_image` to `--data_file_keys` and inject it into the pipeline via `--extra_inputs "edit_image"`; see the commented-out Edit block in the LoRA training script for the full configuration.

We have built a sample image dataset for your testing. You can download this dataset with the following command:

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

We have written recommended training scripts for each model, please refer to the table in the "Model Overview" section above. For how to write model training scripts, please refer to [Model Training](../Pipeline_Usage/Model_Training.md); for more advanced training algorithms, please refer to [Training Framework Detailed Explanation](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/en/Training/).
