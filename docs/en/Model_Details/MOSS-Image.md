# MOSS-Image

MOSS-Image is a family of 5B text-to-image models open-sourced by OpenMOSS. Built on a single-stream DiT backbone paired with a Qwen3.5 text encoder, it accepts natural-language prompts, tag sequences, and JSON layouts, allowing precise control over text placement and object arrangement. It delivers stable results for text rendering in Chinese, English, and Japanese, as well as for both anime and photorealistic styles. Base is the foundation model, well suited for community LoRA fine-tuning; SFT is fine-tuned from Base on curated data, producing clearer text grouping and better organized layouts.

## Installation

Before performing model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information on installation, please refer to [Setup Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Running the following code will load the [openmoss/MOSS-Image-Base](https://www.modelscope.cn/models/openmoss/MOSS-Image-Base) model for inference. VRAM management is enabled, the framework automatically controls parameter loading based on available VRAM.

```python
from diffsynth.pipelines.moss_image import MOSSImagePipeline, ModelConfig
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
vae_vram_config = {**vram_config, "offload_dtype": torch.float32, "onload_dtype": torch.float32, "preparing_dtype": torch.float32, "computation_dtype": torch.float32}
pipe = MOSSImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="openmoss/MOSS-Image-Base", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="openmoss/MOSS-Image-Base", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="openmoss/MOSS-Image-Base", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vae_vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="openmoss/MOSS-Image-Base", origin_file_pattern="tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

prompt = '''A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small sticky note on the refrigerator door is handwritten in Chinese:
"记得买牛奶和鸡蛋！
牛奶两盒，鸡蛋一打。
顺便带一袋面包。"'''

image = pipe(prompt, height=1024, width=1024, seed=42, num_inference_steps=50, cfg_scale=3.0)
image.save("image_MOSS-Image-Base.jpg")
```

## Model Overview

|Model ID|Inference|Low VRAM Inference|Full Training|Full Training Validation|LoRA Training|LoRA Training Validation|
|-|-|-|-|-|-|-|
|[openmoss/MOSS-Image-Base](https://www.modelscope.cn/models/openmoss/MOSS-Image-Base)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_inference/MOSS-Image-Base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_inference_low_vram/MOSS-Image-Base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/full/MOSS-Image-Base.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/validate_full/MOSS-Image-Base.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/lora/MOSS-Image-Base.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/validate_lora/MOSS-Image-Base.py)|
|[openmoss/MOSS-Image-SFT](https://www.modelscope.cn/models/openmoss/MOSS-Image-SFT)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_inference/MOSS-Image-SFT.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_inference_low_vram/MOSS-Image-SFT.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/full/MOSS-Image-SFT.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/validate_full/MOSS-Image-SFT.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/lora/MOSS-Image-SFT.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/validate_lora/MOSS-Image-SFT.py)|
|[DiffSynth-Studio/MOSS-Image-EntroPack](https://www.modelscope.cn/models/DiffSynth-Studio/MOSS-Image-EntroPack)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_inference/MOSS-Image-EntroPack.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_inference_low_vram/MOSS-Image-EntroPack.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/lora/MOSS-Image-entropack.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/validate_lora/MOSS-Image-entropack.py)|

## Model Inference

The model is loaded via `MOSSImagePipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details.

The input parameters for `MOSSImagePipeline` inference include:

* `prompt`: The prompt describing the content to appear in the image. Besides natural language and tag sequences, it also accepts a JSON layout string for precise control over text and object placement.
* `negative_prompt`: The negative prompt describing what should not appear in the image, default value is `""`.
* `cfg_scale`: Classifier-free guidance parameter, default value is 3.0.
* `max_sequence_length`: Maximum number of text tokens to encode, default value is 4096.
* `uncond_skip_layers`: Indices of DiT layers skipped in the negative branch, default value is `(22,)`. Skipping halves the computation of the negative branch; set it to `()` to disable layer skipping.
* `height`: Image height, must be a multiple of 16, default value is 1024.
* `width`: Image width, must be a multiple of 16, default value is 1024.
* `seed`: Random seed. Default is `None`, meaning completely random.
* `rand_device`: The computing device for generating random Gaussian noise matrices, default is `"cpu"`.
* `num_inference_steps`: Number of inference steps, default value is 50.
* `sigma_shift`: Shift coefficient of the flow matching schedule, default value is 5.0.
* `progress_bar_cmd`: Function wrapping the inference timesteps with a progress bar, defaults to `tqdm`.

If VRAM is insufficient, please enable [VRAM Management](../Pipeline_Usage/VRAM_management.md). We provide recommended low-VRAM configurations for each model in the "Model Overview" table above.

## Model Training

Models in the MOSS-Image series are trained uniformly via [`examples/moss_image/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/moss_image/model_training/train.py). The script parameters include:

* General Training Parameters
    * Dataset Configuration
        * `--dataset_base_path`: Root directory of the dataset.
        * `--dataset_metadata_path`: Path to the dataset metadata file.
        * `--dataset_repeat`: Number of dataset repeats per epoch.
        * `--dataset_num_workers`: Number of processes per DataLoader.
        * `--data_file_keys`: Field names to load from metadata, typically paths to image or video files, separated by `,`.
    * Model Loading Configuration
        * `--model_paths`: Paths to load models from, in JSON format.
        * `--model_id_with_origin_paths`: Model IDs with original paths, e.g., `"openmoss/MOSS-Image-Base:transformer/diffusion_pytorch_model*.safetensors"`, separated by commas.
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
        * `--preset_lora_model`: Which model to integrate preset LoRA into, e.g., `dit`.
    * Gradient Configuration
        * `--use_gradient_checkpointing`: Whether to enable gradient checkpointing.
        * `--use_gradient_checkpointing_offload`: Whether to offload gradient checkpointing to CPU memory.
        * `--gradient_accumulation_steps`: Number of gradient accumulation steps.
    * Resolution Configuration
        * `--height`: Height of the image. Leave empty to enable dynamic resolution.
        * `--width`: Width of the image. Leave empty to enable dynamic resolution.
        * `--max_pixels`: Maximum pixel area, images larger than this will be scaled down during dynamic resolution.
* MOSS-Image Specific Parameters
    * `--tokenizer_path`: Path to the tokenizer, leave empty to auto-download from remote.

We provide an example image dataset for testing, which can be downloaded with the following command:

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

We provide recommended training scripts for each model, please refer to the table in "Model Overview" above. For guidance on writing model training scripts, see [Model Training](../Pipeline_Usage/Model_Training.md); for more advanced training algorithms, see [Training Framework Overview](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/en/Training/).
