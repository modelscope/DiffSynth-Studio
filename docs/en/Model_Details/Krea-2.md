# Krea-2

Krea-2 is an image generation model developed by the Krea team.

## Installation

Before performing model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information on installation, please refer to [Setup Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Running the following code will load the [krea/Krea-2-Raw](https://www.modelscope.cn/models/krea/Krea-2-Raw) model for inference. VRAM management is enabled, the framework automatically controls parameter loading based on available VRAM, requiring a minimum of 24GB VRAM.

```python
from diffsynth.pipelines.krea2 import Krea2Pipeline, ModelConfig
import torch

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
pipe = Krea2Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="krea/Krea-2-Raw", origin_file_pattern="raw.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", origin_file_pattern="*.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", origin_file_pattern=""),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 1,
)
prompt = "A cat standing on a stone."
image = pipe(prompt, seed=0, num_inference_steps=52, cfg_scale=4.5)
image.save("image.jpg")
```

## Model Overview

|Model ID|Inference|Low VRAM Inference|Full Training|Full Training Validation|LoRA Training|LoRA Training Validation|
|-|-|-|-|-|-|-|
|[krea/Krea-2-Raw](https://www.modelscope.cn/models/krea/Krea-2-Raw)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_inference/Krea-2-Raw.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_inference_low_vram/Krea-2-Raw.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_training/full/Krea-2-Raw.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_training/validate_full/Krea-2-Raw.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_training/lora/Krea-2-Raw.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_training/validate_lora/Krea-2-Raw.py)|
|[krea/Krea-2-Turbo](https://www.modelscope.cn/models/krea/Krea-2-Turbo)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_inference/Krea-2-Turbo.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_inference_low_vram/Krea-2-Turbo.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_training/full/Krea-2-Turbo.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_training/validate_full/Krea-2-Turbo.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_training/lora/Krea-2-Turbo.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_training/validate_lora/Krea-2-Turbo.py)|

## Model Inference

The model is loaded via `Krea2Pipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details.

The input parameters for `Krea2Pipeline` inference include:

* `prompt`: Prompt describing the content of the image to generate, default value is `""`.
* `negative_prompt`: Negative prompt describing content that should not appear in the image, default value is `""`.
* `cfg_scale`: Classifier-free guidance parameter, default value is 3.5.
* `height`: Image height, must be a multiple of 16, default value is 1024.
* `width`: Image width, must be a multiple of 16, default value is 1024.
* `seed`: Random seed, default is `None`, meaning completely random.
* `rand_device`: Computing device for generating random Gaussian noise matrix, default is `"cpu"`.
* `num_inference_steps`: Number of inference steps, default value is 52.
* `mu`: Timestep dynamic shift parameter, default is `None`.
* `progress_bar_cmd`: Progress bar, default is `tqdm.tqdm`. Can be disabled by setting to `lambda x:x`.

If VRAM is insufficient, please enable [VRAM Management](../Pipeline_Usage/VRAM_management.md). We provide recommended low VRAM configurations for each model in the example code, see the table in the "Model Overview" section above.

## Model Training

Models in the Krea-2 series are trained uniformly via [`examples/krea2/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/krea2/model_training/train.py). The script parameters include:

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
        * `--height`: Height of the image. Leave empty to enable dynamic resolution.
        * `--width`: Width of the image. Leave empty to enable dynamic resolution.
        * `--max_pixels`: Maximum pixel area, images larger than this will be scaled down during dynamic resolution.
* Krea-2 Specific Parameters
    * `--tokenizer_path`: Path to the tokenizer, leave blank to automatically download from remote.
    * `--initialize_model_on_cpu`: Whether to initialize models on CPU.
    * `--align_to_opensource_format`: Whether to align the LoRA format to the opensource format, useful for compatibility with other frameworks.

We have built a sample dataset for your testing. You can download it with the following command:

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "krea2/*" --local_dir ./data/diffsynth_example_dataset
```

We provide recommended training scripts for each model, please refer to the table in "Model Overview" above. For guidance on writing model training scripts, see [Model Training](../Pipeline_Usage/Model_Training.md); for more advanced training algorithms, see [Training Framework Overview](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/en/Training/).


## License

> **⚠️ Notice**: **Krea-2** weights (Raw and Turbo) are released under the [Krea 2 Community License](https://www.krea.ai/krea-2-licensing), **not** the Apache 2.0 license that governs DiffSynth-Studio itself.