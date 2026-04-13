# ERNIE-Image

ERNIE-Image is a powerful image generation model with 8B parameters developed by Baidu, featuring a compact and efficient architecture with strong instruction-following capability. Based on an 8B DiT backbone, it delivers performance comparable to larger (20B+) models in certain scenarios while maintaining parameter efficiency. It offers reliable performance in instruction understanding and execution, text generation (English/Chinese/Japanese), and overall stability.

## Installation

Before performing model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information on installation, please refer to [Setup Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Running the following code will load the [PaddlePaddle/ERNIE-Image](https://www.modelscope.cn/models/PaddlePaddle/ERNIE-Image) model for inference. VRAM management is enabled, the framework automatically controls parameter loading based on available VRAM, requiring a minimum of 3G VRAM.

```python
from diffsynth.pipelines.ernie_image import ErnieImagePipeline, ModelConfig
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
pipe = ErnieImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device='cuda',
    model_configs=[
        ModelConfig(model_id="PaddlePaddle/ERNIE-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="PaddlePaddle/ERNIE-Image", origin_file_pattern="text_encoder/model.safetensors", **vram_config),
        ModelConfig(model_id="PaddlePaddle/ERNIE-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="PaddlePaddle/ERNIE-Image", origin_file_pattern="tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

image = pipe(
    prompt="一只黑白相间的中华田园犬",
    negative_prompt="",
    height=1024,
    width=1024,
    seed=42,
    num_inference_steps=50,
    cfg_scale=4.0,
)
image.save("output.jpg")
```

## Model Overview

|Model ID|Inference|Low VRAM Inference|Full Training|Full Training Validation|LoRA Training|LoRA Training Validation|
|-|-|-|-|-|-|-|
|[PaddlePaddle/ERNIE-Image: T2I](https://www.modelscope.cn/models/PaddlePaddle/ERNIE-Image)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ernie_image/model_inference/Ernie-Image-T2I.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ernie_image/model_inference_low_vram/Ernie-Image-T2I.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ernie_image/model_training/full/Ernie-Image-T2I.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ernie_image/model_training/validate_full/Ernie-Image-T2I.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ernie_image/model_training/lora/Ernie-Image-T2I.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ernie_image/model_training/validate_lora/Ernie-Image-T2I.py)|
|[PaddlePaddle/ERNIE-Image-Turbo: T2I](https://www.modelscope.cn/models/PaddlePaddle/ERNIE-Image-Turbo)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ernie_image/model_inference/Ernie-Image-Turbo-T2I.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ernie_image/model_inference_low_vram/Ernie-Image-Turbo-T2I.py)|—|—|—|—|

## Model Inference

The model is loaded via `ErnieImagePipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details.

The input parameters for `ErnieImagePipeline` inference include:

* `prompt`: The prompt describing the content to appear in the image.
* `negative_prompt`: The negative prompt describing what should not appear in the image, default value is `""`.
* `cfg_scale`: Classifier-free guidance parameter, default value is 4.0.
* `height`: Image height, must be a multiple of 16, default value is 1024.
* `width`: Image width, must be a multiple of 16, default value is 1024.
* `seed`: Random seed. Default is `None`, meaning completely random.
* `rand_device`: The computing device for generating random Gaussian noise matrices, default is `"cuda"`. When set to `cuda`, different GPUs will produce different results.
* `num_inference_steps`: Number of inference steps, default value is 50.

If VRAM is insufficient, please enable [VRAM Management](../Pipeline_Usage/VRAM_management.md). We provide recommended low-VRAM configurations for each model in the "Model Overview" table above.

## Model Training

ERNIE-Image series models are trained uniformly via [`examples/ernie_image/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ernie_image/model_training/train.py). The script parameters include:

* General Training Parameters
    * Dataset Configuration
        * `--dataset_base_path`: Root directory of the dataset.
        * `--dataset_metadata_path`: Path to the dataset metadata file.
        * `--dataset_repeat`: Number of dataset repeats per epoch.
        * `--dataset_num_workers`: Number of processes per DataLoader.
        * `--data_file_keys`: Field names to load from metadata, typically paths to image or video files, separated by `,`.
    * Model Loading Configuration
        * `--model_paths`: Paths to load models from, in JSON format.
        * `--model_id_with_origin_paths`: Model IDs with original paths, e.g., `"PaddlePaddle/ERNIE-Image:transformer/diffusion_pytorch_model*.safetensors"`, separated by commas.
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
* ERNIE-Image Specific Parameters
    * `--tokenizer_path`: Path to the tokenizer, leave empty to auto-download from remote.

We provide an example image dataset for testing, which can be downloaded with the following command:

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

We provide recommended training scripts for each model, please refer to the table in "Model Overview" above. For guidance on writing model training scripts, see [Model Training](../Pipeline_Usage/Model_Training.md); for more advanced training algorithms, see [Training Framework Overview](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/en/Training/).
