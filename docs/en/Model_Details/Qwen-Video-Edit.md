# Qwen-Video-Edit

Qwen-Video-Edit is a video editing model based on the Qwen-Image architecture. The model takes an input video and a text prompt, and generates an edited video that matches the prompt description. It uses QwenImageDiT as the core DiT backbone, combined with Wan2.1 VAE for video encoding/decoding, and a QwenVideoEditAdapter to project video features into the DiT feature space.

## Installation

Before using this project for model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information about installation, please refer to [Install Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Run the following code to quickly load the [yunpeng1998/Qwen-Video-Edit](https://www.modelscope.cn/models/yunpeng1998/Qwen-Video-Edit) model and perform inference. VRAM management is enabled, and the framework will automatically control model parameter loading based on remaining VRAM.

```python
import torch
from modelscope import dataset_snapshot_download
from diffsynth.core import ModelConfig
from diffsynth.pipelines.qwen_video_edit import QwenVideoEditPipeline
from diffsynth.utils.data import VideoData, save_video

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

dataset_snapshot_download(
    "DiffSynth-Studio/diffsynth_example_dataset",
    local_dir="data/diffsynth_example_dataset",
    allow_file_pattern="qwen_video_edit/Qwen-Video-Edit/*"
)

edit_video = VideoData("data/diffsynth_example_dataset/qwen_video_edit/Qwen-Video-Edit/source.mp4")
prompts = [
    "Transform the video into Japanese anime style",
]
pipe = QwenVideoEditPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="yunpeng1998/Qwen-Video-Edit", origin_file_pattern="360P/step-30000.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
    ],
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
video = pipe(edit_video=edit_video, prompts=prompts, height=640, width=384, num_frames=45, cfg_scale=4.0, num_inference_steps=40, seed=0)
save_video(video, "video_Qwen-Video-Edit.mp4", fps=16)
```

## Model Overview

| Model ID | Inference | Low VRAM Inference | Full Training | Validation After Full Training | LoRA Training | Validation After LoRA Training |
| - | - | - | - | - | - | - |
| [yunpeng1998/Qwen-Video-Edit](https://www.modelscope.cn/models/yunpeng1998/Qwen-Video-Edit) | [code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_video_edit/model_inference/Qwen-Video-Edit.py) | [code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_video_edit/model_inference_low_vram/Qwen-Video-Edit.py) | [code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_video_edit/model_training/full/Qwen-Video-Edit.sh) | [code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_video_edit/model_training/validate_full/Qwen-Video-Edit.py) | [code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_video_edit/model_training/lora/Qwen-Video-Edit.sh) | [code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_video_edit/model_training/validate_lora/Qwen-Video-Edit.py) |

## Model Inference

Models are loaded via `QwenVideoEditPipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models).

Input parameters for `QwenVideoEditPipeline` inference include:

* `edit_video`: Input video, i.e., the source video to be edited. Type is `list[PIL.Image.Image]`, loaded via `VideoData`.
* `num_frames`: Number of video frames, default is 45. The model processes video in 45-frame chunks, each chunk corresponds to one prompt in the `prompts` list.
* `height`: Video height, must be a multiple of 16.
* `width`: Video width, must be a multiple of 16.
* `tiled`: Whether to enable VAE tiling inference, default is `False`. Setting to `True` can significantly reduce VRAM usage during VAE encoding/decoding stages, producing slight errors and slightly longer inference time.
* `tile_size`: Tile size during VAE encoding/decoding stages, default is `(30, 52)`, only effective when `tiled=True`.
* `tile_stride`: Tile stride during VAE encoding/decoding stages, default is `(15, 26)`, only effective when `tiled=True`, must be less than or equal to `tile_size`.
* `prompts`: List of prompts, each element corresponds to the editing instruction for one chunk.
* `negative_prompt`: Negative prompt describing content that should not appear in the video, default value is `" "`.
* `cfg_scale`: Classifier-free guidance parameter, default value is 4. When set to 1, it no longer takes effect.
* `seed`: Random seed. Default is `None`, meaning completely random.
* `rand_device`: Computing device for generating random Gaussian noise matrix, default is `"cpu"`. When set to `cuda`, different GPUs will produce different generation results.
* `num_inference_steps`: Number of inference steps, default value is 40.
* `zero_cond_t`: Whether to zero out condition features at timestep t=0.
* `progress_bar_cmd`: Progress bar, default is `tqdm.tqdm`. Can be disabled by setting to `lambda x:x`.

## Model Training

Qwen-Video-Edit is trained through [`examples/qwen_video_edit/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_video_edit/model_training/train.py), and the script parameters include:

* General Training Parameters
    * Dataset Basic Configuration
        * `--dataset_base_path`: Root directory of the dataset.
        * `--dataset_metadata_path`: Metadata file path of the dataset.
        * `--dataset_repeat`: Number of times the dataset is repeated in each epoch.
        * `--dataset_num_workers`: Number of processes for each DataLoader.
        * `--data_file_keys`: Field names to be loaded from metadata, separated by `,`. For Qwen-Video-Edit, set to `"input_video,video"`, where `input_video` is the condition video (source video), and `video` is the target video.
    * Model Loading Configuration
        * `--model_paths`: Paths of models to be loaded. JSON format.
        * `--model_id_with_origin_paths`: Model IDs with original paths, e.g., `"yunpeng1998/Qwen-Video-Edit:360P/step-30000.safetensors"`. Separated by commas.
        * `--extra_inputs`: Extra input parameters required by the model Pipeline, separated by `,`.
        * `--fp8_models`: Models loaded in FP8 format, consistent with `--model_paths` or `--model_id_with_origin_paths` format. Currently only supports models whose parameters are not updated by gradients.
        * `--quant_options`: Dynamically quantize loaded models. Semicolon-separated entries, each `<model_string>:<method>[/<exclude_modules>]`.
    * Training Basic Configuration
        * `--learning_rate`: Learning rate.
        * `--num_epochs`: Number of epochs.
        * `--trainable_models`: Trainable models, e.g., `dit`, `adapter`.
        * `--find_unused_parameters`: Whether there are unused parameters in DDP training, needs to be enabled to avoid errors in multi-GPU training.
        * `--weight_decay`: Weight decay size, see [torch.optim.AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).
        * `--task`: Training task, default is `sft`.
    * Output Configuration
        * `--output_path`: Model saving path.
        * `--remove_prefix_in_ckpt`: Remove prefix in the state dict of the model file.
        * `--save_steps`: Interval of training steps to save the model. If this parameter is left blank, the model is saved once per epoch.
    * LoRA Configuration
        * `--lora_base_model`: Which model to add LoRA to.
        * `--lora_target_modules`: Which layers to add LoRA to.
        * `--lora_rank`: Rank of LoRA.
        * `--lora_checkpoint`: Path of the LoRA checkpoint. If this path is provided, LoRA will be loaded from this checkpoint.
        * `--preset_lora_path`: Preset LoRA checkpoint path. If this path is provided, this LoRA will be loaded in the form of being merged into the base model.
        * `--preset_lora_model`: Model that the preset LoRA is merged into, e.g., `dit`.
    * Gradient Configuration
        * `--use_gradient_checkpointing`: Whether to enable gradient checkpointing.
        * `--use_gradient_checkpointing_offload`: Whether to offload gradient checkpointing to memory.
        * `--gradient_accumulation_steps`: Number of gradient accumulation steps.
    * Video Width/Height Configuration
        * `--height`: Height of the video.
        * `--width`: Width of the video.
        * `--num_frames`: Number of video frames, default is 45.
        * `--max_pixels`: Maximum pixel area of video frames.
* Qwen-Video-Edit Specific Parameters
    * `--tokenizer_path`: Path of the tokenizer, leave blank to automatically download from remote.
    * `--processor_path`: Path of the processor, leave blank to automatically download from remote.
    * `--zero_cond_t`: Whether to zero out condition features at timestep t=0.

We have built a sample video dataset for your testing. You can download this dataset with the following command:

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "qwen_video_edit/Qwen-Video-Edit/*" --local_dir ./data/diffsynth_example_dataset
```

We have written recommended training scripts for the model, please refer to the table in the "Model Overview" section above. For how to write model training scripts, please refer to [Model Training](../Pipeline_Usage/Model_Training.md); for more advanced training algorithms, please refer to [Training Framework Detailed Explanation](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/en/Training/).
