# LingBot-Video

LingBot-Video is a flow-matching video generation model developed by the LingBot team; a single model handles text-to-video, image-to-video and text-to-image tasks.

Huge thanks to [NancyFyong](https://github.com/NancyFyong) for the outstanding contribution to the integration of this model!

## Installation

Before performing model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information on installation, please refer to [Setup Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Running the following code will load the [Robbyant/lingbot-video-dense-1.3b](https://modelscope.cn/models/Robbyant/lingbot-video-dense-1.3b) model for inference. VRAM management is enabled, the framework automatically controls parameter loading based on available VRAM, requiring a minimum of 6GB VRAM.

```python
import torch
import json
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.lingbot_video import LingBotVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download

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

pipe = LingBotVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="transformer/diffusion_pytorch_model.safetensors", **vram_config),
        ModelConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", origin_file_pattern="*.safetensors", **vram_config),
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", origin_file_pattern=""),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/diffsynth_example_dataset",
    local_dir="data/diffsynth_example_dataset",
    allow_file_pattern="lingbot_video/lingbot-video-dense-1.3b_t2v/*",
)
with open("data/diffsynth_example_dataset/lingbot_video/lingbot-video-dense-1.3b_t2v/t2v_example_1.json", "r", encoding="utf-8") as f:
    caption = json.load(f)

video = pipe(
    prompt=caption,
    negative_prompt=pipe.default_negative_prompt,
    height=480, width=832, num_frames=81,
    num_inference_steps=40, cfg_scale=3.0,
    seed=0,
)
save_video(video, "video.mp4", fps=15, quality=10)
```

## Model Overview

|Model ID|Inference|Low VRAM Inference|Full Training|Full Training Validation|LoRA Training|LoRA Training Validation|
|-|-|-|-|-|-|-|
|[Robbyant/lingbot-video-dense-1.3b: T2V](https://modelscope.cn/models/Robbyant/lingbot-video-dense-1.3b)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_inference/lingbot-video-dense-1.3b_t2v.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_inference_low_vram/lingbot-video-dense-1.3b_t2v.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/full/lingbot-video-dense-1.3b_t2v.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/validate_full/lingbot-video-dense-1.3b_t2v.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/lora/lingbot-video-dense-1.3b_t2v.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/validate_lora/lingbot-video-dense-1.3b_t2v.py)|
|[Robbyant/lingbot-video-dense-1.3b: TI2V](https://modelscope.cn/models/Robbyant/lingbot-video-dense-1.3b)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_inference/lingbot-video-dense-1.3b_ti2v.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_inference_low_vram/lingbot-video-dense-1.3b_ti2v.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/full/lingbot-video-dense-1.3b_ti2v.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/validate_full/lingbot-video-dense-1.3b_ti2v.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/lora/lingbot-video-dense-1.3b_ti2v.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/validate_lora/lingbot-video-dense-1.3b_ti2v.py)|
|[Robbyant/lingbot-video-dense-1.3b: T2I](https://modelscope.cn/models/Robbyant/lingbot-video-dense-1.3b)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_inference/lingbot-video-dense-1.3b_t2i.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_inference_low_vram/lingbot-video-dense-1.3b_t2i.py)|-|-|-|-|

## Model Inference

The model is loaded via `LingBotVideoPipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details.

The input parameters for `LingBotVideoPipeline` inference include:

* `prompt`: Structured-JSON caption (`dict`) or a plain string describing the content. LingBot-Video is trained on structured captions; the pipeline normalises a `dict` automatically. Released structured captions ship in the example dataset (see [Prompt rewriting](#prompt-rewriting) below).
* `negative_prompt`: Negative prompt describing content that should not appear. `pipe.default_negative_prompt` ships the official T2V/V2V/TI2V negative prompt; `pipe.default_negative_prompt_image` is the T2I variant with temporal terms removed.
* `input_image`: First-frame PIL image for image-to-video (TI2V). The frame is VAE-encoded to a clean latent pinned into the first temporal slot after every scheduler step, so the model only generates the frames that follow. Leave `None` for T2V / V2V / T2I.
* `input_video`: Input video (a list of frames or a `VideoData`) for video-to-video generation, used together with `denoising_strength`.
* `denoising_strength`: Denoising strength in `[0, 1]`, default `1.0`. Lower values keep more of the input video structure. Only effective when `input_video` is provided.
* `height`: Video / image height, default `480`. Must be a multiple of 16.
* `width`: Video / image width, default `480`. Must be a multiple of 16.
* `num_frames`: Number of frames, default `81`. Must satisfy `4k+1` (the VAE compresses time by 4×). Use `num_frames=1` for text-to-image.
* `cfg_scale`: Classifier-free guidance scale, default `3.0`.
* `num_inference_steps`: Number of inference steps, default `40`.
* `sigma_shift`: Flow-matching timestep shift, default `3.0`.
* `seed`: Random seed. Default is `None`, meaning completely random.
* `rand_device`: Device for generating the initial noise, default `"cpu"`.
* `progress_bar_cmd`: Progress bar, default `tqdm`. Can be disabled by setting to `lambda x: x`.

If VRAM is insufficient, please enable [VRAM Management](../Pipeline_Usage/VRAM_management.md). We provide recommended low-VRAM configurations for each task in the example code, see the table in the "Model Overview" section above.

### Prompt rewriting

LingBot-Video is trained on **structured-JSON captions**, not free-form prose. Feeding a flat sentence is out-of-distribution and visibly degrades quality. The pipeline accepts a caption as a `dict` (the format used at training time) or a plain string, and normalises the `dict` internally.

Released structured captions ship in the example dataset (`t2v_example_*.json`, `ti2v_example.json`, `t2i_example.json` under `DiffSynth-Studio/diffsynth_example_dataset`, downloaded automatically by the inference example scripts). Load one with `json.load` and pass the resulting `dict` to the pipeline, or use one as a template.

To turn a brief idea into a structured caption, use the two-stage rewriter shipped under `examples/lingbot_video/model_training/scripts/prompt_rewriter.py` — stage 1 expands the idea into a natural-language caption, stage 2 maps it into structured JSON. The rewriter is a **separate VLM + stage-2 LoRA adapter** and is not downloaded automatically:

| Role | Model ID | Size |
|-|-|-|
| Rewriter base VLM (stage 1 + 2) | [`Qwen/Qwen3.6-27B`](https://modelscope.cn/models/Qwen/Qwen3.6-27B) | ~55 GB |
| Rewriter stage-2 LoRA adapter | [`Robbyant/lingbot-video-rewriter-lora`](https://modelscope.cn/models/Robbyant/lingbot-video-rewriter-lora) | ~0.5 GB |

```python
import os
os.environ["REWRITER_BASE_MODEL"] = "./models/Qwen/Qwen3.6-27B"
os.environ["REWRITER_ADAPTER"] = "./models/Robbyant/lingbot-video-rewriter-lora"

# Run from the repo root so the package-style import resolves.
from examples.lingbot_video.model_training.scripts.prompt_rewriter import rewrite_prompt
caption = rewrite_prompt("a puppy running across a meadow", mode="t2v", duration=5)
video = pipe(prompt=caption, height=480, width=832, num_frames=81, cfg_scale=3.0)
```

Instead of the env vars you can pass `base=` / `adapter=` to `rewrite_prompt`, or drive a hosted / OpenAI-compatible endpoint by passing a custom object exposing `generate(text, image, use_lora)` as `backend=`.

## Model Training

Models in the LingBot-Video series are trained uniformly via [`examples/lingbot_video/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/train.py). The script parameters include:

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
        * `--height`: Height of the video. Must be divisible by 16.
        * `--width`: Width of the video. Must be divisible by 16.
        * `--max_pixels`: Maximum pixel area, images larger than this will be scaled down during dynamic resolution.
        * `--num_frames`: Number of frames in the video. Must satisfy `4k+1`.
* LingBot-Video Specific Parameters
    * `--processor_path`: Path to the Qwen3-VL processor directory (or `model_id:origin_file_pattern`). Used to tokenize prompts.
    * `--first_frame_as_condition`: Enable image-to-video (TI2V) LoRA / full training. Each clip is conditioned on its own first frame: the frame is VAE-encoded to a clean latent pinned into the first temporal slot (and fed to the Qwen3-VL text encoder as vision input), and excluded from the flow-matching loss.
    * `--max_timestep_boundary`: Max timestep boundary as a fraction of the training schedule, in `[0, 1]`.
    * `--min_timestep_boundary`: Min timestep boundary as a fraction of the training schedule, in `[0, 1]`.
    * `--initialize_model_on_cpu`: Whether to initialize models on CPU.

We provide a sample dataset for your testing. You can download it with the following command:

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "lingbot_video/lingbot-video-dense-1.3b_t2v/*" --local_dir ./data/diffsynth_example_dataset
```

Training captions should be **structured-JSON captions** (the same in-distribution format used at inference). If your dataset stores raw prose, rewrite it once offline with [`examples/lingbot_video/model_training/scripts/rewrite_captions.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/scripts/rewrite_captions.py) before training.

We provide recommended training scripts for each task, please refer to the table in "Model Overview" above. For guidance on writing model training scripts, see [Model Training](../Pipeline_Usage/Model_Training.md); for more advanced training algorithms, see [Training Framework Overview](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/en/Training/).
