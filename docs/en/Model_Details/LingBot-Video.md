# LingBot-Video

LingBot-Video is a flow-matching text-to-video generation model. This document covers DiffSynth-Studio's inference and LoRA SFT training support for the **Dense-1.3B** text-to-video checkpoint.

The integration is built on the standard DiffSynth pipeline stack:

- **DiT** — `LingBotVideoDiT` (`diffsynth/models/lingbot_video_dit.py`), the video denoiser. The Dense-1.3B build uses a plain FFN; the architecture also supports an MoE FFN.
- **Text encoder** — `LingBotVideoTextEncoder` (Qwen3-VL). Prompts are wrapped in a prompt-enhancement chat template, encoded, and the template-prefix tokens are cropped.
- **VAE** — reuses DiffSynth's `QwenImageVAE` (byte-identical to the LingBot-Video VAE), 8× spatial / 4× temporal compression.
- **Scheduler** — DiffSynth's `FlowMatchScheduler` (Wan template): first-order flow-matching Euler for inference; training uses the full-resolution 1000-step flow-matching schedule.

## Installation

Before using this project for model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

LingBot-Video additionally requires `transformers >= 5.x` (for Qwen3-VL) and `imageio` / `imageio-ffmpeg` for video I/O. For more information about installation, please refer to [Install Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Run the following code to quickly load the [Robbyant/lingbot-video-dense-1.3b](https://modelscope.cn/models/Robbyant/lingbot-video-dense-1.3b) model and perform text-to-video inference. The required files are downloaded automatically the first time the code runs.

> **⚠️ Rewrite your prompt into a structured caption first.** LingBot-Video is trained on **structured-JSON captions**, not free-form prose. The plain sentence in the snippet below runs, but it is out-of-distribution and the result will look noticeably soft / low quality. This is expected model behaviour, not a bug — before doing real inference, turn your idea into the structured caption the model expects. See [Prompt rewriting](#prompt-rewriting-important-for-quality) below; the runnable [`lingbot-video-dense-1.3b.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_inference/lingbot-video-dense-1.3b.py) example defaults to a released structured caption and shows the optional rewrite path at the bottom.

```python
import torch
from diffsynth.utils.data import save_video
from diffsynth.pipelines.lingbot_video import LingBotVideoPipeline, ModelConfig

pipe = LingBotVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="transformer/diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Robbyant/lingbot-video-dense-1.3b", origin_file_pattern="processor/"),
)
video = pipe(
    # A plain sentence is a minimal smoke test only — it is out-of-distribution.
    # For real quality, pass a structured caption instead (see "Prompt rewriting").
    prompt="A playful puppy runs across a lush green meadow, its golden fur shining in the bright sunlight. Wildflowers dot the grass, and a clear blue sky with a few white clouds stretches out behind it. Dynamic side-tracking camera.",
    negative_prompt=pipe.default_negative_prompt,
    height=480, width=832, num_frames=81,
    num_inference_steps=40, cfg_scale=3.0, seed=0,
)
save_video(video, "video.mp4", fps=15, quality=10)
```

**Low VRAM:** set `offload_dtype` / `offload_device` on each `ModelConfig` to enable layer-by-layer offloading; `vram_limit` alone has no effect (it only caps resident VRAM once offloading is on). See the low-VRAM example in the table below.

## Model Overview

| Model ID | Inference | Low VRAM Inference | Full Training | Full Training Validation | LoRA Training | LoRA Training Validation |
|-|-|-|-|-|-|-|
|[Robbyant/lingbot-video-dense-1.3b](https://modelscope.cn/models/Robbyant/lingbot-video-dense-1.3b)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_inference/lingbot-video-dense-1.3b.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_inference_low_vram/lingbot-video-dense-1.3b.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/lora/lingbot-video-dense-1.3b.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/validate_lora/lingbot-video-dense-1.3b.py)|

## Model Inference

The model is loaded via `LingBotVideoPipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details.

Input parameters for `LingBotVideoPipeline` inference include:

* `prompt`: Prompt describing the content appearing in the video. Accepts a structured caption (`dict`) or a plain string; see [Prompt rewriting](#prompt-rewriting-important-for-quality).
* `negative_prompt`: Negative prompt describing content that should not appear in the video, default value is `""`. The official T2V negative prompt ships as `pipe.default_negative_prompt` and can be passed via `negative_prompt=pipe.default_negative_prompt`.
* `input_video`: Input video (a list of frames or a `VideoData`) for video-to-video generation, used together with `denoising_strength`.
* `denoising_strength`: Denoising strength, range 0~1, default value is 1.0. Lower values keep more of the input video structure. Only effective when `input_video` is provided.
* `height`: Video height, default 480. Must be a multiple of 16.
* `width`: Video width, default 480. Must be a multiple of 16.
* `num_frames`: Number of video frames, default 81. Must satisfy `4k+1` (the VAE compresses time by 4×).
* `cfg_scale`: Classifier-free guidance scale, default 6.0. A value of 3.0 is recommended for the Dense-1.3B model.
* `num_inference_steps`: Number of inference steps, default 40.
* `sigma_shift`: Flow-matching timestep shift, default 3.0.
* `seed`: Random seed. Default is `None`, meaning completely random.
* `rand_device`: Device for generating the initial noise, default `"cpu"`.
* `progress_bar_cmd`: Progress bar, default `tqdm`. Can be disabled by setting to `lambda x: x`.

When running low on VRAM, please refer to [VRAM Management](../Pipeline_Usage/VRAM_management.md) to enable VRAM management features.

## Prompt rewriting (important for quality)

LingBot-Video is trained on **structured-JSON captions**, not free-form prose. Feeding a flat sentence is out-of-distribution and visibly degrades quality; feeding the structured caption the model expects restores it. The pipeline accepts a caption as a `dict` or a plain string and normalises it to the exact compact-JSON format the DiT was trained on — a `dict` is serialised automatically, a plain string is passed through unchanged, so existing scripts keep working.

To turn a **brief idea** into that structured caption, use the two-stage rewriter shipped with the examples (`examples/lingbot_video/model_inference/prompt_rewriter.py`): stage 1 *expands* the idea into a natural-language caption, stage 2 *maps* it into structured JSON.

The rewriter is a **separate VLM + stage-2 LoRA adapter** (not the DiT), so it is **not downloaded automatically** — you must fetch both weights yourself before running the rewrite:

| Role | Model ID | Size |
|-|-|-|
| Rewriter base VLM (stage 1 + 2) | [`Qwen/Qwen3.6-27B`](https://modelscope.cn/models/Qwen/Qwen3.6-27B) | ~55 GB |
| Rewriter stage-2 LoRA adapter | [`Robbyant/lingbot-video-rewriter-lora`](https://modelscope.cn/models/Robbyant/lingbot-video-rewriter-lora) | ~0.5 GB |

```shell
# 1. Download the rewriter base VLM and its stage-2 LoRA adapter.
modelscope download --model Qwen/Qwen3.6-27B --local_dir ./models/Qwen/Qwen3.6-27B
modelscope download --model Robbyant/lingbot-video-rewriter-lora --local_dir ./models/Robbyant/lingbot-video-rewriter-lora
```

```python
# 2. Point the rewriter at the downloaded weights, then rewrite and run inference.
import os
os.environ["REWRITER_BASE_MODEL"] = "./models/Qwen/Qwen3.6-27B"
os.environ["REWRITER_ADAPTER"] = "./models/Robbyant/lingbot-video-rewriter-lora"

# The rewriter ships with the inference examples; run from
# examples/lingbot_video/model_inference (or add it to sys.path) for this import.
from prompt_rewriter import rewrite_prompt
caption = rewrite_prompt("a puppy running across a meadow", mode="t2v", duration=5)
video = pipe(prompt=caption, height=480, width=832, num_frames=81, cfg_scale=3.0)
```

Instead of the env vars you can pass `base=` / `adapter=` to `rewrite_prompt`, or skip the local VLM entirely and drive a hosted / OpenAI-compatible endpoint by passing a custom object exposing `generate(text, image, use_lora)` as `backend=`. See the optional rewrite section at the bottom of `examples/lingbot_video/model_inference/lingbot-video-dense-1.3b.py`.

If you don't have the rewriter model, three released LingBot-Video t2v captions ship with the repo as ready-to-use examples: `examples/lingbot_video/model_inference/prompts/t2v_example_{1,2,3}.json`. Load one with `json.load` and pass the resulting `dict` to the pipeline (see the inference example script), or copy one as a template for writing your own structured caption.

## Model Training

LingBot-Video is trained through [`examples/lingbot_video/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/lingbot_video/model_training/train.py), which fine-tunes the DiT with LoRA using the flow-matching SFT objective. The script parameters include:

* General Training Parameters
    * Dataset Basic Configuration
        * `--dataset_base_path`: Root directory of the dataset.
        * `--dataset_metadata_path`: Metadata file path of the dataset (a CSV / JSONL with a `video` column and a `prompt` column).
        * `--dataset_repeat`: Number of times the dataset is repeated in each epoch.
        * `--dataset_num_workers`: Number of processes for each DataLoader.
        * `--data_file_keys`: Field names to be loaded from metadata as files, usually video file paths, separated by `,`.
    * Model Loading Configuration
        * `--model_paths`: Paths of models to be loaded. JSON format.
        * `--model_id_with_origin_paths`: Model IDs with original paths, separated by commas.
    * Training Basic Configuration
        * `--learning_rate`: Learning rate.
        * `--num_epochs`: Number of epochs.
        * `--task`: Training task, default is `sft`.
    * Output Configuration
        * `--output_path`: Model saving path.
        * `--remove_prefix_in_ckpt`: Remove prefix in the state dict of the saved model.
        * `--save_steps`: Interval of training steps to save the model. If left blank, the model is saved once per epoch.
    * LoRA Configuration
        * `--lora_base_model`: Which model to add LoRA to, e.g. `dit`.
        * `--lora_target_modules`: Which layers to add LoRA to.
        * `--lora_rank`: Rank of LoRA.
        * `--lora_checkpoint`: Path of a LoRA checkpoint to resume / continue from.
    * Gradient Configuration
        * `--use_gradient_checkpointing`: Whether to enable gradient checkpointing.
        * `--use_gradient_checkpointing_offload`: Whether to offload gradient checkpointing to memory.
        * `--gradient_accumulation_steps`: Number of gradient accumulation steps.
    * Video Width/Height Configuration
        * `--height`: Height of the video. Must be divisible by 16.
        * `--width`: Width of the video. Must be divisible by 16.
        * `--num_frames`: Number of frames in the video. Must satisfy `4k+1`.
* LingBot-Video Specific Parameters
    * `--processor_path`: Path of the Qwen3-VL processor used by the text encoder.

The launch script first downloads the example video-SFT dataset used across DiffSynth-Studio, then trains on it:

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset \
    --include "wanvideo/Wan2.1-T2V-1.3B/*" --local_dir ./data/diffsynth_example_dataset
```

### Attention-only LoRA (default scope)

The recommended launch script patches LoRA on the joint text+video self-attention only:

```
--lora_base_model "dit"
--lora_target_modules "to_q,to_k,to_v,to_out"
--lora_rank 32
--remove_prefix_in_ckpt "pipe.dit."
```

The MoE / FFN experts (`gate_proj`, `up_proj`, `down_proj`) and the router are left frozen. To also adapt the FFN, add those module names to `--lora_target_modules`.

For best results the `prompt` column should hold **structured-JSON captions** (the same in-distribution format used at inference — see [Prompt rewriting](#prompt-rewriting-important-for-quality)). The pipeline normalises each prompt internally. If your dataset stores raw prose, rewrite it once offline with `examples/lingbot_video/model_training/rewrite_captions.py` before training.

We have written recommended training scripts, please refer to the table in the "Model Overview" section above. For how to write model training scripts, please refer to [Model Training](../Pipeline_Usage/Model_Training.md); for more advanced training algorithms, please refer to [Training Framework Detailed Explanation](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/en/Training/).

## Notes

- The text encoder shares its checkpoint fingerprint with the existing `krea2_text_encoder` (identical Qwen3-VL architecture), so the model loader instantiates both when loading LingBot-Video. This is redundant load time only — the pipeline fetches the correct encoder by name and the other is released.
- The 5D-video VAE encode/decode and its latent normalisation live in the pipeline (`LingBotVideoPipeline.encode_video` / `decode_video`), so `QwenImageVAE` stays byte-identical to its image use elsewhere; the pipeline does not separately re-apply `latents_mean` / `latents_std`.
