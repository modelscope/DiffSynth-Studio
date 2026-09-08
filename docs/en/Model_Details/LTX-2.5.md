# LTX-2.5

LTX-2.5 is the joint audio-video generation model released by Lightricks. DiffSynth-Studio supports its inference and training through `LTX2AudioVideoPipeline` (the same pipeline class used by LTX-2 / LTX-2.3). Compared with LTX-2.3, LTX-2.5 introduces a fine-tuned Gemma4 12B text encoder (with packed tokenizer assets and dual audio/video connectors), the DiffVAE diffusion video decoder, automatic duration prediction via a Duration Head, and INT8 quantized checkpoints.

## Installation

Before using this project for inference or training, please install DiffSynth-Studio.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information, see [Setup](../Pipeline_Usage/Setup.md). The LTX-2.5 Gemma4 text encoder requires `transformers>=5.8,<5.15`.

## Quickstart

The code below loads [Lightricks/LTX-2.5](https://www.modelscope.cn/models/Lightricks/LTX-2.5) and runs inference. With `auto_duration` enabled, the pipeline first predicts the clip length from the prompt with the Duration Head and then generates audio and video, all in a single `pipe(...)` call.

```python
import torch
from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig
from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cuda",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = LTX2AudioVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="vae/ltx-2.5-video-vae-bf16.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="vae/ltx-2.5-audio-vae-bf16.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="model_patches/ltx-2.5-duration-head-bf16.safetensors", **vram_config),
    ],
    load_duration_head=True,
)
prompt = "A girl is very happy, she is speaking: “I enjoy working with Diffsynth-Studio, it's a perfect framework.”"
negative_prompt = pipe.default_negative_prompt["LTX-2.3"]
video, audio = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    seed=43,
    height=1024, width=1536, frame_rate=24,
    auto_duration=True,
    cfg_scale=1.0, num_inference_steps=8,
    use_distilled_pipeline=True, use_two_stage_pipeline=True,
    tiled=True,
)
write_video_audio_ltx2(video=video, audio=audio, output_path='video.mp4', fps=24, audio_sample_rate=pipe.audio_vocoder.output_sampling_rate)
```

## Models

|Model ID|Extra parameters|Inference|Low-VRAM inference|Full training|Validate full|LoRA training|Validate LoRA|
|-|-|-|-|-|-|-|-|
|[Lightricks/LTX-2.5: OneStagePipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2.5)|-|[code](/examples/ltx2/model_inference/LTX-2.5-T2AV-OneStage.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2.5-T2AV-OneStage.py)|[code](/examples/ltx2/model_training/full/LTX-2.5-T2AV-splited.sh)|[code](/examples/ltx2/model_training/validate_full/LTX-2.5-T2AV.py)|[code](/examples/ltx2/model_training/lora/LTX-2.5-T2AV-splited.sh)|[code](/examples/ltx2/model_training/validate_lora/LTX-2.5-T2AV.py)|
|[Lightricks/LTX-2.5: TwoStagePipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2.5)|-|[code](/examples/ltx2/model_inference/LTX-2.5-T2AV-TwoStage.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2.5-T2AV-TwoStage.py)|-|-|-|-|
|[Lightricks/LTX-2.5: OneStagePipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2.5)|`input_images`,`input_images_indexes`|[code](/examples/ltx2/model_inference/LTX-2.5-I2AV-OneStage.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2.5-I2AV-OneStage.py)|-|-|-|-|
|[Lightricks/LTX-2.5: TwoStagePipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2.5)|`input_images`,`input_images_indexes`|[code](/examples/ltx2/model_inference/LTX-2.5-I2AV-TwoStage.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2.5-I2AV-TwoStage.py)|-|-|-|-|
|[Lightricks/LTX-2.5: TwoStagePipeline-A2V](https://www.modelscope.cn/models/Lightricks/LTX-2.5)|`retake_audio`,`audio_sample_rate`,`stage2_lora_config`|[code](/examples/ltx2/model_inference/LTX-2.5-A2V-TwoStage.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2.5-A2V-TwoStage.py)|-|-|-|-|
|[Lightricks/LTX-2.5: TwoStagePipeline-Retake](https://www.modelscope.cn/models/Lightricks/LTX-2.5)|`retake_video`,`retake_video_regions`,`stage2_lora_config`|[code](/examples/ltx2/model_inference/LTX-2.5-T2AV-TwoStage-Retake.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2.5-T2AV-TwoStage-Retake.py)|-|-|-|-|
|[Lightricks/LTX-2.5: T2A](https://www.modelscope.cn/models/Lightricks/LTX-2.5)|`generate_video=False`|[code](/examples/ltx2/model_inference/LTX-2.5-T2A.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2.5-T2A.py)|-|-|-|-|
|[Lightricks/LTX-2.5: DistilledPipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2.5)|`auto_duration`,`load_duration_head=True`|[code](/examples/ltx2/model_inference/LTX-2.5-T2AV-DistilledPipeline.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2.5-T2AV-DistilledPipeline.py)|-|-|-|-|
|[Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler](https://www.modelscope.cn/models/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler)|`in_context_videos`,`in_context_downsample_factor`|[code](/examples/ltx2/model_inference/LTX-2.5-IC-LoRA-Pixel-Spatial-Upscaler.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2.5-IC-LoRA-Pixel-Spatial-Upscaler.py)|-|-|-|-|
|[Lightricks/LTX-2.5: INT8-ConvRot](https://www.modelscope.cn/models/Lightricks/LTX-2.5)|INT8 DiT + INT8 Gemma4|[code](/examples/ltx2/model_inference/LTX-2.5-T2AV-INT8-ConvRot.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2.5-T2AV-INT8-ConvRot.py)|-|-|-|-|

## Inference

Models are loaded with `LTX2AudioVideoPipeline.from_pretrained`; see [Load Models](../Pipeline_Usage/Model_Inference.md#load-models). LTX-2.5 shares `LTX2AudioVideoPipeline` with LTX-2.3, and the framework detects the model version from the loaded weights.

LTX-2.5 related `from_pretrained` arguments:

* `load_duration_head`: require the Duration Head (needed for automatic duration prediction). Defaults to `False`.
* `gemma_path`: path to the Gemma4 checkpoint, used to load the packed tokenizer assets. When omitted, it is derived from the text encoder entry in `model_configs`.
* `stage2_lora_config`: the stage-2 distilled LoRA used for two-stage inference with the Dev weights.

For the arguments shared with LTX-2.3, see the [LTX-2 documentation](LTX-2.md#inference). The new or LTX-2.5 specific `LTX2AudioVideoPipeline` arguments are:

* `auto_duration`: predict the clip duration from the prompt. Defaults to `False`. When enabled, `num_frames` is not required and the Duration Head must be loaded.
* `auto_duration_min_seconds` / `auto_duration_max_seconds`: lower and upper bounds (seconds) for the predicted duration. Default to 1.0 and 20.0.
* `generate_video`: whether to generate video. Defaults to `True`. Set it to `False` to generate audio only (T2A); the video VAE and latent upsampler are then not required.
* `use_diffusion_vae`: video decoder selection. `None` (default) selects by model version (LTX-2.5 uses the DiffVAE diffusion decoder); `False` uses the ConvVAE convolutional decoder (requires `ltx-2.5-video-vae-conv-bf16.safetensors`).
* Default negative prompt: `pipe.default_negative_prompt["LTX-2.5"]` prefixes the LTX-2/2.3 list with the 2.5-specific tags (`has_subtitles`, `has_blurbox`, `transition from black`, `transition to black`, `speech_ending_short`); all example scripts use this key.
* `input_images` / `input_images_indexes`: keyframe images and their frame indexes. A single first frame gives image-to-video; first and last (or more) frames give keyframe interpolation.
* `retake_audio` / `audio_sample_rate` / `retake_audio_regions`: audio-to-video (A2V) and audio region retake.

Geometry constraints: `num_frames % 8 == 1`; height and width must be multiples of 32 for the one-stage pipeline and multiples of 64 for the two-stage pipeline.

If you are short of VRAM, enable [VRAM management](../Pipeline_Usage/VRAM_management.md). Each example script ships a recommended low-VRAM configuration (FP8 CPU weight offload plus fine-grained VRAM management); see the table above.

## Training

LTX-2.5 shares the training script [`examples/ltx2/model_training/train.py`](/examples/ltx2/model_training/train.py) with LTX-2 / LTX-2.3. The general training arguments are documented in the [LTX-2 documentation](LTX-2.md#training).

The 22B DiT and the 12B Gemma4 encoder do not fit on a single GPU together, so the LTX-2.5 training scripts use the two-stage (splited) scheme:

1. `--task "sft:data_process"`: run text encoding and VAE encoding and cache the results to disk.
2. `--task "sft:train"`: read the cached results and train the DiT only.

Both stages keep the full `--model_id_with_origin_paths` list and use `--fp8_models` to declare the models that are not forwarded in that stage (stage two loads the text encoder and the VAEs in FP8). The training dataset columns are `video,prompt,input_audio,frame_rate`, which map to `--data_file_keys "video,input_audio"` and `--extra_inputs "input_audio"`.

A sample dataset is available for testing:

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "ltx2/LTX-2.3-T2AV-splited/*" --local_dir ./data/diffsynth_example_dataset
```

After training, use `examples/ltx2/model_training/validate_lora/LTX-2.5-T2AV.py` (LoRA) or `examples/ltx2/model_training/validate_full/LTX-2.5-T2AV.py` (full) to run inference with the trained checkpoint. For more details on writing training scripts, see [Model Training](../Pipeline_Usage/Model_Training.md).

## Unsupported features

The following official LTX-2.5 capabilities are not integrated yet:

* DFR (Diffusion Frame Rate)
* Native HDR / EXR output
* HDR IC-LoRA
* Dub-It dubbing
