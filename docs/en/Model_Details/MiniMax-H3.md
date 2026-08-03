# MiniMax-H3

MiniMax H3 is a general-purpose omni-modal generation system. It supports unified understanding of multimodal contexts composed of text, images, video, and audio, and can generate videos of up to 2K resolution and up to 15 seconds in length with native stereo audio. Thanks to a system design oriented toward task generalization, H3 already acquires broad multimodal context understanding and generation capabilities during pre-training, and therefore excels at following complex multimodal instructions.

## Installation

Before performing model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information on installation, please refer to [Setup Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Running the following code will quickly load the [DiffSynth-Studio/MiniMax-H3-NF4](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4) NF4-quantized model and perform text-to-video-audio inference. VRAM management is enabled, and the framework automatically controls the loading of model parameters based on available VRAM, requiring a minimum of 7GB VRAM.

```python
import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio

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
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-fl2va-nf4.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-text-encoder-nf4.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="video_vae_nf4.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
    ],
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)
prompt = "A girl is very happy, she is speaking in english: “I enjoy working with Diffsynth-Studio, it's a perfect framework.”"
video, audio = pipe(
    prompt=prompt,
    height=480, width=832, num_frames=124,
    num_inference_steps=50, seed=0,
)
write_video_audio(
    video=video,
    audio=audio,
    output_path="minimax_h3_t2va_quant_nf4.mp4",
    fps=24,
    audio_sample_rate=pipe.audio_vae.sample_rate,
)
print("saved minimax_h3_t2va_quant_nf4.mp4", "frames:", len(video), "audio:", tuple(audio.shape))
```

## Model Overview

|Model ID|Inference|Low VRAM Inference|Full Training|Full Training Validation|LoRA Training|LoRA Training Validation|
|-|-|-|-|-|-|-|
|[MiniMax/MiniMax-H3: TI2VA](https://www.modelscope.cn/models/MiniMax/MiniMax-H3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-TI2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-TI2VA.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-TI2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-TI2VA.py)|
|[MiniMax/MiniMax-H3: Ref2VA](https://www.modelscope.cn/models/MiniMax/MiniMax-H3)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-Ref2VA.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Ref2VA.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-Ref2VA.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Ref2VA.py)|
|[DiffSynth-Studio/MiniMax-H3-NF4: TI2VA](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-TI2VA-nf4.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-TI2VA-nf4.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-TI2VA-nf4.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-TI2VA-nf4.py)|
|[DiffSynth-Studio/MiniMax-H3-NF4: Ref2VA](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference/MiniMax-H3-Ref2VA-nf4.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_inference_low_vram/MiniMax-H3-Ref2VA-nf4.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/lora/MiniMax-H3-Ref2VA-nf4.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/validate_lora/MiniMax-H3-Ref2VA-nf4.py)|

The model weights are split into two partitions: the `FL2VA` partition serves text-to-video-audio and keyframe-guided generation, while the `Ref2VA` partition serves reference-driven generation. The two partitions have different DiT and text encoder weights, so choose the `origin_file_pattern` of the matching partition for your task.

## Model Inference

The model is loaded via `MiniMaxH3Pipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details. Besides `model_configs`, the loading parameters include:

* `processor_config`: `ModelConfig` of the Qwen3-VL processor, used to tokenize the prompt and reference images. Defaults to `FL2VA/processor/`; set it explicitly to `Ref2VA/processor/` when using the `Ref2VA` partition.
* `vram_limit`: VRAM budget for VRAM management (in GB). Leave empty for no limit.

The input parameters for `MiniMaxH3Pipeline` inference include:

* `prompt`: Prompt describing the content of the video as well as the lines spoken by the characters.
* `negative_prompt`: Negative prompt, defaults to `" "`. This model is CFG-distilled, so it takes no effect by default.
* `height`: Height of the video, defaults to 768, must be a multiple of 32.
* `width`: Width of the video, defaults to 1344, must be a multiple of 32.
* `num_frames`: Number of frames, defaults to 124. It is snapped up to the nearest `17n+5`, so the returned clip may be slightly longer than requested. The frame rate is fixed at 24.
* `num_inference_steps`: Number of inference steps, defaults to 50.
* `seed`: Random seed, defaults to 42.
* `rand_device`: Device used to generate the random Gaussian noise tensor, defaults to `"cpu"`. When set to `cuda`, results differ across GPUs.
* `cfg_scale`: Classifier-free guidance scale, defaults to 1.0. This model is CFG-distilled, keeping the default value is recommended.
* `flow_shift`: Flow matching timestep shift for the video modality, defaults to 12.0.
* `audio_flow_shift`: Flow matching timestep shift for the audio modality, defaults to 3.0. Video and audio use two independent sigma schedules.
* `tiled`: Whether to enable tiled VAE inference, defaults to `True`. Enabling it significantly reduces VRAM usage during VAE encoding/decoding, at the cost of a slight numerical error and a small increase in inference time.
* `tile_size`: Tile size during VAE encoding/decoding, defaults to 256.
* `tile_overlap`: Tile overlap during VAE encoding/decoding, defaults to 64.
* `keyframes`: List of keyframe images for keyframe-guided generation. Images are resized onto the target canvas.
* `keyframe_indices`: Frame indices of the keyframes in the video, either `0` (first frame) or `-1` (last frame), corresponding one-to-one with `keyframes`.
* `imgvid_cond_noise_aug`: Noise augmentation anchor timestep for image/video conditions, defaults to 0.999.
* `references`: List of reference conditions in request order. Each element is a dict in one of the following four forms:
    * `{"type": "image", "image": PIL.Image}`
    * `{"type": "video", "video": list[PIL.Image]}` (silent video)
    * `{"type": "audio", "audio": Tensor[C, L], "sample_rate": int}`
    * `{"type": "video_audio", "video": list[PIL.Image], "audio": Tensor[C, L], "sample_rate": int}`

    Video frame lists must ALREADY be at 24fps; the pipeline never resamples the frame rate.
* `audio_cond_noise_aug`: Noise augmentation anchor timestep for reference audio conditions, defaults to 1.0.
* `progress_bar_cmd`: Progress bar, defaults to `tqdm`. Set it to `lambda x: x` to disable the progress bar.

The pipeline returns a `(video, audio)` tuple, where the video is a list of PIL images and the audio is a waveform tensor. Use `diffsynth.utils.data.audio_video.write_video_audio` to mux them into an MP4:

```python
write_video_audio(video=video, audio=audio, output_path="video.mp4", fps=24, audio_sample_rate=pipe.audio_vae.sample_rate)
```

If VRAM is insufficient, please enable [VRAM management](../Pipeline_Usage/VRAM_management.md). We provide a recommended low VRAM configuration for each model in the example code, see the table in "Model Overview" above. We also provide NF4-quantized weights to further reduce VRAM requirements; the corresponding scripts are listed in the same table.

## Model Training

Models in the MiniMax-H3 series are trained uniformly via [`examples/minimax_h3/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3/model_training/train.py). The script parameters include:

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
        * `--height`: Height of the video. Leave `height` and `width` empty to enable dynamic resolution.
        * `--width`: Width of the video. Leave `height` and `width` empty to enable dynamic resolution.
        * `--max_pixels`: Maximum pixel area, images larger than this will be scaled down during dynamic resolution.
        * `--num_frames`: Number of frames for the video.
* MiniMax-H3 Specific Parameters
    * `--processor_path`: Path of the Qwen3-VL processor, supports the `model_id:origin_file_pattern` form, used to tokenize the prompt.
    * `--initialize_model_on_cpu`: Whether to initialize models on CPU.

We provide an example dataset for testing, which can be downloaded with the following command:

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --local_dir ./data/diffsynth_example_dataset
```

The LoRA training scripts use a two-stage workflow: first preprocess and cache the dataset with `--task "sft:data_process"`, then run the actual training with `--task "sft:train"`. LoRA is applied to the `qkv_proj,out_proj` modules of the DiT by default, with a rank of 32. Keyframe-guided (FL2VA) training appends `input_image,end_image` to `--extra_inputs`, taking the first and last frame of the training video as conditions respectively.

We provide recommended training scripts for each model, please refer to the table in "Model Overview" above. For guidance on writing model training scripts, see [Model Training](../Pipeline_Usage/Model_Training.md); for more advanced training algorithms, see [Training Framework Overview](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/en/Training/).
