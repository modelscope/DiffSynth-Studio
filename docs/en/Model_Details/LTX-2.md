# LTX-2

LTX-2 is a series of audio-video generation models developed by Lightricks.

## Installation

Before using this project for model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information about installation, please refer to [Installation Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Run the following code to quickly load the [Lightricks/LTX-2](https://www.modelscope.cn/models/Lightricks/LTX-2) model and perform inference. VRAM management has been enabled, and the framework will automatically control model parameter loading based on remaining VRAM. It can run with a minimum of 8GB VRAM.

```python
import torch
from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig
from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2

vram_config = {
    "offload_dtype": torch.float8_e5m2,
    "offload_device": "cpu",
    "onload_dtype": torch.float8_e5m2,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e5m2,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
"""
Offical model repo: https://www.modelscope.cn/models/Lightricks/LTX-2
Repackaged model repo: https://www.modelscope.cn/models/DiffSynth-Studio/LTX-2-Repackage
For base models of LTX-2, offical checkpoint (with model config ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-dev.safetensors"))
and repackaged checkpoints (with model config ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="*.safetensors")) are both supported.
We have repackeged the official checkpoints in DiffSynth-Studio/LTX-2-Repackage repo to support separate loading of different submodules,
and avoid redundant memory usage when users only want to use part of the model.
"""
# use the repackaged modelconfig from "DiffSynth-Studio/LTX-2-Repackage" to avoid redundant model loading
pipe = LTX2AudioVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized", origin_file_pattern="model-*.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="transformer.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="text_encoder_post_modules.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="video_vae_decoder.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="audio_vae_decoder.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="audio_vocoder.safetensors", **vram_config),
        ModelConfig(model_id="DiffSynth-Studio/LTX-2-Repackage", origin_file_pattern="video_vae_encoder.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-spatial-upscaler-x2-1.0.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized"),
    stage2_lora_config=ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-distilled-lora-384.safetensors"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

# use the following modelconfig if you want to initialize model from offical checkpoints from "Lightricks/LTX-2"
# pipe = LTX2AudioVideoPipeline.from_pretrained(
#     torch_dtype=torch.bfloat16,
#     device="cuda",
#     model_configs=[
#         ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized", origin_file_pattern="model-*.safetensors", **vram_config),
#         ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-dev.safetensors", **vram_config),
#         ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-spatial-upscaler-x2-1.0.safetensors", **vram_config),
#     ],
#     tokenizer_config=ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized"),
#     stage2_lora_config=ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-distilled-lora-384.safetensors"),
#     vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
# )

prompt = "A girl is very happy, she is speaking: \"I enjoy working with Diffsynth-Studio, it's a perfect framework.\""
negative_prompt = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)
height, width, num_frames = 512 * 2, 768 * 2, 121
video, audio = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    seed=43,
    height=height,
    width=width,
    num_frames=num_frames,
    tiled=True,
    use_two_stage_pipeline=True,
)
write_video_audio_ltx2(
    video=video,
    audio=audio,
    output_path='ltx2_twostage.mp4',
    fps=24,
    audio_sample_rate=24000,
)
```

## Model Overview
|Model ID|Additional Parameters|Inference|Low VRAM Inference|Full Training|Validation After Full Training|LoRA Training|Validation After LoRA Training|
|-|-|-|-|-|-|-|-|
|[Lightricks/LTX-2: OneStagePipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2)||[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-T2AV-OneStage.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-OneStage.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_training/full/LTX-2-T2AV-splited.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_training/validate_full/LTX-2-T2AV.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_training/lora/LTX-2-T2AV-splited.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_training/validate_lora/LTX-2-T2AV.py)|
|[Lightricks/LTX-2-19b-IC-LoRA-Union-Control](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-IC-LoRA-Union-Control)|`in_context_videos`,`in_context_downsample_factor`|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-T2AV-IC-LoRA-Union-Control.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-IC-LoRA-Union-Control.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_training/lora/LTX-2-T2AV-IC-LoRA-splited.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_training/validate_lora/LTX-2-T2AV-IC-LoRA.py)|
|[Lightricks/LTX-2-19b-IC-LoRA-Detailer](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-IC-LoRA-Detailer)|`in_context_videos`,`in_context_downsample_factor`|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-T2AV-IC-LoRA-Detailer.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-IC-LoRA-Detailer.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_training/lora/LTX-2-T2AV-IC-LoRA-splited.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_training/validate_lora/LTX-2-T2AV-IC-LoRA.py)|
|[Lightricks/LTX-2: TwoStagePipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2)||[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-T2AV-TwoStage.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-TwoStage.py)|-|-|-|-|
|[Lightricks/LTX-2: DistilledPipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2)||[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-T2AV-DistilledPipeline.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-DistilledPipeline.py)|-|-|-|-|
|[Lightricks/LTX-2: OneStagePipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2)|`input_images`|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-I2AV-OneStage.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-I2AV-OneStage.py)|-|-|-|-|
|[Lightricks/LTX-2: TwoStagePipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2)|`input_images`|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-I2AV-TwoStage.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-I2AV-TwoStage.py)|-|-|-|-|
|[Lightricks/LTX-2: DistilledPipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2)|`input_images`|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-I2AV-DistilledPipeline.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-I2AV-DistilledPipeline.py)|-|-|-|-|
|[Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In)||[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Dolly-In.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Dolly-In.py)|-|-|-|-|
|[Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out)||[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Dolly-Out.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Dolly-Out.py)|-|-|-|-|
|[Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left)||[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Dolly-Left.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Dolly-Left.py)|-|-|-|-|
|[Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right)||[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Dolly-Right.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Dolly-Right.py)|-|-|-|-|
|[Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up)||[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Jib-Up.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Jib-Up.py)|-|-|-|-|
|[Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down)||[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Jib-Down.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Jib-Down.py)|-|-|-|-|
|[Lightricks/LTX-2-19b-LoRA-Camera-Control-Static](https://www.modelscope.cn/models/Lightricks/LTX-2-19b-LoRA-Camera-Control-Static)||[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference/LTX-2-T2AV-Camera-Control-Static.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-Camera-Control-Static.py)|-|-|-|-|

## Model Inference

Models are loaded through `LTX2AudioVideoPipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details.

Input parameters for `LTX2AudioVideoPipeline` inference include:

* `prompt`: Prompt describing the content appearing in the video.
* `negative_prompt`: Negative prompt describing content that should not appear in the video, default value is `""`.
* `cfg_scale`: Classifier-free guidance parameter, default value is 3.0.
* `input_images`: List of input images for image-to-video generation.
* `input_images_indexes`: Frame index list of input images in the video.
* `input_images_strength`: Strength of input images, default value is 1.0.
* `denoising_strength`: Denoising strength, range is 0～1, default value is 1.0.
* `seed`: Random seed. Default is `None`, which means completely random.
* `rand_device`: Computing device for generating random Gaussian noise matrix, default is `"cpu"`. When set to `cuda`, different results will be generated on different GPUs.
* `height`: Video height, must be a multiple of 32 (single-stage) or 64 (two-stage).
* `width`: Video width, must be a multiple of 32 (single-stage) or 64 (two-stage).
* `num_frames`: Number of video frames, default value is 121, must be a multiple of 8 + 1.
* `num_inference_steps`: Number of inference steps, default value is 40.
* `tiled`: Whether to enable VAE tiling inference, default is `True`. When set to `True`, it can significantly reduce VRAM usage during VAE encoding/decoding stages, with slight errors and minor inference time extension.
* `tile_size_in_pixels`: Pixel tiling size during VAE encoding/decoding stages, default is 512.
* `tile_overlap_in_pixels`: Pixel tiling overlap size during VAE encoding/decoding stages, default is 128.
* `tile_size_in_frames`: Frame tiling size during VAE encoding/decoding stages, default is 128.
* `tile_overlap_in_frames`: Frame tiling overlap size during VAE encoding/decoding stages, default is 24.
* `use_two_stage_pipeline`: Whether to use two-stage pipeline, default is `False`.
* `use_distilled_pipeline`: Whether to use distilled pipeline, default is `False`.
* `progress_bar_cmd`: Progress bar, default is `tqdm.tqdm`. Can be set to `lambda x:x` to hide the progress bar.

If VRAM is insufficient, please enable [VRAM Management](../Pipeline_Usage/VRAM_management.md). We provide recommended low VRAM configurations for each model in the example code, see the table in the previous "Supported Inference Scripts" section.

## Model Training

LTX-2 series models are uniformly trained through [`examples/ltx2/model_training/train.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/ltx2/model_training/train.py), and the script parameters include:

* General Training Parameters
    * Dataset Basic Configuration
        * `--dataset_base_path`: Root directory of the dataset.
        * `--dataset_metadata_path`: Metadata file path of the dataset.
        * `--dataset_repeat`: Number of times the dataset is repeated in each epoch.
        * `--dataset_num_workers`: Number of processes for each DataLoader.
        * `--data_file_keys`: Field names to be loaded from metadata, usually image or video file paths, separated by `,`.
    * Model Loading Configuration
        * `--model_paths`: Paths of models to be loaded. JSON format.
        * `--model_id_with_origin_paths`: Model IDs with original paths, e.g., `"Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors"`. Separated by commas.
        * `--extra_inputs`: Extra input parameters required by the model Pipeline, e.g., extra parameters when training image editing models, separated by `,`.
        * `--fp8_models`: Models loaded in FP8 format, consistent with `--model_paths` or `--model_id_with_origin_paths` format. Currently only supports models whose parameters are not updated by gradients (no gradient backpropagation, or gradients only update their LoRA).
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
    * Video Width/Height Configuration
        * `--height`: Height of the video. Leave `height` and `width` blank to enable dynamic resolution.
        * `--width`: Width of the video. Leave `height` and `width` blank to enable dynamic resolution.
        * `--max_pixels`: Maximum pixel area of video frames. When dynamic resolution is enabled, video frames with resolution larger than this value will be downscaled, and video frames with resolution smaller than this value will remain unchanged.
        * `--num_frames`: Number of frames in the video.
* LTX-2 Series Specific Parameters
    * `--tokenizer_path`: Path of the tokenizer, applicable to text-to-video models, leave blank to automatically download from remote.
    * `--frame_rate`: frame rate of the training videos.

We have built a sample video dataset for your testing. You can download this dataset with the following command:

```shell
modelscope download --dataset DiffSynth-Studio/example_video_dataset --local_dir ./data/example_video_dataset
```

We have written recommended training scripts for each model, please refer to the table in the "Model Overview" section above. For how to write model training scripts, please refer to [Model Training](../Pipeline_Usage/Model_Training.md); for more advanced training algorithms, please refer to [Training Framework Detailed Explanation](https://github.com/modelscope/DiffSynth-Studio/tree/main/docs/en/Training/).
