# LTX-2

LTX-2 is a series of audio-video generation models developed by Lightricks.

## Installation

Before using this project for model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information about installation, please refer to [Installation Dependencies](/docs/en/Pipeline_Usage/Setup.md).

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
pipe = LTX2AudioVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized", origin_file_pattern="model-*.safetensors", **vram_config),
        ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-dev.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
prompt = "A girl is very happy, she is speaking: \"I enjoy working with Diffsynth-Studio, it's a perfect framework.\""
negative_prompt = "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
height, width, num_frames = 512, 768, 121
video, audio = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    seed=43,
    height=height,
    width=width,
    num_frames=num_frames,
    tiled=True,
)
write_video_audio_ltx2(
    video=video,
    audio=audio,
    output_path='ltx2_onestage.mp4',
    fps=24,
    audio_sample_rate=24000,
)
```

## Model Overview
|Model ID|Additional Parameters|Inference|Low VRAM Inference|Full Training|Validation After Full Training|LoRA Training|Validation After LoRA Training|
|-|-|-|-|-|-|-|-|
|[Lightricks/LTX-2: OneStagePipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2)||[code](/examples/ltx2/model_inference/LTX-2-T2AV-OneStage.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-OneStage.py)|-|-|-|-|
|[Lightricks/LTX-2: TwoStagePipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2)||[code](/examples/ltx2/model_inference/LTX-2-T2AV-TwoStage.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-TwoStage.py)|-|-|-|-|
|[Lightricks/LTX-2: DistilledPipeline-T2AV](https://www.modelscope.cn/models/Lightricks/LTX-2)||[code](/examples/ltx2/model_inference/LTX-2-T2AV-DistilledPipeline.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2-T2AV-DistilledPipeline.py)|-|-|-|-|
|[Lightricks/LTX-2: OneStagePipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2)|`input_images`|[code](/examples/ltx2/model_inference/LTX-2-I2AV-OneStage.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2-I2AV-OneStage.py)|-|-|-|-|
|[Lightricks/LTX-2: TwoStagePipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2)|`input_images`|[code](/examples/ltx2/model_inference/LTX-2-I2AV-TwoStage.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2-I2AV-TwoStage.py)|-|-|-|-|
|[Lightricks/LTX-2: DistilledPipeline-I2AV](https://www.modelscope.cn/models/Lightricks/LTX-2)|`input_images`|[code](/examples/ltx2/model_inference/LTX-2-I2AV-DistilledPipeline.py)|[code](/examples/ltx2/model_inference_low_vram/LTX-2-I2AV-DistilledPipeline.py)|-|-|-|-|

## Model Inference

Models are loaded through `LTX2AudioVideoPipeline.from_pretrained`, see [Loading Models](/docs/en/Pipeline_Usage/Model_Inference.md#loading-models) for details.

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

If VRAM is insufficient, please enable [VRAM Management](/docs/en/Pipeline_Usage/VRAM_management.md). We provide recommended low VRAM configurations for each model in the example code, see the table in the previous "Supported Inference Scripts" section.

## Model Training

The LTX-2 series models currently do not support training functionality. We will add related support as soon as possible.
