# When Image Models Meet AnimateDiff—Model Combination Technology

We have already witnessed the powerful image generation capabilities of the Stable Diffusion model and its ecosystem models. Now, we introduce a new module: AnimateDiff, which allows us to transfer the capabilities of image models to videos. In this article, we showcase an anime-style video rendering solution built on DiffSynth-Studio: Diffutoon.

## Download Models

The following examples will use many models, so let's download them first.

* An anime-style Stable Diffusion architecture model
* Two ControlNet models
* A Textual Inversion model
* An AnimateDiff model

```python
from diffsynth import download_models

download_models([
    "AingDiffusion_v12",
    "AnimateDiff_v2",
    "ControlNet_v11p_sd15_lineart",
    "ControlNet_v11f1e_sd15_tile",
    "TextualInversion_VeryBadImageNegative_v1.3"
])
```

## Download Video

You can choose any video you like. We use [this video](https://www.bilibili.com/video/BV1iG411a7sQ) as a demonstration. You can download this video file with the following command, but please note, do not use it for commercial purposes without obtaining the commercial copyright from the original video creator.

```
modelscope download --dataset Artiprocher/examples_in_diffsynth data/examples/diffutoon/input_video.mp4 --local_dir ./
```

## Generate Anime

```python
from diffsynth import ModelManager, SDVideoPipeline, ControlNetConfigUnit, VideoData, save_video
import torch

# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models([
    "models/stable_diffusion/aingdiffusion_v12.safetensors",
    "models/AnimateDiff/mm_sd_v15_v2.ckpt",
    "models/ControlNet/control_v11p_sd15_lineart.pth",
    "models/ControlNet/control_v11f1e_sd15_tile.pth",
])

# Build pipeline
pipe = SDVideoPipeline.from_model_manager(
    model_manager,
    [
        ControlNetConfigUnit(
            processor_id="tile",
            model_path="models/ControlNet/control_v11f1e_sd15_tile.pth",
            scale=0.5
        ),
        ControlNetConfigUnit(
            processor_id="lineart",
            model_path="models/ControlNet/control_v11p_sd15_lineart.pth",
            scale=0.5
        )
    ]
)
pipe.prompter.load_textual_inversions(["models/textual_inversion/verybadimagenegative_v1.3.pt"])

# Load video
video = VideoData(
    video_file="data/examples/diffutoon/input_video.mp4",
    height=1536, width=1536
)
input_video = [video[i] for i in range(30)]

# Generate
torch.manual_seed(0)
output_video = pipe(
    prompt="best quality, perfect anime illustration, light, a girl is dancing, smile, solo",
    negative_prompt="verybadimagenegative_v1.3",
    cfg_scale=7, clip_skip=2,
    input_frames=input_video, denoising_strength=1.0,
    controlnet_frames=input_video, num_frames=len(input_video),
    num_inference_steps=10, height=1536, width=1536,
    animatediff_batch_size=16, animatediff_stride=8,
)

# Save video
save_video(output_video, "output_video.mp4", fps=30)
```

## Effect Display

<video width="512" height="256" controls>
  <source src="https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/b54c05c5-d747-4709-be5e-b39af82404dd" type="video/mp4">
Your browser does not support the Video tag.
</video>
