# DiffSynth Studio

## Introduction

This branch supports video-to-video translation and is still under development.

## Installation

```
conda env create -f environment.yml
```

## Usage

### Example 1: Toon Shading

https://github.com/Artiprocher/DiffSynth-Studio/assets/35051019/53532f0e-39b1-4791-b920-c975d52ec24a

You can download the models as follows:

* `models/stable_diffusion/flat2DAnimerge_v45Sharp.safetensors`: [link](https://civitai.com/api/download/models/266360?type=Model&format=SafeTensor&size=pruned&fp=fp16)
* `models/AnimateDiff/mm_sd_v15_v2.ckpt`: [link](https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt)
* `models/ControlNet/control_v11p_sd15_lineart.pth`: [link](https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth)
* `models/ControlNet/control_v11f1e_sd15_tile.pth`: [link](https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth)
* `models/Annotators/sk_model.pth`: [link](https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth)
* `models/Annotators/sk_model2.pth`: [link](https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model2.pth)

```python
from diffsynth import ModelManager, SDVideoPipeline, ControlNetConfigUnit, VideoData, save_video, save_frames
import torch


# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_textual_inversions("models/textual_inversion")
model_manager.load_models([
    "models/stable_diffusion/flat2DAnimerge_v45Sharp.safetensors",
    "models/AnimateDiff/mm_sd_v15_v2.ckpt",
    "models/ControlNet/control_v11p_sd15_lineart.pth",
    "models/ControlNet/control_v11f1e_sd15_tile.pth",
])
pipe = SDVideoPipeline.from_model_manager(
    model_manager,
    [
        ControlNetConfigUnit(
            processor_id="lineart",
            model_path="models/ControlNet/control_v11p_sd15_lineart.pth",
            scale=1.0
        ),
        ControlNetConfigUnit(
            processor_id="tile",
            model_path="models/ControlNet/control_v11f1e_sd15_tile.pth",
            scale=0.5
        ),
    ]
)

# Load video
video = VideoData(video_file="data/66dance/raw.mp4", height=1536, width=1536)
input_video = [video[i] for i in range(40*60, 40*60+16)]

# Toon shading
torch.manual_seed(0)
output_video = pipe(
    prompt="best quality, perfect anime illustration, light, a girl is dancing, smile, solo",
    negative_prompt="verybadimagenegative_v1.3",
    cfg_scale=5, clip_skip=2,
    controlnet_frames=input_video, num_frames=16,
    num_inference_steps=10, height=1536, width=1536,
    vram_limit_level=0,
)

# Save images and video
save_frames(output_video, "data/text2video/frames")
save_video(output_video, "data/text2video/video.mp4", fps=16)
```

