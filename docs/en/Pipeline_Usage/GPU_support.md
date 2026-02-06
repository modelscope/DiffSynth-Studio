# GPU/NPU Support

`DiffSynth-Studio` supports various GPUs and NPUs. This document explains how to run model inference and training on these devices.

Before you begin, please follow the [Installation Guide](/docs/en/Pipeline_Usage/Setup.md) to install the required GPU/NPU dependencies.

## NVIDIA GPU

All sample code provided by this project supports NVIDIA GPUs by default, requiring no additional modifications.

## AMD GPU

AMD provides PyTorch packages based on ROCm, so most models can run without code changes. A small number of models may not be compatible due to their reliance on CUDA-specific instructions.

## Ascend NPU
### Inference
When using Ascend NPU, you need to replace `"cuda"` with `"npu"` in your code.

For example, here is the inference code for **Wan2.1-T2V-1.3B**, modified for Ascend NPU:

```diff
import torch
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core.device.npu_compatible_device import get_device_name

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
-   "preparing_device": "cuda",
+   "preparing_device": "npu",
    "computation_dtype": torch.bfloat16,
-   "computation_device": "cuda",
+   "computation_device": "npu",
}
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
-   device="cuda",
+   device="npu",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
-   vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
+   vram_limit=torch.npu.mem_get_info(get_device_name())[1] / (1024 ** 3) - 2,
)

video = pipe(
    prompt="Documentary-style photography: a lively puppy running swiftly across lush green grass. The puppy has brownish-yellow fur, upright ears, and an alert, joyful expression. Sunlight bathes its body, making the fur appear exceptionally soft and shiny. The background is an open field with occasional wildflowers, and faint blue sky with scattered white clouds in the distance. Strong perspective captures the motion of the running puppy and the vitality of the surrounding grass. Mid-shot, side-moving viewpoint.",
    negative_prompt="Overly vibrant colors, overexposed, static, blurry details, subtitles, artistic style, painting, still image, overall grayish tone, worst quality, low quality, JPEG artifacts, ugly, distorted, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, fused fingers, motionless scene, cluttered background, three legs, many people in background, walking backward",
    seed=0, tiled=True,
)
save_video(video, "video.mp4", fps=15, quality=5)
```

#### USP(Unified Sequence Parallel)
If you want to use this feature on NPU, please install additional third-party libraries as follows:
```shell
pip install git+https://github.com/feifeibear/long-context-attention.git
pip install git+https://github.com/xdit-project/xDiT.git
```


### Training
NPU startup script samples have been added for each type of model,the scripts are stored in the `examples/xxx/special/npu_training`, for example `examples/wanvideo/model_training/special/npu_training/Wan2.2-T2V-A14B-NPU.sh`.

In the NPU training scripts, NPU specific environment variables that can optimize performance have been added, and relevant parameters have been enabled for specific models.

#### Environment variables
```shell
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```
`expandable_segments:<value>`: Enable the memory pool expansion segment function, which is the virtual memory feature.

```shell
export CPU_AFFINITY_CONF=1
```
Set 0 or not set: indicates not enabling the binding function

1: Indicates enabling coarse-grained kernel binding

2: Indicates enabling fine-grained kernel binding

#### Parameters for specific models
| Model          | Parameter                 | Note              |
|----------------|---------------------------|-------------------|
| Wan 14B series | --initialize_model_on_cpu | The 14B model needs to be initialized on the CPU |
| Qwen-Image series | --initialize_model_on_cpu | The model needs to be initialized on the CPU |