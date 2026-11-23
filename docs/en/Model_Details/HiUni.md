# HiUni

HiUni is a unified multimodal model open-sourced by ERNIE Research. One checkpoint handles image understanding, text-to-image generation and instruction-based image editing. It is a small model that adds a lightweight pixel-space diffusion head to a vision-language backbone, so images are generated directly as pixels without a VAE or a separate text encoder.

## Installation

Before performing model inference, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information on installation, please refer to [Setup Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Running the following code will load the [ernie-research/HiUni](https://huggingface.co/ernie-research/HiUni) model from Hugging Face for inference. VRAM management is enabled, and the framework automatically controls parameter loading based on available VRAM.

```python
from diffsynth.pipelines.hiuni import HiUniPipeline, ModelConfig
import torch

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

pipe = HiUniPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="ernie-research/HiUni", origin_file_pattern="model-*.safetensors", download_source="huggingface", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="ernie-research/HiUni", origin_file_pattern="tokenizer*.json", download_source="huggingface"),
    processor_config=ModelConfig(model_id="ernie-research/HiUni", origin_file_pattern="preprocessor_config.json", download_source="huggingface"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

prompt = "A cozy reading nook by a rain-streaked window, a ginger cat asleep on a knitted blanket, warm lamp light, film photography."
image = pipe(prompt, height=2048, width=2048, seed=42, num_inference_steps=25, cfg_scale=4.0, shift=4.0)
image.save("image_HiUni.jpg")
```

## Model Overview

|Model ID|Inference|Low VRAM Inference|Full Training|Full Training Validation|LoRA Training|LoRA Training Validation|
|-|-|-|-|-|-|-|
|[ernie-research/HiUni](https://huggingface.co/ernie-research/HiUni)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hiuni/model_inference/HiUni.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/hiuni/model_inference_low_vram/HiUni.py)|-|-|-|-|

## Model Inference

The model is loaded via `HiUniPipeline.from_pretrained`, see [Loading Models](../Pipeline_Usage/Model_Inference.md#loading-models) for details. The weights are hosted on Hugging Face, so the `ModelConfig` entries set `download_source="huggingface"`. `tokenizer_config` and `processor_config` load the tokenizer and the image processor from the repository root and only download their JSON files.

The input parameters for `HiUniPipeline` inference include:

* `prompt`: Text prompt. Acts as the editing instruction in image editing mode.
* `cfg_scale`: Text guidance scale, defaults to 4.0.
* `height`: Output image height, defaults to 1024 for text-to-image. Must be a multiple of 32.
* `width`: Output image width, defaults to 1024 for text-to-image. Must be a multiple of 32.
* `seed`: Random seed, defaults to 42.
* `num_inference_steps`: Number of inference steps, defaults to 25.
* `shift`: Timestep shift affecting sigma computation, defaults to 4.0.
* `edit_image`: Reference image. Passing it switches to image editing mode, defaults to None (text-to-image mode).
* `image_cfg_scale`: Image guidance scale for editing, defaults to 1.0. Values other than 1.0 add a third, unconditional branch.
* `resize_to_edit_image`: Whether to resize the edited image back to the size of `edit_image`, defaults to True.
* `min_pixels`, `max_pixels`: Pixel budget for the reference image and the generated image in editing mode, both default to 262144 (512x512).

### Image Editing

Passing `edit_image` switches to image editing mode. The reference image is encoded by the vision encoder and placed in the user turn before the instruction; its pixels also feed the diffusion head at every step. Unless `height`/`width` are given, the output uses the reference's processed resolution, which keeps its aspect ratio within the `min_pixels`/`max_pixels` budget. With `resize_to_edit_image=True` the result is resized back to the original reference size.

```python
from PIL import Image

edit_image = Image.open("input.jpg").convert("RGB")
image = pipe("Change the background to a snowy mountain.", edit_image=edit_image, seed=42)
```

Guidance combines three predictions when `image_cfg_scale != 1`: `full + cfg_scale * (posi - nega) + image_cfg_scale * (nega - full)`, where `posi` sees the reference and the instruction, `nega` sees only the reference, and `full` sees neither. With the default `image_cfg_scale=1.0` the `full` branch cancels out and is skipped.

### Image Understanding and Text Chat

`pipe.chat` runs autoregressive text generation with the same checkpoint. Pass `images` (a single image or a list) for image understanding, or omit it for plain text chat:

```python
answer = pipe.chat("Describe this image in detail.", images=edit_image)
answer = pipe.chat("Write a short poem about autumn.", think_mode="empty")
```

* `think_mode`: `"think"` (default) lets the model write a `<think>` block before answering, `"empty"` prefills an empty think block, and `"none"` adds no prefill.
* `max_new_tokens`: Maximum number of generated tokens, defaults to 1024.
* `temperature`, `top_k`, `top_p`, `seed`: Sampling options. The default `temperature=0.0` is greedy decoding.
* `min_pixels`, `max_pixels`: Pixel budget for input images, defaulting to 262144 and 4194304.
* `return_raw`: Return the full decoded text including the think block. By default only the answer after `</think>` is returned.

Decoding runs without a KV cache, as in training: every new token recomputes the whole sequence.

### Architecture Notes

* **Shared backbone**: understanding, generation and editing all run through the same language model. The diffusion head reads the backbone's hidden states for the noisy image tokens, so DiffSynth registers a single `hiuni_dit` component.
* **Pixel-space denoising without a VAE**: `latents` holds raster 16x16 RGB patches of shape `(4 * H/32 * W/32, 768)` in `[-1, 1]`. The model predicts the clean image `x0`, guidance is applied to `x0`, and the result is converted to a flow velocity for the scheduler step.
* **Resolution-dependent noise**: the initial noise is scaled by `sqrt(4 * H/32 * W/32 / 256)`, matching training.
