# MiniMax-H3-Image

MiniMax-H3-Image is an open-weights image generation and editing model from MiniMax's H3 family. Generation and editing share one checkpoint: the model produces images from text and edits one or more reference images, keeping the two in a single 33.1B diffusion transformer. It generates about 16-megapixel images at every aspect ratio, from 4096×4096 to 6272×2688, without an upscaler, and the checkpoint is distilled to six sampling steps with no classifier-free guidance. The diffusion transformer has the H3 architecture without its audio layers, and it reuses the MiniMax-H3 text encoder unchanged.

## Installation

Before performing model inference and training, please install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For more information on installation, please refer to [Setup Dependencies](../Pipeline_Usage/Setup.md).

## Quick Start

Running the following code generates an image from text and then edits the image it just produced. The script keeps every component on the accelerator, which needs about 135 GB of VRAM in total (66.7 GB for the text encoder, 66.3 GB for the transformer and 2.3 GB for the image VAE); on a smaller GPU use the [low VRAM example](/examples/minimax_h3_image/model_inference_low_vram/MiniMax-H3-Image.py), which streams the text encoder and the VAE from disk and stores the transformer as int8 in host memory. For a 24 GB consumer GPU or a data-center deployment, see [Inference on Different Hardware](#inference-on-different-hardware).

```python
import torch
from diffsynth.pipelines.minimax_h3_image import MiniMaxH3ImagePipeline, ModelConfig
from diffsynth.core.quant import QuantizeConfig
from PIL import Image


pipe = MiniMaxH3ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors"),
        ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="transformer/model*.safetensors"),
        ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="pdd_fused_heads.safetensors"),
        ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)

prompt = (
    "An overhead photograph of a rustic farm-to-table spread on a weathered wooden table: a round loaf of sourdough "
    "on burlap, wedges of cheese on a dark board, a copper bowl of vegetables, vine tomatoes, olives and walnuts, "
    "warm afternoon light raking across the grain of the wood."
)
image = pipe(prompt=prompt, height=4096, width=4096, seed=42)
image.save("MiniMax-H3-Image.jpg")

prompt = "Change the color of the tomato in <Picture 1> to green."
image = pipe(
    prompt=prompt, height=4096, width=4096, seed=43,
    edit_image=[Image.open("MiniMax-H3-Image.jpg")],
)
image.save("MiniMax-H3-Image-Edit.jpg")
```

## Model Overview

|Model ID|Inference|Low VRAM Inference|Full Training|Full Training Validation|LoRA Training|LoRA Training Validation|
|-|-|-|-|-|-|-|
|[MiniMax/MiniMax-H3-Image](https://www.modelscope.cn/models/MiniMax/MiniMax-H3-Image)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_inference/MiniMax-H3-Image.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_inference_low_vram/MiniMax-H3-Image.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/full/MiniMax-H3-Image.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/validate_full/MiniMax-H3-Image.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/lora/MiniMax-H3-Image.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/validate_lora/MiniMax-H3-Image.py)|
|[DiffSynth-Studio/MiniMax-H3-Image-EntroPack](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-Image-EntroPack) (AdaLN-pruned)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_inference/MiniMax-H3-Image-Pruned.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_inference_low_vram/MiniMax-H3-Image-Pruned.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/full/MiniMax-H3-Image-Pruned.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/validate_full/MiniMax-H3-Image-Pruned.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/lora/MiniMax-H3-Image-Pruned.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/validate_lora/MiniMax-H3-Image-Pruned.py)|
|[DiffSynth-Studio/MiniMax-H3-Image-EntroPack](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-Image-EntroPack) (EntroPack)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_inference/MiniMax-H3-Image-Pruned-EntroPack.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_inference_low_vram/MiniMax-H3-Image-Pruned-EntroPack.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/lora/MiniMax-H3-Image-EntroPack.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/validate_lora/MiniMax-H3-Image-EntroPack.py)|
|[Comfy-Org/MiniMax-H3-Image](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3-Image) (pruned)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_inference/MiniMax-H3-Image-Pruned-Comfy.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_inference_low_vram/MiniMax-H3-Image-Pruned-Comfy.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/full/MiniMax-H3-Image-Pruned-Comfy.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/validate_full/MiniMax-H3-Image-Pruned-Comfy.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/lora/MiniMax-H3-Image-Pruned-Comfy.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/validate_lora/MiniMax-H3-Image-Pruned-Comfy.py)|
|[Comfy-Org/MiniMax-H3-Image](https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3-Image) (int8 pruned)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_inference/MiniMax-H3-Image-Int8-Pruned-Comfy.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_inference_low_vram/MiniMax-H3-Image-Int8-Pruned-Comfy.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/lora/MiniMax-H3-Image-Int8-Pruned-Comfy.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/validate_lora/MiniMax-H3-Image-Int8-Pruned-Comfy.py)|

## Model Inference

MiniMax-H3-Image has two workflows: text-to-image, and editing conditioned on reference images. Both run through `MiniMaxH3ImagePipeline`, and the workflow is selected by the `edit_image` argument, so one script covers both. The prompt is used verbatim, with no chat template, and long, complete descriptions work best.

### Text-to-Image

```python
prompt = (
    "An overhead photograph of a rustic farm-to-table spread on a weathered wooden table: a round loaf of sourdough "
    "on burlap, wedges of cheese on a dark board, a copper bowl of vegetables, vine tomatoes, olives and walnuts, "
    "warm afternoon light raking across the grain of the wood."
)
image = pipe(prompt=prompt, height=4096, width=4096, seed=42)
image.save("MiniMax-H3-Image.jpg")
```

`height` and `width` are the size of the returned image and default to 2048. The model generates at 4 megapixels or more: a smaller request is generated on a larger canvas of the same aspect ratio and resized back, so a 1024×1024 request costs as much as a 2048×2048 one. `min_generation_area` sets that threshold in pixels; pass `1024` to always generate at the requested size. Either way the canvas sides are rounded to multiples of 32.

The effective working range of the model is about 4 MP to 20 MP, and it generally gives the best results at around 16 MP:

| Aspect ratio | Size (width x height) |
|-|-|
| 1:1 | 4096x4096 |
| 4:3 / 3:4 | 4736x3552 / 3552x4736 |
| 3:2 / 2:3 | 5024x3360 / 3360x5024 |
| 16:9 / 9:16 | 5472x3072 / 3072x5472 |
| 21:9 | 6272x2688 |

### Image Editing

```python
from PIL import Image

prompt = "Change the color of the tomato in <Picture 1> to green."
image = pipe(
    prompt=prompt, height=4096, width=4096, seed=43,
    edit_image=[Image.open("MiniMax-H3-Image.jpg")],
)
image.save("MiniMax-H3-Image-Edit.jpg")
```

The prompt refers to the reference images in order as `<Picture 1>`, `<Picture 2>`, and so on, and the model accepts up to 16 references. The references keep their own size: each is resized only when its short edge is outside 1024 to 2048 pixels, and never to the size of the generated image. An edit prompt typically names what to preserve, then what to change, then describes the target image in full.

### Sampling

- The checkpoint samples in six steps over a fixed sigma grid, and the transformer holds one output head per step: with `distilled=True` (the default) `num_inference_steps` must be 6.
- The image VAE works on overlapping tiles (256 px, at least 64 px of overlap) and blends them back together. Tiling is part of the released output, so `tiling=True` (the default) reproduces the reference results; `tiling=False` decodes the whole image at once and needs far more VRAM.
- `cfg_scale` and `negative_prompt` are supported. The checkpoint is guidance-distilled, so the defaults (`cfg_scale=1.0` with an empty `negative_prompt`) reproduce the released behaviour; a larger `cfg_scale` runs the unconditional branch as well and combines the two as `uncond + cfg_scale * (cond - uncond)`, which is for experimentation only — the distilled weights were not trained for it.
- `distilled=False` with `num_inference_steps=N` switches to an ordinary flow-match schedule; the transformer's six output heads are then selected by the nearest noise level instead of by step index.

Measured on a single H20 (96 GB): the quick start script keeps the text encoder, the transformer and the VAE resident on the accelerator, which needs about 135 GB of VRAM and therefore does not fit on one 96 GB card. The low VRAM example streams the text encoder and the VAE from disk and stores the transformer and the per-step output heads as int8 in host memory: generating a 4096×4096 image and then editing it takes about 4.5 minutes in total including model loading, and peaks at about 38 GiB of VRAM.

## Inference on Different Hardware

MiniMax-H3-Image has a very large number of parameters, so we provide several inference recipes for different hardware budgets.

### Consumer NVIDIA GPU: RTX 4090

An RTX 4090 has only 24 GB of GPU memory, so the model has to be quantized. We recommend running it with DiffSynth and EntroPack.

Install the Python environment:

```shell
pip install diffsynth==2.2.2
pip install cupy-cuda12x  # or cupy-cuda13x
```

#### Option 1: 24 GB VRAM and 32 GB RAM

INT8 quantization with ConvRot for accuracy, dynamic CPU offload for the DiT, and disk offload streaming for the text encoder; the image quality is almost lossless.

```python
import torch
from diffsynth.pipelines.minimax_h3_image import MiniMaxH3ImagePipeline, ModelConfig

quant_vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
offload_vram_config = {
    # With more host memory, set offload_vram_config = quant_vram_config instead.
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = MiniMaxH3ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="DiffSynth-Studio/MiniMax-H3-Image-EntroPack",
            origin_file_pattern="models/dit_8bit_int8.safetensors",
            origin_config_file="models/dit_8bit_int8.json",
            **quant_vram_config,
        ),
        ModelConfig(
            model_id="DiffSynth-Studio/EntroPackPreQuants",
            origin_file_pattern="models/MiniMax-H3/text_encoder_int8_convrot.safetensors",
            origin_config_file="models/MiniMax-H3/text_encoder_int8_convrot.json",
            **offload_vram_config,
        ),
        ModelConfig(
            model_id="MiniMax/MiniMax-H3-Image",
            origin_file_pattern="pdd_fused_heads.safetensors",
            **quant_vram_config,
        ),
        ModelConfig(
            model_id="DiffSynth-Studio/MiniMax-H3-Image-EntroPack",
            origin_file_pattern="models/vae_int8.safetensors",
            **quant_vram_config,
        ),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="processor/"),
    vram_limit=20,
)
prompt = "A cat is sitting on the table. A book is on the table."
image = pipe(prompt=prompt, height=4096, width=4096, seed=42)
image.save("MiniMax-H3-Image-Pruned-EntroPack.jpg")
```

#### Option 2: 24 GB VRAM and 8 GB RAM

8.5-bit, 5-bit and 2.9-bit mixed quantization that averages 3 bits per parameter, with no offload at all, which is the fastest recipe when generating many images.

```python
import torch
from diffsynth.pipelines.minimax_h3_image import MiniMaxH3ImagePipeline, ModelConfig

pipe = MiniMaxH3ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="DiffSynth-Studio/MiniMax-H3-Image-EntroPack",
            origin_file_pattern="models/dit_extreme_3bpp.safetensors",
            origin_config_file="models/dit_extreme_3bpp.json",
        ),
        ModelConfig(
            model_id="DiffSynth-Studio/EntroPackPreQuants",
            origin_file_pattern="models/MiniMax-H3/text_encoder_3bpp.safetensors",
            origin_config_file="models/MiniMax-H3/text_encoder_3bpp.json",
        ),
        ModelConfig(
            model_id="MiniMax/MiniMax-H3-Image",
            origin_file_pattern="pdd_fused_heads.safetensors",
        ),
        ModelConfig(
            model_id="DiffSynth-Studio/MiniMax-H3-Image-EntroPack",
            origin_file_pattern="models/vae_int8.safetensors",
        ),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="processor/"),
)
prompt = "A cat is sitting on the table. A book is on the table."
image = pipe(prompt=prompt, height=4096, width=4096, seed=42)
image.save("MiniMax-H3-Image-Pruned-EntroPack.jpg")
```

### NVIDIA GPU: H20

The H20 is a low-cost inference GPU that is widely used in data-center servers. With DiffSynth and EntroPack, start by registering a custom attention kernel of DiffSynth that enables the quantized Sage Attention kernel (the code below must run before `import diffsynth`).

```python
import os
os.environ["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] = "sage_attention_h20_fast"
from sageattention import sageattn_qk_int8_pv_fp8_cuda_sm90
from diffsynth.core.attention.common import rearrange_out, rearrange_qkv
from diffsynth.core.attention.registry import register_attention_implementation

def sage_attention_h20_fast(q, k, v, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    required_in_pattern, required_out_pattern = "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = sageattn_qk_int8_pv_fp8_cuda_sm90(
        q, k, v,
        tensor_layout="NHD",
        qk_quant_gran="per_warp",
        pv_accum_dtype="fp32+fp32",
        smooth_k=False,
        is_causal=False,
        sm_scale=scale,
    )
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out

register_attention_implementation(
    "sage_attention_h20_fast",
    sage_attention_h20_fast,
    supports_causal=False,
    supports_window=False,
    supports_mask=False,
)
```

Then run the pipeline with INT8 quantization, ConvRot and compilation:

```python
import torch
from diffsynth.pipelines.minimax_h3_image import MiniMaxH3ImagePipeline, ModelConfig

pipe = MiniMaxH3ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="DiffSynth-Studio/MiniMax-H3-Image-EntroPack",
            origin_file_pattern="models/dit_8bit_int8.safetensors",
            origin_config_file="models/dit_8bit_int8.json",
        ),
        ModelConfig(
            model_id="DiffSynth-Studio/EntroPackPreQuants",
            origin_file_pattern="models/MiniMax-H3/text_encoder_int8_convrot.safetensors",
            origin_config_file="models/MiniMax-H3/text_encoder_int8_convrot.json",
        ),
        ModelConfig(
            model_id="MiniMax/MiniMax-H3-Image",
            origin_file_pattern="pdd_fused_heads.safetensors",
        ),
        ModelConfig(
            model_id="DiffSynth-Studio/MiniMax-H3-Image-EntroPack",
            origin_file_pattern="models/vae_int8.safetensors",
        ),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="processor/"),
)
pipe.compile_pipeline()
prompt = "A cat is sitting on the table. A book is on the table."
image = pipe(prompt=prompt, height=4096, width=4096, seed=42)
image.save("MiniMax-H3-Image-Pruned-EntroPack.jpg")
```

This recipe needs about 50 GB of VRAM. The first inference is slower because of compilation and every inference after it is faster, which suits a deployed online inference service.

## Model Training

MiniMax-H3-Image supports both full fine-tuning and LoRA fine-tuning. The 33.1B transformer takes 66.3 GB in bfloat16 and the text encoder another 66.7 GB, so both training recipes use a two-stage workflow: the first stage (`--task "sft:data_process"`) preprocesses the dataset and caches the prompt embeddings and VAE latents, with the transformer offloaded to disk; the second stage (`--task "sft:train"`) trains on the cached dataset with the text encoder and the VAE offloaded to disk. `--lora_target_modules` is left empty so that the framework picks the modules itself; for this transformer it patches the attention and feed-forward projections of the block stacks plus each block's AdaLN projection, with a rank of 32.

The training data is an image folder plus a `metadata.csv` that holds the image names and prompts:

```csv
image,prompt
1.jpg,"a professional photo of a cute white and brown dog sitting on a wall under cherry blossoms"
```

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3_image/MiniMax-H3-Image/*" --local_dir ./data/diffsynth_example_dataset

# Stage 1: cache the deterministic preprocessing outputs.
accelerate launch examples/minimax_h3_image/model_training/train.py \
  --dataset_base_path ./data/diffsynth_example_dataset/minimax_h3_image/MiniMax-H3-Image \
  --dataset_metadata_path ./data/diffsynth_example_dataset/minimax_h3_image/MiniMax-H3-Image/metadata.csv \
  --data_file_keys "image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3-Image:transformer/model*.safetensors,MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors,MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-Image-split-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task "sft:data_process" \
  --offload_models "MiniMax/MiniMax-H3-Image:transformer/model*.safetensors,MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors"

# Stage 2: train the LoRA on the cached dataset.
accelerate launch examples/minimax_h3_image/model_training/train.py \
  --dataset_base_path "./models/train/MiniMax-H3-Image-split-cache" \
  --data_file_keys "image" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3-Image:transformer/model*.safetensors,MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors,MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-Image-split-lora" \
  --lora_base_model "dit" \
  --lora_target_modules "" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task "sft:train" \
  --offload_models "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors"
```

Editing training is provided as a commented variant at the end of both the LoRA and the full script: uncomment it and drop the text-to-image stages above. Its dataset is `minimax_h3_image/MiniMax-H3-Image-Edit`, whose `metadata.csv` carries an extra `edit_image` column that holds the reference image path, and `edit_image` is listed in both `--data_file_keys` and `--extra_inputs` so that the references are cached alongside the targets.

The two-stage split is what makes training fit on one accelerator: at no point are the transformer, the text encoder and the VAE resident together. The training scripts use `--max_pixels 1048576` so that every sample is resized to about one megapixel, and the validation scripts load the trained weights with the same offloading configuration as the low VRAM example. We provide recommended training scripts, please refer to the table in "Model Overview" above. For guidance on writing model training scripts, see [Model Training](../Pipeline_Usage/Model_Training.md).
