# Model Quantization

Quantization reduces VRAM usage by lowering the numerical precision of model weights, allowing large models to run on smaller GPUs. `DiffSynth-Studio` provides a unified quantization entry point `QuantizeConfig`, supporting multiple quantization backends such as bitsandbytes, torchao, and comfy-kitchen, as well as online quantization, loading pre-quantized weights, mixed quantization, and quantization + LoRA training.

This document uses `Z-Image` as an example. If you want to use `diffsynth.core.quant` in your own codebase, refer to the [`diffsynth.core.quant` API documentation](../API_Reference/core/quant.md).

> **Difference between quantization and FP8 in VRAM management**
>
> The FP8 in [VRAM management](./VRAM_management.md) controls the storage precision of weights in VRAM through parameters such as `offload_dtype` / `onload_dtype`. It applies to all parameters and requires no third-party libraries, but only supports simple FP8 conversion.
>
> The quantization in this document (`QuantizeConfig`) is a dedicated scheme for `nn.Linear` layers, supporting finer formats such as NF4, INT8, INT4, MXFP4, and NVFP4. It can save/load quantized weights and supports activation quantization and quantization + LoRA training. The two can be combined.

## Installation

Different quantization backends require the corresponding third-party libraries:

| Backend | Install Command | Project Page |
| --- | --- | --- |
| bitsandbytes | `pip install bitsandbytes` | [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) |
| torchao | `pip install torchao>=0.16` | [torchao](https://github.com/pytorch/ao) |
| comfy-kitchen | `pip install comfy-kitchen` | [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) |

Install all at once: `pip install "diffsynth[quant]"`

## Quick Start

Pass `quantize` to any `ModelConfig` to enable online quantization for that model. The following code loads Z-Image's DiT with NF4 quantization:

```python
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
from diffsynth.core.quant import QuantizeConfig
import torch

pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="Tongyi-MAI/Z-Image", origin_file_pattern="transformer/*.safetensors",
            quantize=QuantizeConfig(method="bitsandbytes_nf4"),
        ),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
)
prompt = "A delicate portrait of an underwater girl, blue dress flowing, hair gently drifting, light and shadow clear, surrounded by bubbles, serene expression, exquisite details, dreamlike and beautiful."
image = pipe(prompt=prompt, seed=42, num_inference_steps=50, cfg_scale=4)
image.save("image_z_image_nf4.jpg")
```

## Supported Quantization Methods

The following are all built-in quantization methods. `method` is the name passed to `QuantizeConfig`. The naming follows the `W<weight bits>A<activation bits>` convention: `w8a16` means only the weights are quantized (weight-only), while `w8a8` means both weights and activations are quantized.

| method | Backend | Weight / Activation | Serializable | LoRA Training |
| --- | --- | --- | --- | --- |
| `bitsandbytes_nf4` | bitsandbytes | NF4 / none | ✅ | ✅ |
| `bitsandbytes_fp4` | bitsandbytes | FP4 / none | ✅ | ✅ |
| `torchao_int8_w8a16` | torchao | INT8 / none | ✅ | ✅ |
| `torchao_fp8_w8a16` | torchao | FP8 / none | ✅ | ✅ |
| `torchao_int4_w4a16` | torchao | INT4 / none | ✅ | ❌ |
| `torchao_nvfp4_w4a16` | torchao | NVFP4 / none | ✅ | ✅ |
| `torchao_int8_w8a8` | torchao | INT8 / INT8 dynamic | ✅ | ❌ |
| `torchao_fp8_w8a8` | torchao | FP8 / FP8 dynamic | ✅ | ❌ |
| `torchao_int4_w4a8` | torchao | INT4 / FP8 dynamic | ✅ | ❌ |
| `torchao_mxfp8_w8a8` | torchao | MXFP8 / MXFP8 | ✅ | ❌ |
| `torchao_mxfp4_w4a4` | torchao | MXFP4 / MXFP4 | ✅ | ❌ |
| `torchao_nvfp4_w4a4` | torchao | NVFP4 / NVFP4 | ✅ | ❌ |
| `comfy_kitchen_int8_w8a8` | comfy_kitchen | INT8 / INT8 dynamic | ✅ | ✅ |
| `comfy_kitchen_fp8_w8a8` | comfy_kitchen | FP8 E4M3 / FP8 | ✅ | ✅ |

Some notes:

- **Activation quantization** (`w8a8` / `w4a4`) quantizes activations in addition to compressing weights. On hardware that supports the corresponding low-precision matrix multiplication, this can deliver real speedups, while weight-only schemes typically only save VRAM.
- **LoRA training**: only methods marked ✅ in the "LoRA Training" column can be used for quantization + LoRA training.
- `comfy_kitchen_*` methods read and write ComfyUI's quantized weight format, interoperable with the ComfyUI ecosystem. comfy-kitchen requires CUDA 13.0 or later.
- Formats such as MXFP8 / MXFP4 / NVFP4 have compute hardware requirements; see the [torchao](https://github.com/pytorch/ao) documentation for compatibility details.

You can query all available methods and their parameters in code:

```python
from diffsynth.core.quant import describe_quant_method, QUANT_METHODS, backends

backends.load_all_backends()
print(sorted(QUANT_METHODS))          # all registered method names

describe_quant_method("bitsandbytes_nf4")
```

The output is as follows. `backend_config_kwargs (user-tunable)` lists the adjustable parameters of the method and their default values; these parameters determine the quantization behavior, and you can modify them as needed. For the torchao backend, some parameters are passed directly to torchao's own config (e.g. `Int8WeightOnlyConfig`):

> Unless you know what these parameters mean, we recommend keeping the default values.

```
method: bitsandbytes_nf4
backend: bitsandbytes
detail: 4bit, nf4, weight-only
backend config: diffsynth.core.quant.backends.bitsandbytes.BitsAndBytesNF4Config
backend_config_kwargs (user-tunable):
  compress_statistics = True
  blocksize           = None
  quant_storage       = torch.uint8
pinned by method (not overridable):
  quant_type = 'nf4'
```

## QuantizeConfig in Detail

`QuantizeConfig` describes "which method to use, which layers to quantize, and how to run after quantization":

- **`method`**: the quantization method name, see the table above. Required.
- **`mode`**: how quantized layers run.
    - `"dynamic"` (default): keeps the quantized Linears, dequantizing on demand at forward time, with low VRAM usage.
    - `"dequant_once"`: after quantization, restores all quantized layers to plain fp `nn.Linear` once (keeping the quantization error). Suitable for scenarios that need standard `nn.Linear`; no longer saves VRAM.
- **`target_modules` / `exclude_modules`**: filter the `nn.Linear` layers to quantize by layer name, given as lists. A layer matches if its full dotted name equals an entry, or ends with `"." + entry` (e.g. `"img_mod.1"` matches `transformer_blocks.0.img_mod.1`).
- **`backend_config_kwargs`**: a dict of parameters passed to the backend config, determining the quantization behavior (for the torchao backend, some parameters are passed directly to torchao's own config). Use `describe_quant_method(method)` to query the available parameters.
- **`load_prequantized`**: set to `True` when the checkpoint already holds quantized weights, so they are loaded directly (see below).

Example: exclude quantization-sensitive layers and adjust NF4 backend parameters:

```python
from diffsynth.core.quant import QuantizeConfig

quantize = QuantizeConfig(
    method="bitsandbytes_nf4",
    mode="dynamic",
    exclude_modules=["time_embedder.proj_in", "time_embedder.proj_out", "proj_out"],
    backend_config_kwargs={"compress_statistics": False},
)
```

Activation quantization methods are used in exactly the same way, just with a different `method`:

```python
quantize = QuantizeConfig(method="comfy_kitchen_int8_w8a8", backend_config_kwargs={"convrot_groupsize": 128})
```

## Loading Pre-quantized Weights

Besides online quantization, you can also load checkpoints that are already quantized, avoiding the quantization overhead on every load.

For officially released quantized models (e.g. `ideogram-ai/ideogram-4-nf4`), the quantization info is already written in the config, so you can load them like ordinary models:

```python
ModelConfig(model_id="ideogram-ai/ideogram-4-nf4", origin_file_pattern="transformer/diffusion_pytorch_model.safetensors")
```

For checkpoints you saved yourself (see the next section), pass `quantize` explicitly with `load_prequantized=True` when loading. The `method` and `exclude_modules` must match those used when saving:

```python
from diffsynth.core.quant import QuantizeConfig

ModelConfig(
    path="models/z-image-nf4/transformer.safetensors",
    quantize=QuantizeConfig(method="bitsandbytes_nf4", load_prequantized=True),
)
```

## Saving a Quantized Model

To save the result of an online quantization for reuse, use `save_quantized_model`:

```python
from diffsynth.core.loader import ModelConfig
from diffsynth.core.quant import QuantizeConfig
from diffsynth.utils.quant.serialization import save_quantized_model

model_config = ModelConfig(
    model_id="Tongyi-MAI/Z-Image",
    origin_file_pattern="transformer/*.safetensors",
    quantize=QuantizeConfig(method="bitsandbytes_nf4"),
)
save_quantized_model(model_config, "models/z-image-nf4/transformer.safetensors")
```

It downloads and loads the original fp weights, performs the quantization, and saves the quantized state dict as `.safetensors`. After saving, you can load it as described in the previous section.

## Mixed Quantization

Different layers have different sensitivity to quantization. `MixedQuantizeConfig` allows applying different methods to different layer sets. For example, use INT8 for precision-sensitive modulation layers and NF4 for the rest:

```python
from diffsynth.core.quant import QuantizeConfig, MixedQuantizeConfig

mod_layers = ["img_mod.1", "txt_mod.1", "norm_out.linear", "img_in", "txt_in", "proj_out"]
quantize = MixedQuantizeConfig(configs=[
    QuantizeConfig(method="bitsandbytes_nf4", exclude_modules=mod_layers),
    QuantizeConfig(method="torchao_int8_w8a16", target_modules=mod_layers),
])
```

The layer sets matched by the sub-configs must not overlap. All sub-configs must share the same `mode`. `MixedQuantizeConfig` exposes the same interface as `QuantizeConfig`, and can be passed directly to `ModelConfig(quantize=...)` or `save_quantized_model`.

> When loading a pre-quantized mixed checkpoint, set `load_prequantized=True` on the `MixedQuantizeConfig` itself, not on the sub-configs.

## Quantization + LoRA Training

In most cases, a quantized model does not support training, but it does support LoRA training with the base model frozen, enabling training of large models with very little VRAM.

Methods usable for quantization + LoRA training are listed in the last column of the [methods table](#supported-quantization-methods). There are two ways to do it.

### Approach 1: Train with a pre-quantized base model

The training script points `--model_id_with_origin_paths` at the pre-quantized model:

```bash
accelerate launch examples/.../train.py \
  --model_id_with_origin_paths "DiffSynth-Studio/MiniMax-H3-NF4:minimax-h3-fl2va-nf4.safetensors,..." \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --output_path "./models/train/xxx-nf4"
```

### Approach 2: Online quantization with `--quant_options`

If no pre-quantized weights are available, use `--quant_options` to quantize the loaded models online at training startup. The value is a semicolon-separated list of entries, each formatted as `<model_string>:<method>[/<exclude_modules>]`:

- `<model_string>`: the model to quantize; must match the entry in `--model_paths` / `--model_id_with_origin_paths` exactly.
- `<method>`: the quantization method name, see the [methods table](#supported-quantization-methods).
- `<exclude_modules>`: optional, a comma-separated list of layer names kept in full precision.

Here is quantized LoRA training for Z-Image-Turbo (full script at `examples/z_image/model_training/special/quant_training/Z-Image-Turbo-bitsandbytes_nf4.sh`):

```bash
accelerate launch examples/z_image/model_training/train.py \
  --model_id_with_origin_paths "Tongyi-MAI/Z-Image-Turbo:transformer/*.safetensors,Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,Tongyi-MAI/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors" \
  --quant_options "Tongyi-MAI/Z-Image-Turbo:transformer/*.safetensors:bitsandbytes_nf4;Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors:bitsandbytes_nf4" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,w1,w2,w3" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --output_path "./models/train/Z-Image-Turbo_quant_lora"
```

Above, NF4 quantization is enabled for both the DiT and the text encoder. Modules that do not participate in training, such as `text_encoder` and `vae`, can be quantized freely; the trained `dit` can only be quantized under LoRA training, and the method must support LoRA training — specifying a non-differentiable method makes training fail immediately with an error.

With `exclude_modules` to keep quantization-sensitive layers in full precision:

```bash
  --quant_options "MiniMaxAI/MiniMax-H3:FL2VA/transformer/model*.safetensors:bitsandbytes_nf4/time_embedder.proj_in,time_embedder.proj_out,video_patch_proj,audio_patch_proj"
```

> `--quant_options` always uses `mode="dynamic"` and does not expose advanced options such as `backend_config_kwargs` or mixed quantization. For those, use Approach 1: save quantized weights with `save_quantized_model` first, then train with the pre-quantized base model.

### Shared notes

- During training, the quantized base model stays frozen; only the LoRA branches are updated, so what gets saved is fp-precision LoRA weights.
- For inference, load as "quantized base model + LoRA": first load the quantized base model as described in [Loading Pre-quantized Weights](#loading-pre-quantized-weights), then `pipe.load_lora(pipe.dit, "epoch-x.safetensors")`.
- Prefer Approach 1 for large models: online quantization has to load the full fp weights first, which makes startup slow and peak memory high.

## Custom Quantization Backends

If the built-in methods don't meet your needs, you can implement your own quantization backend. See [Integrating a Quantization Backend](../Developer_Guide/Integrating_Quantization_Backend.md) for the full walkthrough with a runnable toy INT9 example, and the [`diffsynth.core.quant` API documentation](../API_Reference/core/quant.md#extension-interface-custom-backends) for the full interface signatures and contracts.

## Combining Quantization with VRAM Management

Quantization and [VRAM management](./VRAM_management.md) address different levels of the problem and can be enabled together:

- Quantization reduces **the storage size of each layer**, e.g. NF4 is about 1/4 of bf16.
- VRAM management decides **which layers stay in VRAM right now**, loading the rest from RAM/disk on demand.

Combining both can further reduce the VRAM required for inference: first compress weights to 4bit/8bit, then use `vram_limit` to split the compressed model between VRAM and RAM.

```python
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
from diffsynth.core.quant import QuantizeConfig
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
pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="Tongyi-MAI/Z-Image", origin_file_pattern="transformer/*.safetensors",
            quantize=QuantizeConfig(method="bitsandbytes_nf4"), **vram_config,
        ),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors", **vram_config),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
```

Two notes:

- Parameters such as `offload_dtype` / `onload_dtype` in `vram_config` apply to unquantized parameters; the storage precision of quantized layers is determined by the quantization method and is unaffected by these parameters.
- **Disk Offload is incompatible with online quantization.** Disk Offload reads parameters from disk layer by layer, requiring that the quantized parameters are already saved on disk, so you cannot run Disk Offload first and then quantize online. To combine quantization with Disk Offload, first save the quantized weights following the [Best Practices](#best-practices) workflow, then load the pre-quantized checkpoint.

## Best Practices

Taking MiniMax-H3 as an example, this section shows the complete workflow from saving quantized weights to loading for inference.

### Step 1: Save the Quantized Weights

The MiniMax-H3 FL2VA base model is about 66G (bf16), and online NF4 quantization would be very slow. We recommend quantizing and saving once, then loading repeatedly. `save_quantized_model` returns the hash of the saved file:

```python
from diffsynth.core.loader import ModelConfig
from diffsynth.core.quant import QuantizeConfig
from diffsynth.utils.quant.serialization import save_quantized_model

quantize = QuantizeConfig(
    method="bitsandbytes_nf4",
    mode="dynamic",
    exclude_modules=[
        "time_embedder.proj_in", "time_embedder.proj_out",
        "video_patch_proj", "audio_patch_proj", "condition_proj",
        "final_layer.video_out", "final_layer.audio_out",
    ],
)
model_config = ModelConfig(
    model_id="MiniMaxAI/MiniMax-H3",
    origin_file_pattern="FL2VA/transformer/model*.safetensors",
    quantize=quantize,
)
model_hash = save_quantized_model(model_config, "models/MiniMax-H3-NF4/minimax-h3-fl2va-nf4.safetensors")
print(model_hash)
```

`exclude_modules` lists the layers sensitive to quantization (timestep embeddings, input/output projections), kept in bf16 to preserve quality.

### Step 2: Register the Hash in the Model Config

The framework identifies the model type and quantization config by file hash. The registration entry is as follows; `quant_config` must match the `QuantizeConfig` used when saving, plus `load_prequantized: True`:

```python
config_entry = {
    # Example: ModelConfig(model_id="...", origin_file_pattern="minimax-h3-fl2va-nf4.safetensors")
    "model_hash": model_hash,
    "model_name": "minimax_h3_dit",
    "model_class": "diffsynth.models.minimax_h3_dit.MiniMaxH3DiT",
    "quant_config": {"method": "bitsandbytes_nf4", "load_prequantized": True, "exclude_modules": ["time_embedder.proj_in", "time_embedder.proj_out", "video_patch_proj", "audio_patch_proj", "condition_proj", "final_layer.video_out", "final_layer.audio_out"]},
}
```

There are two ways to register:

**Option 1: register dynamically in Python code (recommended, plug-and-play).** No framework changes needed — add the entry to `MODEL_CONFIGS` before loading the model, effective for the current process:

```python
from diffsynth.configs import MODEL_CONFIGS

MODEL_CONFIGS.append(config_entry)
```

**Option 2: write it into the config file (permanent).** Add the entry above to the `MODEL_CONFIGS` list in `diffsynth/configs/model_configs.py`, so you no longer need to register it manually. If your quantized weights are publicly released, you are also welcome to submit the entry to us as a PR, so other users can load them directly.

### Step 3: Load for Inference

Once registered, loading your own quantized weights works just like loading an ordinary model, without passing `quantize`:

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
        ModelConfig(path="models/MiniMax-H3-NF4/minimax-h3-fl2va-nf4.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)
prompt = "A girl is very happy, she is speaking in english: 'I enjoy working with Diffsynth-Studio, it's a perfect framework.'"
video, audio = pipe(prompt=prompt, height=480, width=832, num_frames=124, num_inference_steps=50, seed=0)
write_video_audio(video=video, audio=audio, output_path="t2va.mp4", fps=24, audio_sample_rate=32000)
```

We have uploaded the NF4 quantized weights of MiniMax-H3 to ModelScope ([DiffSynth-Studio/MiniMax-H3-NF4](https://modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-NF4)), so you can use them directly without quantizing yourself. If you want to upload your own saved quantized weights to ModelScope, you can use the modelscope SDK:

```python
from modelscope.hub.api import HubApi

api = HubApi()
api.login("your_access_token")
api.create_model("your-username/MiniMax-H3-NF4", visibility=1)
api.upload_folder(
    repo_id="your-username/MiniMax-H3-NF4",
    folder_path="models/MiniMax-H3-NF4",
    repo_type="model",
)
```
