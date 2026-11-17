# `diffsynth.core.loader`: Model Download and Loading

This document introduces the model download and loading functionalities in `diffsynth.core.loader`.

## ModelConfig

`ModelConfig` in `diffsynth.core.loader` is used to annotate model download sources, local paths, VRAM management configurations, and other information.

### Downloading and Loading Models from Remote Sources

Taking the model [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny) as an example, after filling in `model_id` and `origin_file_pattern` in `ModelConfig`, the model can be automatically downloaded. By default, it downloads to the `./models` path, which can be modified through the [environment variable DIFFSYNTH_MODEL_BASE_PATH](../../Pipeline_Usage/Environment_Variables.md#diffsynth_model_base_path).

By default, even if the model has already been downloaded, the program will still query the remote for any missing files. To completely disable remote requests, set the [environment variable DIFFSYNTH_SKIP_DOWNLOAD](../../Pipeline_Usage/Environment_Variables.md#diffsynth_skip_download) to `True`.

```python
from diffsynth.core import ModelConfig

config = ModelConfig(
    model_id="DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny",
    origin_file_pattern="model.safetensors",
)
# Download models
config.download_if_necessary()
print(config.path)
```

After calling `download_if_necessary`, the model will be automatically downloaded, and the path will be returned to `config.path`.

### Loading Models from Local Paths

If loading models from local paths, you need to fill in `path`:

```python
from diffsynth.core import ModelConfig

config = ModelConfig(path="models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny/model.safetensors")
```

If the model contains multiple shard files, input them in list form:

```python
from diffsynth.core import ModelConfig

config = ModelConfig(path=[
    "models/Qwen/Qwen-Image/text_encoder/model-00001-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00002-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00003-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00004-of-00004.safetensors"
])
```

### Loading Pre-quantized Models

A pre-quantized checkpoint is usually published together with a JSON file that describes how to rebuild the model (an entry of `MODEL_CONFIGS` with the matching `quant_config`). Fill in `origin_file_pattern` for the weights and `origin_config_file` for the JSON file. After calling `download_if_necessary`, the JSON file is downloaded to `config.config_file` and its entry is registered automatically, so the checkpoint can be loaded like an ordinary model:

```python
from diffsynth.core import ModelConfig

config = ModelConfig(
    model_id="DiffSynth-Studio/MiniMax-H3-Image-EntroPack",
    origin_file_pattern="models/dit_extreme_3bpp.safetensors",
    origin_config_file="models/dit_extreme_3bpp.json",
)
config.download_if_necessary()
print(config.path, config.config_file)
```

If the JSON file is already available locally, pass it as `config_file` and call `config.register_prequantized()` before loading the model.

#### Config File Format

The JSON file contains one JSON object, which is exactly one `MODEL_CONFIGS` entry:

| Field | Required | Description |
| --- | --- | --- |
| `model_hash` | Yes | Hash of the quantized checkpoint, computed by [`hash_model_file`](#model-hash) (MD5 over the tensor names and shapes). The loader uses it to detect the model type, so it must match the checkpoint. |
| `model_name` | Yes | Name used by pipelines to fetch the model, e.g. `qwen_image_dit`. |
| `model_class` | Yes | Import path of the model class, e.g. `diffsynth.models.qwen_image_dit.QwenImageDiT`. |
| `quant_config` | Yes | Keyword arguments of [`QuantizeConfig`](../../Pipeline_Usage/Quantization.md). `load_prequantized` must be `true`, and the other fields must match the quantization used when the checkpoint was saved. |
| `state_dict_converter` | No | Import path of a state dict converter, see [Integrating Your Model](../../Developer_Guide/Integrating_Your_Model.md). |
| `extra_kwargs` | No | Extra keyword arguments passed to the model constructor. |

When the checkpoint is quantized with a single method, `quant_config` holds that method and its arguments:

```json
{
    "model_hash": "8c6f5ccfe9ffe861675ca0da95bb9295",
    "model_name": "minimax_h3_text_encoder",
    "model_class": "diffsynth.models.minimax_h3_text_encoder.MiniMaxH3TextEncoder",
    "quant_config": {
        "method": "entropack_lossy_quant",
        "load_prequantized": true,
        "backend_config_kwargs": {"target_bpp": 4.0, "row_rdo_iterations": 2, "row_rdo_candidates": 20}
    }
}
```

When different modules use different methods, use `configs` instead of `method`; each item is a `QuantizeConfig` selecting its modules with `target_modules` / `exclude_modules`, all items share the same `mode`, and `load_prequantized` is set on the outer object:

```json
{
    "model_hash": "adab44d835193ce1b2bd0f85f0586f0e",
    "model_name": "minimax_h3_image_dit",
    "model_class": "diffsynth.models.minimax_h3_image_dit.MiniMaxH3ImageDiTPruned",
    "quant_config": {
        "load_prequantized": true,
        "configs": [
            {"method": "entropack_lossy_quant", "backend_config_kwargs": {"target_bpp": 2.91}, "exclude_modules": ["video_patch_proj", "condition_proj"]},
            {"method": "entropack_lossy_quant", "backend_config_kwargs": {"target_bpp": 8.5}, "target_modules": ["video_patch_proj", "condition_proj"]}
        ]
    }
}
```

The file name is arbitrary; naming it after the checkpoint (`model.safetensors` → `model.json`) is a common convention. If an entry with the same `model_hash` is already in `MODEL_CONFIGS`, the entry in the file is ignored.

### VRAM Management Configuration

`ModelConfig` also contains VRAM management configuration information. See [VRAM Management](../../Pipeline_Usage/VRAM_management.md#more-usage-methods) for details.

## Model File Loading

`diffsynth.core.loader` provides a unified `load_state_dict` for loading state dicts from model files.

Loading a single model file:

```python
from diffsynth.core import load_state_dict

state_dict = load_state_dict("models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny/model.safetensors")
```

Loading multiple model files (merged into one state dict):

```python
from diffsynth.core import load_state_dict

state_dict = load_state_dict([
    "models/Qwen/Qwen-Image/text_encoder/model-00001-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00002-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00003-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00004-of-00004.safetensors"
])
```

## Model Hash

Model hash is used to determine the model type. The hash value can be obtained through `hash_model_file`:

```python
from diffsynth.core import hash_model_file

print(hash_model_file("models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny/model.safetensors"))
```

The hash value of multiple model files can also be calculated, which is equivalent to calculating the model hash value after merging the state dict:

```python
from diffsynth.core import hash_model_file

print(hash_model_file([
    "models/Qwen/Qwen-Image/text_encoder/model-00001-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00002-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00003-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00004-of-00004.safetensors"
]))
```

The model hash value is only related to the keys and tensor shapes in the state dict of the model file, and is unrelated to the numerical values of the model parameters, file saving time, and other information. When calculating the model hash value of `.safetensors` format files, `hash_model_file` is almost instantly completed without reading the model parameters. However, when calculating the model hash value of `.bin`, `.pth`, `.ckpt`, and other binary files, all model parameters need to be read, so **we do not recommend developers to continue using these formats of files.**

By [writing model Config](../../Developer_Guide/Integrating_Your_Model.md#step-3-writing-model-config) and filling in model hash value and other information into `diffsynth/configs/model_configs.py`, developers can let `DiffSynth-Studio` automatically identify the model type and load it.

## Model Loading

`load_model` is the external entry for loading models in `diffsynth.core.loader`. It will call [skip_model_initialization](../../API_Reference/core/vram.md#skipping-model-parameter-initialization) to skip model parameter initialization. If [Disk Offload](../../Pipeline_Usage/VRAM_management.md#disk-offload) is enabled, it calls [DiskMap](../../API_Reference/core/vram.md#state-dict-disk-mapping) for lazy loading. If Disk Offload is not enabled, it calls [load_state_dict](#model-file-loading) to load model parameters. If necessary, it will also call [state dict converter](../../Developer_Guide/Integrating_Your_Model.md#step-2-model-file-format-conversion) for model format conversion. Finally, it calls `model.eval()` to switch to inference mode.

Here is a usage example with Disk Offload enabled:

```python
from diffsynth.core import load_model, enable_vram_management, AutoWrappedLinear, AutoWrappedModule
from diffsynth.models.qwen_image_dit import QwenImageDiT, RMSNorm
import torch

prefix = "models/Qwen/Qwen-Image/transformer/diffusion_pytorch_model"
model_path = [prefix + f"-0000{i}-of-00009.safetensors" for i in range(1, 10)]

model = load_model(
    QwenImageDiT,
    model_path,
    module_map={
        torch.nn.Linear: AutoWrappedLinear,
        RMSNorm: AutoWrappedModule,
    },
    vram_config={
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": "disk",
        "onload_device": "disk",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    },
    vram_limit=0,
)
```