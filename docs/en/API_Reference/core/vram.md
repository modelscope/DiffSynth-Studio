# `diffsynth.core.vram`: VRAM Management

This document introduces the underlying VRAM management functionalities in `diffsynth.core.vram`. If you wish to use these functionalities in other codebases, you can refer to this document.

## Skipping Model Parameter Initialization

When loading models in `PyTorch`, model parameters default to occupying VRAM or memory and initializing parameters, but these parameters will be overwritten when loading pretrained weights, leading to redundant computations. `PyTorch` does not provide an interface to skip these redundant computations. We provide `skip_model_initialization` in `diffsynth.core.vram` to skip model parameter initialization.

Default model loading approach:

```python
from diffsynth.core import load_state_dict
from diffsynth.models.qwen_image_controlnet import QwenImageBlockWiseControlNet

model = QwenImageBlockWiseControlNet() # Slow
path = "models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny/model.safetensors"
state_dict = load_state_dict(path, device="cpu")
model.load_state_dict(state_dict, assign=True)
```

Model loading approach that skips parameter initialization:

```python
from diffsynth.core import load_state_dict, skip_model_initialization
from diffsynth.models.qwen_image_controlnet import QwenImageBlockWiseControlNet

with skip_model_initialization():
    model = QwenImageBlockWiseControlNet() # Fast
path = "models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny/model.safetensors"
state_dict = load_state_dict(path, device="cpu")
model.load_state_dict(state_dict, assign=True)
```

In `DiffSynth-Studio`, all pretrained models follow this loading logic. After developers [integrate models](/docs/en/Developer_Guide/Integrating_Your_Model.md), they can directly load models quickly using this approach.

## State Dict Disk Mapping

For pretrained weight files of a model, if we only need to read a set of parameters rather than all parameters, State Dict Disk Mapping can accelerate this process. We provide `DiskMap` in `diffsynth.core.vram` for on-demand loading of model parameters.

Default weight loading approach:

```python
from diffsynth.core import load_state_dict

path = "models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny/model.safetensors"
state_dict = load_state_dict(path, device="cpu") # Slow
print(state_dict["img_in.weight"])
```

Using `DiskMap` to load only specific parameters:

```python
from diffsynth.core import DiskMap

path = "models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny/model.safetensors"
state_dict = DiskMap(path, device="cpu") # Fast
print(state_dict["img_in.weight"])
```

`DiskMap` is the basic component of Disk Offload in `DiffSynth-Studio`. After developers [configure fine-grained VRAM management schemes](/docs/en/Developer_Guide/Enabling_VRAM_management.md), they can directly enable Disk Offload.

`DiskMap` is a functionality implemented using the characteristics of `.safetensors` files. Therefore, when using `.bin`, `.pth`, `.ckpt`, and other binary files, model parameters are fully loaded, which causes Disk Offload to not support these formats of files. **We do not recommend developers to continue using these formats of files.**

## Replacable Modules for VRAM Management

When `DiffSynth-Studio`'s VRAM management is enabled, the modules inside the model will be replaced with replacable modules in `diffsynth.core.vram.layers`. For usage, see [Fine-grained VRAM Management Scheme](/docs/en/Developer_Guide/Enabling_VRAM_management.md#writing-fine-grained-vram-management-schemes).