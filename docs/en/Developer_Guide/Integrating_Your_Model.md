# Integrating Model Architecture

This document introduces how to integrate models into the `DiffSynth-Studio` framework for use by modules such as `Pipeline`.

## Step 1: Integrate Model Architecture Code

All model architecture implementations in `DiffSynth-Studio` are unified in `diffsynth/models`. Each `.py` code file implements a model architecture, and all models are loaded through `ModelPool` in `diffsynth/models/model_loader.py`. When integrating new model architectures, please create a new `.py` file under this path.

```shell
diffsynth/models/
├── general_modules.py
├── model_loader.py
├── qwen_image_controlnet.py
├── qwen_image_dit.py
├── qwen_image_text_encoder.py
├── qwen_image_vae.py
└── ...
```

In most cases, we recommend integrating models in native `PyTorch` code form, with the model architecture class directly inheriting from `torch.nn.Module`, for example:

```python
import torch

class NewDiffSynthModel(torch.nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)
        self.activation = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x
```

If the model architecture implementation contains additional dependencies, we strongly recommend removing them, otherwise this will cause heavy package dependency issues. In our existing models, Qwen-Image's Blockwise ControlNet is integrated in this way. The code is lightweight, please refer to `diffsynth/models/qwen_image_controlnet.py`.

If the model has been integrated by Huggingface Library ([`transformers`](https://huggingface.co/docs/transformers/main/index), [`diffusers`](https://huggingface.co/docs/diffusers/main/index), etc.), we can integrate the model in a simpler way:

<details>
<summary>Integrating Huggingface Library Style Model Architecture Code</summary>

The loading method for these models in Huggingface Library is:

```python
from transformers import XXX_Model

model = XXX_Model.from_pretrained("path_to_your_model")
```

`DiffSynth-Studio` does not support loading models through `from_pretrained` because this conflicts with VRAM management and other functions. Please rewrite the model architecture in the following format:

```python
import torch

class DiffSynth_XXX_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import XXX_Config, XXX_Model
        config = XXX_Config(**{
            "architectures": ["XXX_Model"],
            "other_configs": "Please copy and paste the other configs here.",
        })
        self.model = XXX_Model(config)
        
    def forward(self, x):
        outputs = self.model(x)
        return outputs
```

Where `XXX_Config` is the Config class corresponding to the model. For example, the Config class for `Qwen2_5_VLModel` is `Qwen2_5_VLConfig`, which can be found by consulting its source code. The content inside Config can usually be found in the `config.json` file in the model library. `DiffSynth-Studio` will not read the `config.json` file, so the content needs to be copied and pasted into the code.

In rare cases, version updates of `transformers` and `diffusers` may cause some models to be unable to import. Therefore, if possible, we still recommend using the model integration method in Step 1.1.

In our existing models, Qwen-Image's Text Encoder is integrated in this way. The code is lightweight, please refer to `diffsynth/models/qwen_image_text_encoder.py`.

</details>

## Step 2: Model File Format Conversion

Due to the variety of model file formats provided by developers in the open-source community, we sometimes need to convert model file formats to form correctly formatted [state dict](https://docs.pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html). This is common in the following situations:

* Model files built by different code libraries, for example [Wan-AI/Wan2.1-T2V-1.3B](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B) and [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers).
* Models modified during integration, for example, the Text Encoder of [Qwen/Qwen-Image](https://www.modelscope.cn/models/Qwen/Qwen-Image) adds a `model.` prefix in `diffsynth/models/qwen_image_text_encoder.py`.
* Model files containing multiple models, for example, the VACE Adapter and base DiT model of [Wan-AI/Wan2.1-VACE-14B](https://www.modelscope.cn/models/Wan-AI/Wan2.1-VACE-14B) are mixed and stored in the same set of model files.

In our development philosophy, we hope to respect the wishes of model authors as much as possible. If we repackage the model files, for example [Comfy-Org/Qwen-Image_ComfyUI](https://www.modelscope.cn/models/Comfy-Org/Qwen-Image_ComfyUI), although we can call the model more conveniently, traffic (model page views and downloads, etc.) will be directed elsewhere, and the original author of the model will also lose the power to delete the model. Therefore, we have added the `diffsynth/utils/state_dict_converters` module to the framework for file format conversion during model loading.

This part of logic is very simple. Taking Qwen-Image's Text Encoder as an example, only 10 lines of code are needed:

```python
def QwenImageTextEncoderStateDictConverter(state_dict):
    state_dict_ = {}
    for k in state_dict:
        v = state_dict[k]
        if k.startswith("visual."):
            k = "model." + k
        elif k.startswith("model."):
            k = k.replace("model.", "model.language_model.")
        state_dict_[k] = v
    return state_dict_
```

## Step 3: Writing Model Config

Model Config is located in `diffsynth/configs/model_configs.py`, used to identify model types and load them. The following fields need to be filled in:

* `model_hash`: Model file hash value, which can be obtained through the `hash_model_file` function. This hash value is only related to the keys and tensor shapes in the model file's state dict, and is unrelated to other information in the file.
* `model_name`: Model name, used for `Pipeline` to identify the required model. If different structured models play the same role in `Pipeline`, the same `model_name` can be used. When integrating new models, just ensure that `model_name` is different from other existing functional models. The corresponding model is fetched through `model_name` in the `Pipeline`'s `from_pretrained`.
* `model_class`: Model architecture import path, pointing to the model architecture class implemented in Step 1, for example `diffsynth.models.qwen_image_text_encoder.QwenImageTextEncoder`.
* `state_dict_converter`: Optional parameter. If model file format conversion is needed, the import path of the model conversion logic needs to be filled in, for example `diffsynth.utils.state_dict_converters.qwen_image_text_encoder.QwenImageTextEncoderStateDictConverter`.
* `extra_kwargs`: Optional parameter. If additional parameters need to be passed when initializing the model, these parameters need to be filled in. For example, models [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny) and [DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint](https://www.modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint) both adopt the `QwenImageBlockWiseControlNet` structure in `diffsynth/models/qwen_image_controlnet.py`, but the latter also needs additional configuration `additional_in_dim=4`. Therefore, this configuration information needs to be filled in the `extra_kwargs` field.

We provide a piece of code to quickly understand how models are loaded through this configuration information:

```python
from diffsynth.core import hash_model_file, load_state_dict, skip_model_initialization
from diffsynth.models.qwen_image_text_encoder import QwenImageTextEncoder
from diffsynth.utils.state_dict_converters.qwen_image_text_encoder import QwenImageTextEncoderStateDictConverter
import torch

model_hash = "8004730443f55db63092006dd9f7110e"
model_name = "qwen_image_text_encoder"
model_class = QwenImageTextEncoder
state_dict_converter = QwenImageTextEncoderStateDictConverter
extra_kwargs = {}

model_path = [
    "models/Qwen/Qwen-Image/text_encoder/model-00001-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00002-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00003-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00004-of-00004.safetensors",
]
if hash_model_file(model_path) == model_hash:
    with skip_model_initialization():
        model = model_class(**extra_kwargs)
    state_dict = load_state_dict(model_path, torch_dtype=torch.bfloat16, device="cuda")
    state_dict = state_dict_converter(state_dict)
    model.load_state_dict(state_dict, assign=True)
    print("Done!")
```

> Q: The logic of the above code looks very simple, why is this part of code in `DiffSynth-Studio` extremely complex?
> 
> A: Because we provide aggressive VRAM management functions that are coupled with the model loading logic, this leads to the complexity of the framework structure. We have tried our best to simplify the interface exposed to developers.

The `model_hash` in `diffsynth/configs/model_configs.py` is not uniquely existing. Multiple models may exist in the same model file. For this situation, please use multiple model Configs to load each model separately, and write the corresponding `state_dict_converter` to separate the parameters required by each model.

## Step 4: Verifying Whether the Model Can Be Recognized and Loaded

After model integration, the following code can be used to verify whether the model can be correctly recognized and loaded. The following code will attempt to load the model into memory:

```python
from diffsynth.models.model_loader import ModelPool

model_pool = ModelPool()
model_pool.auto_load_model(
    [
        "models/Qwen/Qwen-Image/text_encoder/model-00001-of-00004.safetensors",
        "models/Qwen/Qwen-Image/text_encoder/model-00002-of-00004.safetensors",
        "models/Qwen/Qwen-Image/text_encoder/model-00003-of-00004.safetensors",
        "models/Qwen/Qwen-Image/text_encoder/model-00004-of-00004.safetensors",
    ],
)
```

If the model can be recognized and loaded, you will see the following output:

```
Loading models from: [
    "models/Qwen/Qwen-Image/text_encoder/model-00001-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00002-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00003-of-00004.safetensors",
    "models/Qwen/Qwen-Image/text_encoder/model-00004-of-00004.safetensors"
]
Loaded model: {
    "model_name": "qwen_image_text_encoder",
    "model_class": "diffsynth.models.qwen_image_text_encoder.QwenImageTextEncoder",
    "extra_kwargs": null
}
```

## Step 5: Writing Model VRAM Management Scheme

`DiffSynth-Studio` supports complex VRAM management. See [Enabling VRAM Management](/docs/en/Developer_Guide/Enabling_VRAM_management.md) for details.