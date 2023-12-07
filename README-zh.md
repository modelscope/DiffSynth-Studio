# DiffSynth Studio

## 介绍

DiffSynth 是一个全新的 Diffusion 引擎，我们重构了 Text Encoder、UNet、VAE 等架构，保持与开源社区模型兼容性的同时，提升了计算性能。目前这个版本仅仅是一个初始版本，实现了文生图和图生图功能，支持 SD 和 SDXL 架构。未来我们计划基于这个全新的代码库开发更多有趣的功能。

## 安装

如果你只想在 Python 代码层面调用 DiffSynth Studio，你只需要安装 `torch`（深度学习框架）和 `transformers`（仅用于实现分词器）。

```
pip install torch transformers
```

如果你想使用 UI，还需要额外安装 `streamlit`（一个 webui 框架）和 `streamlit-drawable-canvas`（用于图生图画板）。

```
pip install streamlit streamlit-drawable-canvas
```

## 使用

通过 Python 代码调用

```python
from diffsynth.models import ModelManager
from diffsynth.prompts import SDPrompter, SDXLPrompter
from diffsynth.pipelines import SDPipeline, SDXLPipeline


model_manager = ModelManager()
model_manager.load_from_safetensors("xxxxxxxx.safetensors")
prompter = SDPrompter()
pipe = SDPipeline()

prompt = "a girl"
negative_prompt = ""

image = pipe(
    model_manager, prompter,
    prompt, negative_prompt=negative_prompt,
    num_inference_steps=20, height=512, width=512,
)
image.save("image.png")
```

如果需要用 SDXL 架构模型，请把 `SDPrompter`、`SDPipeline` 换成 `SDXLPrompter`, `SDXLPipeline`。

当然，你也可以使用我们提供的 UI，但请注意，我们的 UI 程序很简单，且未来可能会大幅改变。

```
python -m streamlit run Diffsynth_Studio.py
```
