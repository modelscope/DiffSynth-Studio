# DiffSynth Studio

## 介绍

DiffSynth is a new Diffusion engine. We have restructured architectures like Text Encoder, UNet, VAE, among others, maintaining compatibility with models from the open-source community while enhancing computational performance. This version is currently in its initial stage, supporting text-to-image and image-to-image functionalities, supporting SD and SDXL architectures. In the future, we plan to develop more interesting features based on this new codebase.

## 安装

If you only want to use DiffSynth Studio at the Python code level, you just need to install torch (a deep learning framework) and transformers (only used for implementing a tokenizer).

```
pip install torch transformers
```

If you wish to use the UI, you'll also need to additionally install `streamlit` (a web UI framework) and `streamlit-drawable-canvas` (used for the image-to-image canvas).

```
pip install streamlit streamlit-drawable-canvas
```

## 使用

Use DiffSynth Studio in Python

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

If you want to use SDXL architecture models, replace `SDPrompter` and `SDPipeline` with `SDXLPrompter` and `SDXLPipeline`, respectively.

Of course, you can also use the UI we provide. The UI is simple but may be changed in the future.

```
python -m streamlit run Diffsynth_Studio.py
```
