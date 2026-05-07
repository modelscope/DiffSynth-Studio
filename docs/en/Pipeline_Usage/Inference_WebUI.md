# Inference WebUI

DiffSynth-Studio provides an Inference WebUI to help developers quickly validate model performance.

> The Inference WebUI is a debugging tool designed for developers, not a creation tool for end-users. For a richer feature set and more user-friendly interactive experience, we recommend using the [AIGC Zone](https://modelscope.cn/aigc/home) on ModelScope (for users in China) or the [Civision Zone](https://modelscope.ai/civision/home) (for users outside China).

## Launching the Inference WebUI

The Inference WebUI is built on [Streamlit](https://streamlit.io/). We recommend installing DiffSynth-Studio in `[all]` mode:

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .[all]
```

Launch command:

```shell
streamlit run examples/dev_tools/webui.py --server.fileWatcherType none
```

## How It Works

As a standalone tool, the Inference WebUI dynamically generates corresponding UI controls by parsing the type annotations of parameters in the Pipeline's `from_pretrained` and `__call__` methods. Therefore, the interface interaction logic is fully consistent with the code invocation logic, serving as a visual entry point for DiffSynth-Studio code.

Taking `ZImagePipeline.__call__` in `diffsynth.pipelines.z_image` as an example:

```python
@torch.no_grad()
def __call__(
    self,
    # Prompt
    prompt: str = "",
    negative_prompt: str = "",
    cfg_scale: float = 1.0,
    # Image
    input_image: Image.Image = None,
    denoising_strength: float = 1.0,
    ...
)
```

After parsing, the WebUI will automatically render the following interface:

![](https://github.com/user-attachments/assets/55795022-7a9b-4383-b048-7feabdfcdddf)

## Usage Tips

- Supports automatic loading of model information such as `model_id` and `origin_file_pattern` from sample code in `./examples`, simplifying the configuration process;
- Parameters such as `vram_limit`, `tokenizer_config`, and `lora` cannot be retrieved through code parsing and need to be filled in manually.
