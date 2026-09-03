# Echo-Memory (optional Wan 1.3B overlay)

[Echo-Memory](https://github.com/Echo-Team-Joy-Future-Academy-JD/Echo-Memory) ([arXiv:2606.09803](https://arxiv.org/abs/2606.09803)) is a controlled memory study on **Wan 2.1 1.3B**. Released rows live at [Echo-Team/Echo-Memory](https://huggingface.co/Echo-Team/Echo-Memory).

This folder is an **optional loader**, not a new backbone:

- `load_echo_memory_dit(pipe)` overlays `context_k1` (or another row) onto the official Wan DiT.
- Matching keys use the original DiffSynth / Wan names, so no conversion is required here.
- Camera-action MLP, block-wise SSM, spatial memory, and the multi-chunk revisit protocol stay in the Echo-Memory repository.

```python
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from echo_memory import load_echo_memory_dit

pipe = WanVideoPipeline.from_pretrained(...)
load_echo_memory_dit(pipe)  # Echo-Team/Echo-Memory context_k1
```

See `examples/wanvideo/model_inference/Echo-Memory.py`.
