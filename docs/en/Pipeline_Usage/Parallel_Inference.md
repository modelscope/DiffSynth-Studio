# Parallel Inference

Parallel inference lets several GPUs compute **the same sample** together, shortening the generation time of a single image or video. It suits single-node multi-GPU setups where the DiT forward dominates the runtime, such as high-resolution images and long videos.

Two parallel axes can be stacked:

* **Sequence parallelism** splits the token sequence of one sample across GPUs, parallelizing inside a single forward.
* **CFG parallelism** puts the positive and negative classifier-free-guidance forwards on two groups of GPUs, each computing one branch. It applies to models that use guidance (`cfg_scale != 1`).

For principles and model integration see [Enabling Parallel Inference for a Model](../Developer_Guide/Enabling_Parallel_Inference.md); for interfaces see [`diffsynth.core.distributed`](../API_Reference/core/distributed.md).

## Using the Ready-Made Scripts

Supported models ship parallel examples under `examples/<series>/model_inference_parallel/`, one for each same-named script in `model_inference/`. They differ in only three ways: `parallel_config` is passed in, they are launched with `torchrun`, and results are saved on rank 0 only. Scripts for models that use guidance already enable CFG parallelism.

```bash
torchrun --standalone --nproc_per_node=6 examples/z_image/model_inference_parallel/Z-Image-Turbo.py
```

The comment at the top of each script is its own launch command, with the GPU count and the split across the two axes already chosen for that model — copy it as is.

## Enabling It in Your Own Script

Pass `parallel_config` when building the pipeline; the rest of the code stays the same:

```python
import torch
import torch.distributed as dist
from diffsynth.core.distributed import ParallelConfig
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig

pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
    parallel_config=ParallelConfig(),
)
image = pipe(prompt="a red cube on a white table", seed=0, num_inference_steps=8)
if dist.get_rank() == 0:
    image.save("output.jpg")
```

```bash
torchrun --standalone --nproc_per_node=6 your_script.py
```

* `--nproc_per_node` is the **total GPU count**, equal to the product of the axis sizes. The configuration only says which axes are on; the sizes follow from the total: with sequence parallelism alone it equals the total, and with both axes on the CFG axis takes 2 and the sequence axis takes half the total.
* Every GPU computes the full result, so put saving and printing inside `if dist.get_rank() == 0:` to avoid several processes writing the same file.
* Without `parallel_config` the single-GPU path is used. Pipelines within one process share the same communication group, which can be released with `destroy_parallel()` when finished.

## Configuring parallel_config

```python
from diffsynth.core.distributed import ParallelConfig, SequenceParallelConfig, CFGParallelConfig

parallel_config = ParallelConfig(
    sequence=SequenceParallelConfig(),
    cfg=CFGParallelConfig(),
    timeout_seconds=1800,
)
```

| Field | Default | Description |
| --- | --- | --- |
| `ParallelConfig.sequence` | `None` | Sequence-parallel axis; enabled by passing `SequenceParallelConfig(...)` |
| `ParallelConfig.cfg` | `None` | CFG-parallel axis; enabled by passing `CFGParallelConfig(...)` |
| `ParallelConfig.timeout_seconds` | `1800` | Timeout in seconds for establishing the communication group |
| `SequenceParallelConfig.backend` | `"ulysses"` | Exchange backend for sequence parallelism |

**Passing an axis's configuration enables that axis**; `ParallelConfig()` enables no parallelism. Axis sizes are not part of the configuration; they follow from `--nproc_per_node`.

`SequenceParallelConfig` also has `gather_between_blocks`, for models with a side computation between blocks that operates on the full sequence. Models that need it (such as MOVA) set it automatically when sequence parallelism is enabled; no manual configuration is required.

## Choosing the Number of GPUs

**Total GPUs = CFG axis size × sequence axis size.** With CFG parallelism on, the CFG axis is fixed at 2, so the total must be even and the sequence axis gets half.

**Head counts constrain only the sequence axis**: the Q and KV head counts taking part in the exchange must both be divisible by the sequence axis size, with GQA models limited by their KV head count. The CFG axis takes no part in the attention exchange and is independent of head counts. When the constraint fails, the run raises at the exchange, and the error message carries the actual head count.

Maximum **sequence axis size** up to 8 GPUs, per series:

| series | Maximum sequence axis size |
| --- | --- |
| boogu_image, fibo | 7 |
| z_image, ideogram4, krea2, wanvideo (`Wan2.1-T2V-1.3B`, `Echo-Memory`) | 6 |
| mova | 4 |
| all other series | 8 |

Divisors of the maximum work as well. Enabling CFG parallelism doubles the total GPU ceiling — a model whose sequence axis caps at 4 can use all 8 GPUs as `CFG2 × SP4`.

### How to Split the Two Axes

**When the model uses guidance, give GPUs to the CFG axis first.** At equal GPU counts the CFG layout is usually faster: the CFG axis exchanges one prediction per step, while the sequence axis performs two activation all-to-alls at every block, so its communication share grows with its size. Measurements on 8 GPUs show `CFG2 × SP4` clearly ahead of `SP8`, and the latter gains almost nothing over `SP4`.

Sequence parallelism suits long-sequence tasks best, such as video generation with MiniMax-H3 or Wan. For very short sequences (60 seconds of audio is only a few hundred tokens) sequence parallelism can be slower than a single GPU; there, enabling only the CFG axis is an option.

**Do not enable CFG parallelism when `cfg_scale = 1`.** Distilled models usually have no negative branch, so the CFG axis has nothing to do and enabling it raises. Use a configuration with sequence parallelism alone.

## Coverage

**No parallel entry point**: `stable_diffusion`, `stable_diffusion_xl`, `yue2`.

**Sequence parallelism only, no CFG parallelism**: `minimax_music3`, `qwen_video_edit`.

The ready-made parallel scripts cover basic generation (text-to-image/video/audio, image-to-video), editing and reference images, and weight variants such as quantized, pruned and comfy checkpoints.

Other paths differ from the basic scripts in model structure or loading, and their correctness under parallelism has not been verified. Please open an issue if you run into a compatibility problem.

## Using It with VRAM Management

When `vram_limit` is enabled, set all four devices to `cuda` so weights stay resident on the GPU:

```python
vram_config = dict(
    offload_dtype=torch.bfloat16, offload_device="cuda",
    onload_dtype=torch.bfloat16, onload_device="cuda",
    preparing_dtype=torch.bfloat16, preparing_device="cuda",
    computation_dtype=torch.bfloat16, computation_device="cuda",
)
```

Pass it to each model as `ModelConfig(..., **vram_config)` and provide `vram_limit` in `from_pretrained`.

With devices set to `cpu`, weights are re-uploaded at every step. That transfer is the same on one GPU and in parallel, so it dilutes the speedup.

## Appendix: the xDiT-based USP Approach (No Longer Maintained)

<details>

<summary>Details</summary>

DiffSynth-Studio previously used a multi-GPU inference approach based on Unified Sequence Parallel (USP), splitting the token sequence in the DiT across GPUs for parallel processing, with [xDiT](https://github.com/xdit-project/xDiT) underneath. That approach introduces additional communication overhead, so the actual speedup is usually lower than the number of GPUs.

It supported the [Wan](../Model_Details/Wan.md) and MOVA models.

First, install the `xDiT` dependency.

```bash
pip install "xfuser[flash-attn]>=0.4.3"
```

Then use `torchrun` to launch multi-GPU inference.

```bash
torchrun --standalone --nproc_per_node=8 examples/wanvideo/acceleration/unified_sequence_parallel.py
```

When building the pipeline, configure `use_usp=True` to enable it. A code example is shown below.

```python
import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
import torch.distributed as dist

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    use_usp=True,
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
)

# Text-to-video
video = pipe(
    prompt="一名宇航员身穿太空服，面朝镜头骑着一匹机械马在火星表面驰骋。红色的荒凉地表延伸至远方，点缀着巨大的陨石坑和奇特的岩石结构。机械马的步伐稳健，扬起微弱的尘埃，展现出未来科技与原始探索的完美结合。宇航员手持操控装置，目光坚定，仿佛正在开辟人类的新疆域。背景是深邃的宇宙和蔚蓝的地球，画面既科幻又充满希望，让人不禁畅想未来的星际生活。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
)
if dist.get_rank() == 0:
    save_video(video, "video1.mp4", fps=15, quality=5)
```

This approach is no longer maintained; use `parallel_config` described above instead. Do not enable both at the same time.

</details>