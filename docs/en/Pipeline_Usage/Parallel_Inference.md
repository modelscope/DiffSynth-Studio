# Parallel Inference

Parallel inference lets several GPUs compute **the same sample** together, shortening the generation time of a single image or video. It suits single-node multi-GPU setups where the DiT forward dominates the runtime, such as high-resolution images and long videos. For principles and model integration see [Enabling Parallel Inference for a Model](../Developer_Guide/Enabling_Parallel_Inference.md); for interfaces see [`diffsynth.core.distributed`](../API_Reference/core/distributed.md).

## Using the Ready-Made Scripts

Supported models ship parallel examples under `examples/<series>/model_inference_parallel/`, one for each same-named script in `model_inference/`. They differ in only three ways: `parallel_config` is passed in, they are launched with `torchrun`, and results are saved on rank 0 only.

```bash
torchrun --standalone --nproc_per_node=6 examples/z_image/model_inference_parallel/Z-Image-Turbo.py
```

The comment at the top of each script is its own launch command, with the GPU count already chosen for that model's head counts — copy it as is.

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

* The number of sequence shards equals `--nproc_per_node`; it is not set in the configuration.
* Every GPU computes the full result, so put saving and printing inside `if dist.get_rank() == 0:` to avoid several processes writing the same file.
* Without `parallel_config` the single-GPU path is used. Pipelines within one process share the same communication group, which can be released with `destroy_parallel()` when finished.

## Configuring parallel_config

```python
from diffsynth.core.distributed import ParallelConfig, SequenceParallelConfig

parallel_config = ParallelConfig(
    sequence=SequenceParallelConfig(backend="ulysses", gather_between_blocks=False),
    timeout_seconds=1800,
)
```

| Field | Default | Description |
| --- | --- | --- |
| `ParallelConfig.sequence` | `SequenceParallelConfig()` | Sequence parallelism configuration |
| `ParallelConfig.timeout_seconds` | `1800` | Timeout in seconds for establishing the communication group |
| `SequenceParallelConfig.backend` | `"ulysses"` | Exchange backend for sequence parallelism |
| `SequenceParallelConfig.gather_between_blocks` | `False` | `False`: the whole run of blocks is split at its entry and restored at its exit, with shards passed between blocks. `True`: each block splits and restores on its own, so sequences are complete between blocks |

`ParallelConfig()` is exactly the defaults above, equivalent to enabling Ulysses sequence parallelism. The number of shards is not part of the configuration; it follows `--nproc_per_node`.

`gather_between_blocks` is for models with a side computation between blocks that operates on the full sequence, such as the bridge between MOVA's two towers. MOVA sets it automatically when sequence parallelism is enabled, so it usually needs no manual change. Turning it on adds one split and restore per block.

## Choosing the Number of GPUs

The GPU count must divide the attention head counts: both Q heads and KV heads, with GQA models limited by their KV head count. Otherwise the run raises at the exchange.

Maximum usable GPU counts up to 8, per series:

| series | Maximum GPUs |
| --- | --- |
| boogu_image, fibo | 7 |
| z_image, ideogram4, krea2, wanvideo (`Wan2.1-T2V-1.3B`, `Echo-Memory`) | 6 |
| mova | 4 |
| all other series | 8 |

Divisors of the maximum work as well: a model capped at 6 can run on 1, 2, 3 or 6 GPUs.

Sequence parallelism suits long-sequence tasks best, such as video generation with MiniMax-H3 or Wan. The more GPUs, the larger the share of communication; for very short sequences (60 seconds of audio is only a few hundred tokens) parallel inference can be slower than a single GPU, and the single-GPU scripts are the better choice there.

## Coverage

Pipelines whose `from_pretrained` accepts `parallel_config`: ace_step, anima, boogu_image, diffsynth_music, ernie_image, fibo, flux, flux2, flux3, hidream_o1_image, ideogram4, joyai_image, krea2, lingbot_video, ltx2, minimax_h3, minimax_h3_image, minimax_music3, moss_image, mova, qwen_image, qwen_image_21, qwen_video_edit, sensenova_u1, wan_video, z_image.

The ready-made parallel scripts cover basic generation (text-to-image/video/audio, image-to-video), editing and reference images, and weight variants such as quantized, pruned and comfy checkpoints.

Other paths differ from the basic scripts in model structure or loading, and their correctness under sequence parallelism has not been verified. Please open an issue if you run into a compatibility problem.

`stable_diffusion`, `stable_diffusion_xl` and `yue2` have no parallel entry point.

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