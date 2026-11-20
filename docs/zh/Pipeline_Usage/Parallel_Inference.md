# 并行推理

并行推理让多张卡合算**同一个样本**，缩短单张图或一段视频的生成时间。适用于单机多卡、DiT 前向占主要耗时的场景，例如高分辨率图像与长视频。原理与模型接入见[启用模型的并行推理能力](../Developer_Guide/Enabling_Parallel_Inference.md)，接口见 [`diffsynth.core.distributed`](../API_Reference/core/distributed.md)。

## 用现成脚本

已支持的模型都有并行示例，放在 `examples/<series>/model_inference_parallel/`，与 `model_inference/` 下的同名脚本一一对应，差别只有三处：传入 `parallel_config`、用 `torchrun` 启动、只在 rank0 保存结果。

```bash
torchrun --standalone --nproc_per_node=6 examples/z_image/model_inference_parallel/Z-Image-Turbo.py
```

每个脚本开头的注释就是它自己的启动命令，卡数已按该模型的头数选好，直接复制即可。

## 在自己的脚本里启用

构建 pipeline 时传入 `parallel_config`，其余代码不变：

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

* 序列切分的份数等于 `--nproc_per_node`，在配置里不指定。
* 每张卡都会算出完整结果，保存与打印放在 `if dist.get_rank() == 0:` 里，避免多进程写同一个文件。
* 不传 `parallel_config` 就是单卡路径。同一进程内的多个 pipeline 复用同一个通信组，结束时可调用 `destroy_parallel()` 释放。

## 配置 parallel_config

```python
from diffsynth.core.distributed import ParallelConfig, SequenceParallelConfig

parallel_config = ParallelConfig(
    sequence=SequenceParallelConfig(backend="ulysses", gather_between_blocks=False),
    timeout_seconds=1800,
)
```

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `ParallelConfig.sequence` | `SequenceParallelConfig()` | 序列并行配置 |
| `ParallelConfig.timeout_seconds` | `1800` | 建立通信组的超时秒数 |
| `SequenceParallelConfig.backend` | `"ulysses"` | 序列并行的交换后端 |
| `SequenceParallelConfig.gather_between_blocks` | `False` | `False`：整段 block 只在入口切分、出口还原，块间传递分片；`True`：每个 block 各自切分与还原，块间序列完整 |

`ParallelConfig()` 即上表的默认值，等价于启用 Ulysses 序列并行。切分份数不在配置里，由 `--nproc_per_node` 决定。

`gather_between_blocks` 用于块间存在对完整序列运行的旁路计算的模型，例如 MOVA 双塔之间的桥；MOVA 在启用序列并行时自动置位，通常无需手动修改。开启后每个 block 都多一次切分与还原。

## 选几张卡

卡数必须整除注意力的头数：Q 头数与 KV 头数都要满足，GQA 模型由 KV 头数决定上限。不满足时启动后会在交换处报错。

各 series 在 8 卡以内的最大可用卡数：

| series | 最大卡数 |
| --- | --- |
| boogu_image、fibo | 7 |
| z_image、ideogram4、krea2、wanvideo（`Wan2.1-T2V-1.3B`、`Echo-Memory`） | 6 |
| mova | 4 |
| 其余 series | 8 |

最大卡数的约数同样可用，例如上限 6 的模型可以跑 1、2、3、6 卡。

序列并行更适合长序列任务，例如 MiniMax-H3、Wan 等视频生成场景。卡数越多，通信占比越高；序列很短时（例如 60 秒音频只有数百 token）并行可能比单卡更慢，这类场景建议直接用单卡脚本。

## 支持范围

`from_pretrained` 接受 `parallel_config` 的 pipeline：ace_step、anima、boogu_image、diffsynth_music、ernie_image、fibo、flux、flux2、flux3、hidream_o1_image、ideogram4、joyai_image、krea2、lingbot_video、ltx2、minimax_h3、minimax_h3_image、minimax_music3、moss_image、mova、qwen_image、qwen_image_21、qwen_video_edit、sensenova_u1、wan_video、z_image。

现成的并行脚本覆盖基础生成（文生图/视频/音频、图生视频）、编辑与参考图，以及量化、pruned、comfy 等权重变体。

其余路径的模型结构或加载方式与基础脚本不同，序列并行的正确性未经验证，遇到兼容性问题请提 issue。

`stable_diffusion`、`stable_diffusion_xl`、`yue2` 没有并行入口。

## 与显存管理同用

启用 `vram_limit` 时，把四个 device 都设为 `cuda`，让权重常驻显卡：

```python
vram_config = dict(
    offload_dtype=torch.bfloat16, offload_device="cuda",
    onload_dtype=torch.bfloat16, onload_device="cuda",
    preparing_dtype=torch.bfloat16, preparing_device="cuda",
    computation_dtype=torch.bfloat16, computation_device="cuda",
)
```

以 `ModelConfig(..., **vram_config)` 传给各模型，并在 `from_pretrained` 里给出 `vram_limit`。

device 设为 `cpu` 时每步都要重新上传权重，这部分传输在单卡与并行下相同，会稀释加速比。

## 附：基于 xDiT 的 USP 方案（已停止维护）

<details>

<summary>详情</summary>

DiffSynth-Studio 曾采用基于 [xDiT](https://github.com/xdit-project/xDiT) 的统一序列并行（USP）方案，在 DiT 中将 token 序列拆分至多张显卡并行处理。该方案会引入额外通信开销，实际加速比通常低于显卡数量。

该方案曾支持 [Wan](../Model_Details/Wan.md) 与 MOVA 两个模型。

首先安装 `xDiT` 依赖。
```bash
pip install "xfuser[flash-attn]>=0.4.3"
```

然后使用 `torchrun` 启动多卡推理。
```bash
torchrun --standalone --nproc_per_node=8 examples/wanvideo/acceleration/unified_sequence_parallel.py
```

构建 pipeline 时配置 `use_usp=True` 即可启用。代码示例如下。
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

该方案已停止维护，请改用上文的 `parallel_config`；两者不要同时启用。

</details>
