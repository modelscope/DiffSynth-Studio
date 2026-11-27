# 并行推理

并行推理让多张卡合算**同一个样本**，缩短单张图或一段视频的生成时间。适用于单机多卡、DiT 前向占主要耗时的场景，例如高分辨率图像与长视频。

有两个可以叠加的并行轴：

* **序列并行**把一条样本的 token 序列切给多卡，一次前向内部并行。
* **CFG 并行**把 classifier-free guidance 的正负两次前向分给两组卡，每卡只算一支。适用于带 guidance（`cfg_scale != 1`）的模型。

原理与模型接入见[启用模型的并行推理能力](../Developer_Guide/Enabling_Parallel_Inference.md)，接口见 [`diffsynth.core.distributed`](../API_Reference/core/distributed.md)。

## 用现成脚本

已支持的模型都有并行示例，放在 `examples/<series>/model_inference_parallel/`，与 `model_inference/` 下的同名脚本一一对应，差别只有三处：传入 `parallel_config`、用 `torchrun` 启动、只在 rank0 保存结果。带 guidance 的脚本已启用 CFG 并行。

```bash
torchrun --standalone --nproc_per_node=6 examples/z_image/model_inference_parallel/Z-Image-Turbo.py
```

每个脚本开头的注释就是它自己的启动命令，卡数与两个轴的划分已按该模型选好，直接复制即可。

## 在自己的脚本里启用

构建 pipeline 时传入 `parallel_config`，其余代码不变：

```python
import torch
import torch.distributed as dist
from diffsynth.core.distributed import ParallelConfig, SequenceParallelConfig
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
    parallel_config=ParallelConfig(sequence=SequenceParallelConfig()),
)
image = pipe(prompt="a red cube on a white table", seed=0, num_inference_steps=8)
if dist.get_rank() == 0:
    image.save("output.jpg")
```

```bash
torchrun --standalone --nproc_per_node=6 your_script.py
```

* `--nproc_per_node` 是**总卡数**，等于各轴 size 之积。配置里只写启用哪些轴，份数由总卡数推出：只开序列并行时它等于总卡数；两个轴都开时 CFG 轴占 2，序列轴占总卡数的一半。
* 每张卡都会算出完整结果，保存与打印放在 `if dist.get_rank() == 0:` 里，避免多进程写同一个文件。
* 不传 `parallel_config` 就是单卡路径。同一进程内的多个 pipeline 复用同一个通信组，结束时可调用 `destroy_parallel()` 释放。

## 配置 parallel_config

```python
from diffsynth.core.distributed import ParallelConfig, SequenceParallelConfig, CFGParallelConfig

parallel_config = ParallelConfig(
    sequence=SequenceParallelConfig(),
    cfg=CFGParallelConfig(),
    timeout_seconds=1800,
)
```

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `ParallelConfig.sequence` | `None` | 序列并行轴；传入 `SequenceParallelConfig(...)` 启用 |
| `ParallelConfig.cfg` | `None` | CFG 并行轴；传入 `CFGParallelConfig(...)` 启用 |
| `ParallelConfig.timeout_seconds` | `1800` | 建立通信组的超时秒数 |
| `SequenceParallelConfig.backend` | `"ulysses"` | 序列并行的交换后端 |

**传入哪个轴的配置，就启用哪个轴**；`ParallelConfig()` 不启用任何并行。各轴份数不在配置里，由 `--nproc_per_node` 决定。

`SequenceParallelConfig` 还有一个 `gather_between_blocks`，用于块间存在对完整序列运行的旁路计算的模型。需要它的模型（例如 MOVA）在启用序列并行时自动置位，无需手动配置。

## 选几张卡

**总卡数 = CFG 轴份数 × 序列轴份数。** 启用 CFG 并行时 CFG 轴固定占 2，所以总卡数必须是偶数，序列轴拿到一半。

**头数只约束序列轴**：参与交换的 Q 头数与 KV 头数都要能被序列轴份数整除，GQA 模型由 KV 头数决定上限。CFG 轴不参与 attention 交换，与头数无关。不满足时启动后会在交换处报错，错误消息里带着真实头数。

各 series 在 8 卡以内的**序列轴最大份数**：

| series | 序列轴最大份数 |
| --- | --- |
| boogu_image、fibo | 7 |
| z_image、ideogram4、krea2、wanvideo（`Wan2.1-T2V-1.3B`、`Echo-Memory`） | 6 |
| mova | 4 |
| 其余 series | 8 |

最大份数的约数同样可用。启用 CFG 并行后总卡数上限翻倍——例如序列轴上限 4 的模型，`CFG2 × SP4` 可以用满 8 卡。

### 怎么分配这两个轴

**模型带 guidance 时，优先把卡分给 CFG 轴。** 同样卡数下 CFG 布局通常更快：CFG 轴每步只交换一次预测结果，而序列轴每个 block 都要做两次激活 all-to-all，份数越大通信占比越高。实测 8 卡上 `CFG2 × SP4` 明显快于 `SP8`，后者相对 `SP4` 几乎没有额外收益。

序列并行更适合长序列任务，例如 MiniMax-H3、Wan 等视频生成场景。序列很短时（例如 60 秒音频只有数百 token）序列并行可能比单卡更慢，这类场景可以只开 CFG 轴。

**`cfg_scale = 1` 时不要启用 CFG 并行。** 蒸馏模型通常没有负分支，此时 CFG 轴无事可做，启用会直接报错。用只开序列并行的配置。

## 支持范围

**没有并行入口**：`stable_diffusion`、`stable_diffusion_xl`、`yue2`。

**只支持序列并行，不支持 CFG 并行**：`minimax_music3`、`qwen_video_edit`。

现成的并行脚本覆盖基础生成（文生图/视频/音频、图生视频）、编辑与参考图，以及量化、pruned、comfy 等权重变体。

其余路径的模型结构或加载方式与基础脚本不同，并行的正确性未经验证，遇到兼容性问题请提 issue。

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
