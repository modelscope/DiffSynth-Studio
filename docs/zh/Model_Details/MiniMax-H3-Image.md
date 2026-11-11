# MiniMax-H3-Image

MiniMax-H3-Image 是 MiniMax H3 家族中的图像生成模型，以开放权重形式发布。生成与编辑共用同一个模型权重：模型既能根据文本生成图像，也能根据一张或多张参考图完成图像编辑，二者由同一个 33.1B 的扩散 Transformer 承担。它可以在任意画幅比例下直接生成约 1600 万像素的图像（从 4096×4096 到 6272×2688），无需超分；模型被蒸馏到 6 步采样，且不需要无分类器引导（CFG）。该扩散 Transformer 采用 H3 架构并去掉了音频分支，文本编码器与 MiniMax-H3 完全一致。

## 安装

在使用本模型进行推理和训练之前，请先安装 DiffSynth-Studio。

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

更多安装方式请参考[安装依赖](../Pipeline_Usage/Setup.md)。

## 快速开始

运行以下代码即可先用文本生成一张图像，再对刚生成的图像做一次编辑。该脚本把全部组件都放在显卡上，总共需要约 135 GB 显存（文本编码器 66.7 GB、Transformer 66.3 GB、图像 VAE 2.3 GB）；显存不够时请使用[低显存示例](/examples/minimax_h3_image/model_inference_low_vram/MiniMax-H3-Image.py)，它把文本编码器与 VAE 按磁盘流式加载、把 Transformer 以 int8 存放在主机内存。面向 24GB 消费级显卡与数据中心部署，参见[不同硬件上的推理方案](#不同硬件上的推理方案)。

```python
import torch
from diffsynth.pipelines.minimax_h3_image import MiniMaxH3ImagePipeline, ModelConfig
from diffsynth.core.quant import QuantizeConfig
from PIL import Image


pipe = MiniMaxH3ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors"),
        ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="transformer/model*.safetensors"),
        ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="pdd_fused_heads.safetensors"),
        ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)

prompt = (
    "An overhead photograph of a rustic farm-to-table spread on a weathered wooden table: a round loaf of sourdough "
    "on burlap, wedges of cheese on a dark board, a copper bowl of vegetables, vine tomatoes, olives and walnuts, "
    "warm afternoon light raking across the grain of the wood."
)
image = pipe(prompt=prompt, height=4096, width=4096, seed=42)
image.save("MiniMax-H3-Image.jpg")

prompt = "Change the color of the tomato in <Picture 1> to green."
image = pipe(
    prompt=prompt, height=4096, width=4096, seed=43,
    edit_image=[Image.open("MiniMax-H3-Image.jpg")],
)
image.save("MiniMax-H3-Image-Edit.jpg")
```

## 模型总览

|模型 ID|推理|低显存推理|全量训练|全量训练后验证|LoRA 训练|LoRA 训练后验证|
|-|-|-|-|-|-|-|
|[MiniMax/MiniMax-H3-Image](https://www.modelscope.cn/models/MiniMax/MiniMax-H3-Image)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_inference/MiniMax-H3-Image.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_inference_low_vram/MiniMax-H3-Image.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/full/MiniMax-H3-Image.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/validate_full/MiniMax-H3-Image.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/lora/MiniMax-H3-Image.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/validate_lora/MiniMax-H3-Image.py)|
|[DiffSynth-Studio/MiniMax-H3-Image-EntroPack](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-Image-EntroPack) (AdaLN-pruned)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_inference/MiniMax-H3-Image-Pruned.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_inference_low_vram/MiniMax-H3-Image-Pruned.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/full/MiniMax-H3-Image-Pruned.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/validate_full/MiniMax-H3-Image-Pruned.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/lora/MiniMax-H3-Image-Pruned.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/validate_lora/MiniMax-H3-Image-Pruned.py)|
|[DiffSynth-Studio/MiniMax-H3-Image-EntroPack](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-Image-EntroPack) (EntroPack)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_inference/MiniMax-H3-Image-Pruned-EntroPack.py)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_inference_low_vram/MiniMax-H3-Image-Pruned-EntroPack.py)|-|-|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/lora/MiniMax-H3-Image-EntroPack.sh)|[code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/minimax_h3_image/model_training/validate_lora/MiniMax-H3-Image-EntroPack.py)|

## 模型推理

MiniMax-H3-Image 有两种工作模式：文生图，以及基于参考图的图像编辑。二者都通过 `MiniMaxH3ImagePipeline` 完成，由 `edit_image` 参数选择具体模式，因此同一份脚本即可覆盖两种用法。提示词会被原样使用，不套用对话模板，建议使用较长且完整的描述。

### 文生图

```python
prompt = (
    "An overhead photograph of a rustic farm-to-table spread on a weathered wooden table: a round loaf of sourdough "
    "on burlap, wedges of cheese on a dark board, a copper bowl of vegetables, vine tomatoes, olives and walnuts, "
    "warm afternoon light raking across the grain of the wood."
)
image = pipe(prompt=prompt, height=4096, width=4096, seed=42)
image.save("MiniMax-H3-Image.jpg")
```

`height` 与 `width` 是返回图像的尺寸，默认 2048。模型在 400 万像素及以上生成效果最好：当请求尺寸小于 `min_generation_area`（默认 2048×2048 像素面积）时，模型会按相同画幅比例在更大的画布上生成，再缩放回请求尺寸，因此 1024×1024 的请求与 2048×2048 的请求开销相同；把 `min_generation_area` 设为 `1024` 可关闭该行为。无论哪种情况，画布边长都会对齐到 32 的倍数。

模型的有效工作区间约为 400 万到 2000 万像素，通常在 1600 万像素附近效果最佳：

| 画幅比例 | 尺寸（宽 x 高） |
|-|-|
| 1:1 | 4096x4096 |
| 4:3 / 3:4 | 4736x3552 / 3552x4736 |
| 3:2 / 2:3 | 5024x3360 / 3360x5024 |
| 16:9 / 9:16 | 5472x3072 / 3072x5472 |
| 21:9 | 6272x2688 |

### 图像编辑

```python
from PIL import Image

prompt = "Change the color of the tomato in <Picture 1> to green."
image = pipe(
    prompt=prompt, height=4096, width=4096, seed=43,
    edit_image=[Image.open("MiniMax-H3-Image.jpg")],
)
image.save("MiniMax-H3-Image-Edit.jpg")
```

提示词按顺序用 `<Picture 1>`、`<Picture 2>` 等指代参考图，最多支持 16 张参考图。参考图保持自身尺寸：仅当短边不在 1024 到 2048 像素之间时才会缩放，且绝不会被缩放到生成图像的尺寸。编辑提示词通常先说明要保留的内容，再说明要修改的内容，最后完整描述目标图像。

### 采样

- 模型在固定的 sigma 网格上进行 6 步采样，Transformer 每一步对应一个独立输出头：`distilled=True`（默认）时 `num_inference_steps` 必须是 6。
- 图像 VAE 以重叠分块（256 像素、至少 64 像素重叠）编解码并线性混合。分块是官方输出的组成部分，因此 `tiling=True`（默认）可以复现参考结果；`tiling=False` 会整图解码，显存占用高得多。
- 支持 `cfg_scale` 与 `negative_prompt`。该模型经过引导蒸馏，默认值（`cfg_scale=1.0`、空的 `negative_prompt`）即复现官方行为；`cfg_scale` 大于 1 会额外执行一次无条件分支，并按 `uncond + cfg_scale * (cond - uncond)` 合并，仅用于实验——蒸馏权重并未针对这种用法训练。
- `distilled=False` 配合 `num_inference_steps=N` 会切换到普通 flow-match 采样；此时 Transformer 的 6 个输出头按噪声水平就近选择，而不是按步序号。

在单张 H20（96 GB）上实测：快速开始的脚本把文本编码器、Transformer 与 VAE 全部驻留在显卡上，共需约 135 GB 显存，因此单张 96 GB 显卡放不下；低显存示例把文本编码器与 VAE 改为磁盘流式、把 Transformer 与每步输出头以 int8 存放在主机内存，生成一张 4096×4096 图像再对其做一次编辑，含模型加载共约 4.5 分钟，显存峰值约 38 GiB。

## 不同硬件上的推理方案

MiniMax-H3-Image 的模型参数量非常大，针对不同的硬件条件，我们提供了多种推理方案。

### 消费级 NVIDIA GPU RTX 4090

以 RTX 4090 为例，GPU 显存只有 24G，因此需要对模型进行量化压缩，我们推荐使用 DiffSynth + EntroPack 进行推理。

安装 Python 环境：

```shell
pip install diffsynth==2.2.2
pip install cupy-cuda12x  # or cupy-cuda13x
```

#### 方案一：24GB 显存 + 32GB 内存

开启 INT8 量化 + ConvRot 提升精度，DiT 部分开启动态 CPU Offload，Text Encoder 部分开启 Disk Offload 流式加载和推理，图像生成质量几乎无损。

```python
import torch
from diffsynth.pipelines.minimax_h3_image import MiniMaxH3ImagePipeline, ModelConfig

quant_vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
offload_vram_config = {
    # 如果有更多内存，可以设置 offload_vram_config = quant_vram_config
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = MiniMaxH3ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="DiffSynth-Studio/MiniMax-H3-Image-EntroPack",
            origin_file_pattern="models/dit_8bit_int8.safetensors",
            origin_config_file="models/dit_8bit_int8.json",
            **quant_vram_config,
        ),
        ModelConfig(
            model_id="DiffSynth-Studio/EntroPackPreQuants",
            origin_file_pattern="models/MiniMax-H3/text_encoder_int8_convrot.safetensors",
            origin_config_file="models/MiniMax-H3/text_encoder_int8_convrot.json",
            **offload_vram_config,
        ),
        ModelConfig(
            model_id="MiniMax/MiniMax-H3-Image",
            origin_file_pattern="pdd_fused_heads.safetensors",
            **quant_vram_config,
        ),
        ModelConfig(
            model_id="DiffSynth-Studio/MiniMax-H3-Image-EntroPack",
            origin_file_pattern="models/vae_int8.safetensors",
            **quant_vram_config,
        ),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="processor/"),
    vram_limit=20,
)
prompt = "A cat is sitting on the table. A book is on the table."
image = pipe(prompt=prompt, height=4096, width=4096, seed=42)
image.save("MiniMax-H3-Image-Pruned-EntroPack.jpg")
```

#### 方案二：24GB 显存 + 8GB 内存

以 8.5bit、5bit 和 2.9bit 混合量化到 3bit 的平均精度，无任何 Offload，生成大量图片时速度最快。

```python
import torch
from diffsynth.pipelines.minimax_h3_image import MiniMaxH3ImagePipeline, ModelConfig

pipe = MiniMaxH3ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="DiffSynth-Studio/MiniMax-H3-Image-EntroPack",
            origin_file_pattern="models/dit_extreme_3bpp.safetensors",
            origin_config_file="models/dit_extreme_3bpp.json",
        ),
        ModelConfig(
            model_id="DiffSynth-Studio/EntroPackPreQuants",
            origin_file_pattern="models/MiniMax-H3/text_encoder_3bpp.safetensors",
            origin_config_file="models/MiniMax-H3/text_encoder_3bpp.json",
        ),
        ModelConfig(
            model_id="MiniMax/MiniMax-H3-Image",
            origin_file_pattern="pdd_fused_heads.safetensors",
        ),
        ModelConfig(
            model_id="DiffSynth-Studio/MiniMax-H3-Image-EntroPack",
            origin_file_pattern="models/vae_int8.safetensors",
        ),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="processor/"),
)
prompt = "A cat is sitting on the table. A book is on the table."
image = pipe(prompt=prompt, height=4096, width=4096, seed=42)
image.save("MiniMax-H3-Image-Pruned-EntroPack.jpg")
```

### NVIDIA GPU H20

H20 是目前广泛用于数据中心服务器上的低成本推理 GPU，使用 DiffSynth + EntroPack，首先使用 DiffSynth 中的自定义注意力算子，启用 Sage Attention 中量化后的注意力算子（以下代码需要在 `import diffsynth` 之前执行）。

```python
import os
os.environ["DIFFSYNTH_ATTENTION_IMPLEMENTATION"] = "sage_attention_h20_fast"
from sageattention import sageattn_qk_int8_pv_fp8_cuda_sm90
from diffsynth.core.attention.common import rearrange_out, rearrange_qkv
from diffsynth.core.attention.registry import register_attention_implementation

def sage_attention_h20_fast(q, k, v, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    required_in_pattern, required_out_pattern = "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = sageattn_qk_int8_pv_fp8_cuda_sm90(
        q, k, v,
        tensor_layout="NHD",
        qk_quant_gran="per_warp",
        pv_accum_dtype="fp32+fp32",
        smooth_k=False,
        is_causal=False,
        sm_scale=scale,
    )
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out

register_attention_implementation(
    "sage_attention_h20_fast",
    sage_attention_h20_fast,
    supports_causal=False,
    supports_window=False,
    supports_mask=False,
)
```

启用 INT8 量化 + ConvRot 提升精度 + 编译加速，开始模型推理：

```python
import torch
from diffsynth.pipelines.minimax_h3_image import MiniMaxH3ImagePipeline, ModelConfig

pipe = MiniMaxH3ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="DiffSynth-Studio/MiniMax-H3-Image-EntroPack",
            origin_file_pattern="models/dit_8bit_int8.safetensors",
            origin_config_file="models/dit_8bit_int8.json",
        ),
        ModelConfig(
            model_id="DiffSynth-Studio/EntroPackPreQuants",
            origin_file_pattern="models/MiniMax-H3/text_encoder_int8_convrot.safetensors",
            origin_config_file="models/MiniMax-H3/text_encoder_int8_convrot.json",
        ),
        ModelConfig(
            model_id="MiniMax/MiniMax-H3-Image",
            origin_file_pattern="pdd_fused_heads.safetensors",
        ),
        ModelConfig(
            model_id="DiffSynth-Studio/MiniMax-H3-Image-EntroPack",
            origin_file_pattern="models/vae_int8.safetensors",
        ),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="processor/"),
)
pipe.compile_pipeline()
prompt = "A cat is sitting on the table. A book is on the table."
image = pipe(prompt=prompt, height=4096, width=4096, seed=42)
image.save("MiniMax-H3-Image-Pruned-EntroPack.jpg")
```

这个推理方案需要约 50GB 显存，首次推理有减速效果，其后每次推理都会加速，适合部署线上推理服务。

## 模型训练

MiniMax-H3-Image 支持全量微调与 LoRA 微调。33.1B 的 Transformer 在 bf16 下占 66.3 GB，文本编码器另占 66.7 GB，因此两种训练配方都采用两阶段拆分流程：第一阶段（`--task "sft:data_process"`）对数据集做预处理并缓存文本嵌入与 VAE latent，此时 Transformer 卸载到磁盘；第二阶段（`--task "sft:train"`）在缓存数据集上训练，文本编码器与 VAE 卸载到磁盘。`--lora_target_modules` 留空，由框架自行决定 LoRA 加在哪些模块上；对该 Transformer 而言，它会覆盖各 block 堆栈的注意力与前馈投影，以及每个 block 的 AdaLN 投影，秩为 32。

训练数据是一个图片目录加一份 `metadata.csv`，其中记录图片文件名与提示词：

```csv
image,prompt
1.jpg,"a professional photo of a cute white and brown dog sitting on a wall under cherry blossoms"
```

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3_image/MiniMax-H3-Image/*" --local_dir ./data/diffsynth_example_dataset

# 第一阶段：缓存确定性的预处理结果。
accelerate launch examples/minimax_h3_image/model_training/train.py \
  --dataset_base_path ./data/diffsynth_example_dataset/minimax_h3_image/MiniMax-H3-Image \
  --dataset_metadata_path ./data/diffsynth_example_dataset/minimax_h3_image/MiniMax-H3-Image/metadata.csv \
  --data_file_keys "image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3-Image:transformer/model*.safetensors,MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors,MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-Image-split-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task "sft:data_process" \
  --offload_models "MiniMax/MiniMax-H3-Image:transformer/model*.safetensors,MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors"

# 第二阶段：在缓存数据集上训练 LoRA。
accelerate launch examples/minimax_h3_image/model_training/train.py \
  --dataset_base_path "./models/train/MiniMax-H3-Image-split-cache" \
  --data_file_keys "image" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3-Image:transformer/model*.safetensors,MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors,MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-Image-split-lora" \
  --lora_base_model "dit" \
  --lora_target_modules "" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task "sft:train" \
  --offload_models "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors"
```

编辑训练以注释形式附在 LoRA 与全量脚本的末尾：取消注释并删掉上方的文生图两阶段即可。它使用 `minimax_h3_image/MiniMax-H3-Image-Edit` 数据集，其 `metadata.csv` 多出一列 `edit_image` 记录参考图路径，`edit_image` 同时出现在 `--data_file_keys` 与 `--extra_inputs` 中，参考图会与目标图一同被缓存。

正是两阶段拆分让训练能够放进单张显卡：整个流程中 Transformer、文本编码器与 VAE 从不同时驻留。训练脚本使用 `--max_pixels 1048576`，把每个样本缩放到约 100 万像素；验证脚本则以与低显存示例相同的卸载配置加载训练产物。推荐训练脚本见上文「模型总览」表格。关于如何编写模型训练脚本，请参考[模型训练](../Pipeline_Usage/Model_Training.md)。
