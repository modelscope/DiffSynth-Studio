# 训练脚本模板与参数分析

**适用场景**：SKILL.md Step 6（创建训练脚本（.sh））创建 `.sh` 训练脚本时使用。本文件包含完整的脚本模板、参数分析、以及需配置项的决定方案。

> `train.py` 的代码模板和需配置项见 [references/training-code-templates.md](training-code-templates.md)。

## 一、`.sh` 训练脚本

### 1.1 LoRA 训练脚本

```bash
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "ernie_image/Ernie-Image-T2I/*" --local_dir ./data/diffsynth_example_dataset

# ===== 固定参数（无需修改） =====
accelerate launch examples/{series}/model_training/train.py \
    --learning_rate 1e-4 --num_epochs 5 \
    --lora_rank 32 \
    --lora_target_modules "" \
    --use_gradient_checkpointing --find_unused_parameters
```

```bash
# ===== 需配置参数（按下方说明逐一替换） =====
    --dataset_base_path "./data/diffsynth_example_dataset/{series}/{ModelName}" \
    --dataset_metadata_path "./data/diffsynth_example_dataset/{series}/{ModelName}/metadata.csv" \
    --model_id_with_origin_paths "model_id:transformer/*.safetensors,model_id:text_encoder/*.safetensors,model_id:vae/*.safetensors" \
    --lora_base_model "dit" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --max_pixels 1048576 \
    --dataset_repeat 50 \
    --output_path "./models/train/{ModelName}_lora" \
    --data_file_keys "image" \
    --extra_inputs "edit_image"
```

> **`--dataset_base_path` / `--dataset_metadata_path`**：`{series}` = 小写系列名，`{ModelName}` = 与训练脚本文件名一致。
>
> **`--model_id_with_origin_paths`**：从 `pipeline.py` 中的 ModelConfig 列表复制，格式 `"model_id:origin_pattern"`，逗号分隔。见下方配置表第 2 项。
>
> **`--lora_base_model`**：Pipeline 主模型属性名。多数为 `"dit"`，`self.video_dit` → `"video_dit"`，`self.vace` → `"vace"`。见下方第四节决策方案。
>
> **`--remove_prefix_in_ckpt`**：与 `--lora_base_model` 对应，格式为 `"pipe.{lora_base_model}."`。`lora_base_model="dit"` → `"pipe.dit."`，`"video_dit"` → `"pipe.video_dit."`，`"vace"` → `"pipe.vace."`。
>
> **`--max_pixels`**：图像任务用 max_pixels（1048576）；视频任务用 `--height/--width/--num_frames`，从 Pipeline 的 `__call__` 默认参数获取。
>
> **`--dataset_repeat`**：图像任务 50，视频任务 100。
>
> **`--output_path`**：`"./models/train/{ModelName}_lora"`，`{ModelName}` 替换为实际模型名。
>
> **`--lora_target_modules`**：固定写 `""`，交给框架自动探测，见下方第三节。**这一行必须显式保留**——省略该参数不等于自动探测，会退回 parser 默认值 `"q,k,v,o,ffn.0,ffn.2"`（Wan 专用层名），非 Wan 架构启动即报 `ValueError`。
>
> **`--data_file_keys`**：列出 metadata.csv 中所有**包含文件路径的列**（image/video/audio 等），逗号分隔。这些列会被 `UnifiedDataset` 通过 `special_operator_map` / `main_data_operator` 加载为 tensor/PIL 对象。纯文本列（prompt）不需要。例如基础 T2I 用 `"image"`，图像编辑用 `"image,edit_image"`，音视频联合用 `"video,input_audio"`。
>
> **`--extra_inputs`**：列出 metadata.csv 中**需要注入 Pipeline 推理流程**的额外列名（不包括 image/video/prompt），逗号分隔。这些值通过 `parse_extra_inputs` 进入 `inputs_shared`，供 Pipeline Units 使用。**同一个列名可能同时出现在 data_file_keys 和 extra_inputs 中**——前者负责文件加载，后者负责值注入。例如图像编辑用 `"edit_image"`（同时也在 data_file_keys 中），音视频联合用 `"input_audio"`（也在 data_file_keys 中），多层图像用 `"layer_num,layer_input_image"`。无额外输入时可省略此行。

### 1.2 全量训练脚本

```bash
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "ernie_image/Ernie-Image-T2I/*" --local_dir ./data/diffsynth_example_dataset

# ===== 固定参数（无需修改） =====
accelerate launch --config_file examples/{series}/model_training/full/accelerate_config_zero3.yaml \
    examples/{series}/model_training/train.py \
    --learning_rate 1e-5 --num_epochs 2 \
    --trainable_models "dit" --use_gradient_checkpointing --find_unused_parameters
```

```bash
# ===== 需配置参数（按下方说明逐一替换） =====
    --dataset_base_path "./data/diffsynth_example_dataset/{series}/{ModelName}" \
    --dataset_metadata_path "./data/diffsynth_example_dataset/{series}/{ModelName}/metadata.csv" \
    --model_id_with_origin_paths "model_id:transformer/*.safetensors,model_id:text_encoder/*.safetensors,model_id:vae/*.safetensors" \
    --remove_prefix_in_ckpt "pipe.dit." \
    --max_pixels 1048576 \
    --dataset_repeat 50 \
    --output_path "./models/train/{ModelName}_full" \
    --data_file_keys "image" \
    --extra_inputs "edit_image"
```

> **`--dataset_base_path` / `--dataset_metadata_path`**：`{series}` = 小写系列名，`{ModelName}` = 与训练脚本文件名一致。
>
> **`--model_id_with_origin_paths`**：从 `pipeline.py` 中的 ModelConfig 列表复制，格式 `"model_id:origin_pattern"`，逗号分隔。见下方配置表第 2 项。
>
> **`--remove_prefix_in_ckpt`**：与 `--trainable_models` 对应，格式为 `"pipe.{trainable_models}."`。通常 `"dit"` → `"pipe.dit."`，例外：VAP 模型用 `"pipe.vap."`。
>
> **`--max_pixels`**：图像任务用 max_pixels（1048576）；视频任务用 `--height/--width/--num_frames`，从 Pipeline 的 `__call__` 默认参数获取。
>
> **`--dataset_repeat`**：图像任务 50，视频任务 100。
>
> **`--output_path`**：`"./models/train/{ModelName}_full"`，`{ModelName}` 替换为实际模型名。
>
> **`--data_file_keys`**：列出 metadata.csv 中所有**包含文件路径的列**（image/video/audio 等），逗号分隔。这些列会被 `UnifiedDataset` 通过 `special_operator_map` / `main_data_operator` 加载为 tensor/PIL 对象。纯文本列（prompt）不需要。例如基础 T2I 用 `"image"`，图像编辑用 `"image,edit_image"`，音视频联合用 `"video,input_audio"`。
>
> **`--extra_inputs`**：列出 metadata.csv 中**需要注入 Pipeline 推理流程**的额外列名（不包括 image/video/prompt），逗号分隔。这些值通过 `parse_extra_inputs` 进入 `inputs_shared`，供 Pipeline Units 使用。**同一个列名可能同时出现在 data_file_keys 和 extra_inputs 中**——前者负责文件加载，后者负责值注入。例如图像编辑用 `"edit_image"`（同时也在 data_file_keys 中），音视频联合用 `"input_audio"`（也在 data_file_keys 中），多层图像用 `"layer_num,layer_input_image"`。无额外输入时可省略此行。

### 1.3 LoRA vs Full 脚本差异对照

| 差异项 | LoRA 脚本 | Full 脚本 |
|--------|-----------|-----------|
| `accelerate launch` | 直接运行 | `--config_file .../accelerate_config_zero3.yaml` |
| `--learning_rate` | `1e-4` | `1e-5` |
| `--num_epochs` | `5` | `2` |
| `--dataset_repeat` | `50`（图像）/ `100`（视频） | `50`（图像）/ `100`（视频） |
| `--lora_base_model` | 需配置（见下方决策方案） | 不存在 |
| `--lora_target_modules` | `""`（固定，自动探测） | 不存在 |
| `--lora_rank` | `32` | 不存在 |
| `--trainable_models` | 不存在 | `"dit"` |
| `--output_path` | `*_lora` | `*_full` |
| `--find_unused_parameters` | 始终包含 | 始终包含 |

---

## 二、参数分类：固定 vs 需配置

基于 DiffSynth-Studio 中 ~115+ 个已有训练脚本的分析。

### 2.1 完全固定参数（所有模型一致，直接复制）

| 参数 | LoRA 值 | Full 值 | 说明 |
|------|---------|---------|------|
| `--learning_rate` | `1e-4` | `1e-5` | 固定约定，不随模型变化 |
| `--lora_rank` | `32` | — | 默认值，用户可调整 |
| `--lora_target_modules` | `""` | — | 自动探测，**必须显式写出**（见第三节） |
| `--use_gradient_checkpointing` | 始终包含 | 始终包含 | 强制参数，不可省略 |
| `--trainable_models` | — | `"dit"` | 固定值 |
| `--find_unused_parameters` | 始终包含 | 始终包含 | 固定，两种训练都需要 |

### 2.2 需根据模型配置（共 10 项）

| # | 参数 | 决定方案 | 参考来源 |
|---|------|---------|---------|
| 1 | `{series}` / `{ModelName}` | 从 SKILL.md Step 3（分析训练需求）获取。`{series}` = 小写系列名（目录名），`{ModelName}` = 与训练脚本文件名一致 | SKILL.md Step 3 |
| 1.1 | `--dataset_base_path` / `--dataset_metadata_path` | 基于 #1 的 `{series}` 和 `{ModelName}` 拼接路径 | 同 #1 |
| 2 | `--model_id_with_origin_paths` | **从 pipeline.py 中的 ModelConfig 列表复制**，将每个 `ModelConfig(model_id="...", origin_file_pattern="...")` 转为 `"model_id:origin_pattern"` 逗号分隔格式 | `pipeline.py` 中的 `from_pretrained` 调用 |
| 3 | `--lora_base_model` | 根据 Pipeline 主模型属性名决定（见下方决策方案），**不是固定 `"dit"`**（仅 LoRA） | 见下方决策方案 |
| 3.1 | `--remove_prefix_in_ckpt` | 与主模型属性名对应，格式 `"pipe.{属性名}."`。LoRA 对应 `--lora_base_model`，Full 对应 `--trainable_models` | 同 #3 |
| 4 | `--data_file_keys` | 列出 metadata.csv 中所有**包含文件路径的列**（image/video/audio 等），逗号分隔。这些列会被 `UnifiedDataset` 加载为 tensor/PIL。纯文本列（prompt）不需要 | SKILL.md Step 4b（确认字段与映射关系）输出的列名规划表 |
| 5 | `--extra_inputs` | 列出 metadata.csv 中**需要注入 Pipeline 推理流程**的额外列名，逗号分隔。这些值通过 `parse_extra_inputs` 进入 `inputs_shared`。**同一个列名可能同时出现在 data_file_keys 和 extra_inputs 中**（前者加载文件，后者注入值） | SKILL.md Step 4b（确认字段与映射关系）输出的列名规划表 |
| 6 | 数据集下载行 | `modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "{series}/{ModelName}/*" --local_dir ./data/diffsynth_example_dataset` | `{series}/{ModelName}` 来自 #1 |
| 7 | `--max_pixels` 或 `--height/--width/--num_frames` | 图像用 max_pixels（1048576）；视频用具体尺寸，从 Pipeline 的 `__call__` 默认参数获取 | Pipeline 的 `__call__` 默认参数或同 series 已有脚本 |
| 8 | `--dataset_repeat` | 图像 50，视频 100 | 任务类型决定 |
| 9 | `--output_path` | `"./models/train/{ModelName}_lora"` 或 `"./models/train/{ModelName}_full"` | 由 `{ModelName}` 和训练类型决定 |

---

## 三、`--lora_target_modules`：默认 `""`（自动探测）

**一律写 `--lora_target_modules ""`**，不再按架构手填层名列表。空字符串会走 `parse_lora_target_modules` 的自动探测分支（`diffsynth/diffusion/training_module.py`），由框架自己找出该 DiT 上要注入的层。上游已有先例：`anima/anima-preview.sh`、`stable_diffusion/stable-diffusion-v1-5.sh`。

### 3.1 自动探测做了什么

`auto_detect_lora_target_modules` 从模型根节点递归 `named_children()`：

- 遇到 `nn.ModuleList` 且 `len > 1`（即 transformer block 列表）时，收集其内部所有 `nn.Linear`
- 只收 `min(in_features, out_features) >= 512` 的 Linear
- 输出**逐 index 的完整路径**（如 `blocks.0.attn.to_q`），精确匹配，不存在后缀歧义

所以探测结果通常比手写列表**更宽**：attention 的 q/k/v/out、FFN、以及宽度达标的 modulation / norm linear 都会被覆盖，且无需了解该架构的层命名习惯。

**⚠️ 必须显式写 `""`，不能整行省略。** `parsers.py` 中 `--lora_target_modules` 的 argparse 默认值是 `"q,k,v,o,ffn.0,ffn.2"`（Wan 架构的层名）。省略该参数不会触发自动探测，而是把这份 Wan 专用列表套到任意架构上：peft 按 `.q` / `.ffn.0` 这样的后缀精确匹配，非 Wan 架构往往一个都匹配不上，`inject_adapter_in_model` 会在启动时抛 `ValueError: Target modules ... not found in the base model`；只命中一部分层时不报错，但注入范围比预期窄。

### 3.2 自动探测的盲区（何时改回显式列表）

探测只覆盖「block 列表内部的宽 Linear」，以下情况会漏：

| 盲区 | 典型例子 | 处理 |
|---|---|---|
| block 列表**之外**的层 | 顶层 `proj_out`、patch embed（`img_in` / `txt_in`）、`time_embedding` | 若确需训练这些层，显式列出 |
| `min(in, out) < 512` 的窄 Linear | 小维度的 modulation / scale-shift | 同上 |
| block 不是 `nn.ModuleList`，或列表长度 ≤ 1 | 自定义容器、单 block 结构 | 探测结果为空，必须显式列出 |

**判断方法**：看训练首屏日志。自动探测会打印 `No LoRA target modules specified. The framework will automatically search for them.` 和 `LoRA will be patched at [...]`。列表为空或缺少预期层时，才回退到 §3.3 兜底表显式指定。

### 3.3 兜底表：各系列显式取值

仅在 §3.2 判定需要显式指定时查本表。取值来自各系列 `model_training/` 下 `.sh` 脚本的实际值，本表可能滞后，**优先以该系列已有脚本为准**。

| DiT 架构类型 | 代表 Series | `lora_target_modules` 值 |
|---|---|---|
| **FLUX dual-stream** | `flux` | `a_to_qkv,b_to_qkv,ff_a.0,ff_a.2,ff_b.0,ff_b.2,a_to_out,b_to_out,proj_out,norm.linear,norm1_a.linear,norm1_b.linear,to_qkv_mlp` |
| **FLUX.2** | `flux2` | `to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj` + 逐个展开的 `single_transformer_blocks.{i}.attn.to_out` |
| **Qwen-Image** | `qwen_image` | `to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1` |
| **Wan-style** (MMDiT) | `wanvideo` | `q,k,v,o,ffn.0,ffn.2` |
| **LongCat-Video** | `wanvideo` (LongCat-Video) | `adaLN_modulation.1,attn.qkv,attn.proj,cross_attn.q_linear,cross_attn.kv_linear,cross_attn.proj,ffn.w1,ffn.w2,ffn.w3` |
| **LTX2** | `ltx2` | `to_k,to_q,to_v,to_out.0` |
| **MoVA** | `mova` | `q,k,v,o,ffn.0,ffn.2` |
| **Z-Image** | `z_image` | `to_q,to_k,to_v,to_out.0,w1,w2,w3` |
| **MiniMax-H3** | `minimax_h3` | `attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2` |
| **ACE-Step** | `ace_step` | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| **ERNIE-Image** | `ernie_image` | `to_q,to_k,to_v,to_out.0` |

**⚠️ 显式指定时不要使用通配符。** `--lora_target_modules` 非空时只做 `split(",")` 后交给 peft 按模块名后缀精确匹配，`single_transformer_blocks.*.attn.to_out` 这样的写法**不会命中任何模块**。需要逐 block 生效时必须像 flux2 那样把每个 index 显式列出——而这正是 `""` 自动探测已经替你做的事，也是优先用 `""` 的理由之一。

### 3.4 显式指定时的取值流程

```
§3.2 判定必须显式指定（自动探测漏层 / 结果为空）
│
├── 同 series 已有训练脚本
│   └── 直接复用其 lora_target_modules 值，不要删减
│
├── 属于 §3.3 兜底表的架构类型
│   └── 使用对应行的值
│
└── 全新架构（表中没有）
    └── 查 DiT 源码：在 diffsynth/models/ 中找到目标 DiT 类，
        看 forward 中的层命名模式，
        提取 attention 的 q/k/v/out 投影 + FFN 的线性层名，
        再补上探测漏掉的 block 外层（如 proj_out）
```

---

## 四、`--lora_base_model` 决策方案

`--lora_base_model` 指定 LoRA 权重注入到 Pipeline 的哪个模型属性上。**该值必须与 Pipeline 类中主扩散模型的属性名一致**，而非固定为 `"dit"`。

`lora_base_model` 在 `split_pipeline_units` 中用于确定哪些模型需要反向传播梯度，因此必须精确匹配 Pipeline 的属性名。

### 4.1 如何确定

1. **查看 Pipeline 的 `__init__` 或 `from_pretrained`**：找到主扩散模型被赋值给 `self.xxx` 的属性名
   ```python
   # 在 pipeline.py 中查找：
   self.dit = ...         # → lora_base_model = "dit"
   self.video_dit = ...   # → lora_base_model = "video_dit"
   self.vace = ...        # → lora_base_model = "vace"
   ```

2. **查看 Pipeline 的 `__call__` 方法**：推理时调用哪个模型作为主生成器
   ```python
   # 在 pipeline.py 的 __call__ 中查找：
   noise_pred = self.dit(...)         # → "dit"
   noise_pred = self.video_dit(...)   # → "video_dit"
   ```

### 4.2 已有模型的实际值

| Pipeline 属性名 | `lora_base_model` 值 | 代表 Series | 说明 |
|---------------|---------------------|-------------|------|
| `self.dit` | `"dit"` | flux, flux2, qwen_image, wanvideo(大部分), ltx2, z_image, anima | 绝大多数模型 |
| `self.video_dit` | `"video_dit"` | mova | MoVA 视频模型 |
| `self.vace` | `"vace"` | wanvideo (Wan2.1-VACE, Wan2.2-VACE-Fun) | Wan VACE 控制模块 |

### 4.3 决策流程

```
打开 pipeline.py → 搜索 "self.xxx = ..." 找到主扩散模型属性名
→ 如果是 self.dit → "dit"
→ 如果是 self.video_dit → "video_dit"
→ 如果是 self.vace → "vace"
→ 其他名称 → 使用实际属性名
```

---

## 五、验证脚本

验证脚本直接从**同名推理脚本复制**，仅做两处修改：在 `from_pretrained` 后插入训练权重加载，将输入数据替换为训练集文件。

### 5.1 LoRA 验证脚本模板

**基础 T2I 示例**（推理脚本 → 验证脚本）：

```python
# 从 examples/{series}/model_inference/{ModelName}.py 复制，插入 LoRA 加载 + 训练集数据
from diffsynth.pipelines.{series_name} import {SeriesName}Pipeline, ModelConfig
import torch

pipe = {SeriesName}Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        # 与推理脚本一致，完整复制
        ModelConfig(model_id="...", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="...", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="...", origin_file_pattern="vae/*.safetensors"),
    ],
)

# ===== 新增：加载 LoRA 权重 =====
pipe.load_lora(pipe.{base_model}, "models/train/{ModelName}_lora/epoch-{N}.safetensors")
# {base_model} 与训练脚本 --lora_base_model 一致

# ===== 修改：使用训练集 prompt =====
prompt = "data/diffsynth_example_dataset/{series}/{ModelName}/metadata.csv 中的 prompt"

image = pipe(
    prompt=prompt,
    negative_prompt="",
    height=1024, width=1024,  # 与推理脚本调用参数一致
    seed=0,
    num_inference_steps=50,
    cfg_scale=4.0,
)
image.save("image_lora.jpg")
```

**图像编辑示例**（带 edit_image）：

```python
# 从推理脚本复制，插入 LoRA 加载
from diffsynth.pipelines.{series_name} import {SeriesName}Pipeline, ModelConfig
from PIL import Image
import torch

pipe = {SeriesName}Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="...", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="...", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="...", origin_file_pattern="vae/*.safetensors"),
    ],
    processor_config=ModelConfig(model_id="...", origin_file_pattern="processor/"),
)

# ===== 新增：加载 LoRA 权重 =====
pipe.load_lora(pipe.{base_model}, "models/train/{ModelName}_lora/epoch-{N}.safetensors")

# ===== 修改：使用训练集数据 =====
prompt = "metadata.csv 中的 prompt"
edit_image = Image.open("data/diffsynth_example_dataset/{series}/{ModelName}/images/001.jpg")

image = pipe(
    prompt, edit_image=edit_image, seed=0,
    num_inference_steps=40, height=1024, width=1024,
)
image.save("image_lora.jpg")
```

### 5.2 Full 验证脚本模板

**基础 T2I 示例**：

```python
# 从推理脚本复制，插入全量权重加载 + 训练集数据
from diffsynth.pipelines.{series_name} import {SeriesName}Pipeline, ModelConfig
from diffsynth import load_state_dict
import torch

pipe = {SeriesName}Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        # 与推理脚本一致
        ModelConfig(model_id="...", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="...", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="...", origin_file_pattern="vae/*.safetensors"),
    ],
)

# ===== 新增：加载全量权重 =====
state_dict = load_state_dict("./models/train/{ModelName}_full/epoch-{N}.safetensors")
pipe.{base_model}.load_state_dict(state_dict)

# ===== 修改：使用训练集 prompt =====
prompt = "metadata.csv 中的 prompt"

image = pipe(
    prompt=prompt,
    negative_prompt="",
    height=1024, width=1024,
    seed=0,
    num_inference_steps=50,
    cfg_scale=4.0,
)
image.save("image_full.jpg")
```

### 5.3 视频任务验证脚本

**LoRA 视频示例**：

```python
# 从推理脚本复制
from diffsynth.pipelines.{series_name} import {SeriesName}Pipeline, ModelConfig
from diffsynth.utils.data import save_video
import torch

pipe = {SeriesName}Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="...", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="...", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="...", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="...", origin_file_pattern="google/umt5-xxl/"),
)

# ===== 新增：加载 LoRA 权重 =====
pipe.load_lora(pipe.{base_model}, "models/train/{ModelName}_lora/epoch-{N}.safetensors")

# ===== 修改：使用训练集 prompt =====
prompt = "metadata.csv 中的 prompt"

video = pipe(
    prompt=prompt,
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清...",
    seed=0, tiled=True,
    height=720, width=1280, num_frames=121,  # 与推理脚本一致
)
save_video(video, "video_lora.mp4", fps=15, quality=5)
```

**Full 视频示例**：

```python
# 与 LoRA 唯一区别是权重加载方式
from diffsynth.pipelines.{series_name} import {SeriesName}Pipeline, ModelConfig
from diffsynth import load_state_dict
from diffsynth.utils.data import save_video
import torch

pipe = {SeriesName}Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="...", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="...", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="...", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(model_id="...", origin_file_pattern="google/umt5-xxl/"),
)

# ===== 加载全量权重 =====
state_dict = load_state_dict("./models/train/{ModelName}_full/epoch-{N}.safetensors")
pipe.{base_model}.load_state_dict(state_dict)

prompt = "metadata.csv 中的 prompt"

video = pipe(
    prompt=prompt,
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清...",
    seed=0, tiled=True,
    height=720, width=1280, num_frames=121,
)
save_video(video, "video_full.mp4", fps=15, quality=5)
```

### 5.4 需配置项总结

| 配置项 | 决定方案 |
|--------|---------|
| Pipeline import | 与推理脚本一致，`{series_name}` / `{SeriesName}` 从同系列已有脚本复制 |
| `model_configs` 列表 | 与推理脚本一致，完整复制 |
| `tokenizer_config` / `processor_config` | 与推理脚本一致 |
| LoRA 权重加载 | `pipe.load_lora(pipe.{base_model}, "models/train/{ModelName}_lora/epoch-{N}.safetensors")` |
| Full 权重加载 | `state_dict = load_state_dict("path")` → `pipe.{base_model}.load_state_dict(state_dict)` |
| `{base_model}` | 与训练脚本 `--lora_base_model` 一致，对应 Pipeline 属性名 |
| epoch 数字 | LoRA 用 `num_epochs - 1`（如 epoch-4），Full 通常为 epoch-1 |
| prompt / 输入数据 | 从训练集 `metadata.csv` 中取第一条记录对应的 prompt 和数据文件 |
| Pipeline 调用参数 | **与推理脚本完全一致**，仅替换数据来源 |
