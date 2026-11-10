# 拆分训练：单阶段脚本改写为双阶段

**适用场景**：LoRA 训练出现 OOM（单卡显存不足）时，将原 `lora/{ModelName}.sh` 脚本直接改写为双阶段脚本。不是创建新文件，而是直接修改原脚本。

## ⚠️ 关键规则：第一阶段 `--dataset_repeat` 必须为 1

**双阶段脚本中，只有第一阶段（`sft:data_process`）的 `--dataset_repeat` 设置为 `1`。第二阶段（`sft:train`）的 `--dataset_repeat` 保持原值不变（图像 50，视频 100），绝不降低。**

**原因**：第一阶段的作用是缓存数据集前处理结果到磁盘。设置 `--dataset_repeat 1` 意味着每个数据样本只缓存一次，这是缓存阶段的正确行为。如果设置更大的 repeat 值，第一阶段会对同一个数据样本重复缓存多次，产生大量冗余 `.pth` 文件，浪费磁盘空间并拖慢缓存速度。

**第二阶段** 是实际训练阶段，需要数据集重复采样来提供足够的训练轮次，因此 `--dataset_repeat` 必须保持原值（图像 50，视频 100），不可降低。

| 阶段 | `--dataset_repeat` | 原因 |
|------|-------------------|------|
| 第一阶段（缓存） | **改为 `1`** | 每个样本只需缓存一次 |
| 第二阶段（训练） | **保持原值（50 或 100）**，不降低 | 训练需要数据重复采样 |

## 原理

`sft` 模式在一个训练循环中同时完成前处理（VAE 编码、文本编码等）和 DiT 训练。当 VAE、TextEncoder 和 DiT 同时加载时，显存可能不足。

拆分训练将计算分为两个阶段：
1. **第一阶段**（`sft:data_process`）：只跑前处理计算（VAE 编码、文本编码等），把结果缓存到硬盘，DiT 通过 `--offload_models` 排除在 forward 之外。
2. **第二阶段**（`sft:train`）：从缓存读取前处理结果，只训练 DiT，前处理模型通过 `--offload_models` 排除。

**两阶段都保留完整的 `--model_id_with_origin_paths`，用 `--offload_models` 声明该阶段不需要 forward 的模型。** `--offload_models` 的格式与 `--model_id_with_origin_paths` 完全相同（逗号分隔的 `model_id:origin_pattern`）。

直接从 `--model_id_with_origin_paths` 中删掉条目也能跑，但那样必须自己确认该模型不会被 Pipeline 间接调用，需要了解 Pipeline 的运行细节。用 `--offload_models` 不需要这层判断。

## 改写步骤

打开 `lora/{ModelName}.sh`，将原来的**一条** `accelerate launch` 命令替换为**两条**命令。

### 第一步：复制原脚本中的 `accelerate launch` 参数

从原脚本中提取所有参数，记录以下关键值：
- 原始 `--model_id_with_origin_paths` 中的三个模型条目：`transformer`（DiT）、`text_encoder`、`vae`
- 原始 `--output_path`
- 原始 `--dataset_repeat`

### 第二步：构建第一阶段命令

将原 `accelerate launch` 命令改写为第一阶段：

| 参数 | 原值 | 修改后 |
|------|------|--------|
| `--task` | （无） | 追加 `--task "sft:data_process"` |
| `--dataset_repeat` | `50` | **`1`（必须为 1，不可使用其他值）** |
| `--output_path` | 原始路径 | `"./models/train/{ModelName}-LoRA-splited-cache"` |
| `--model_id_with_origin_paths` | 包含 DiT + TextEncoder + VAE | **保持不变** |
| `--offload_models` | （无） | 追加，填入 **DiT（transformer）条目** |

### 第三步：构建第二阶段命令

同样基于原脚本参数改写为第二阶段：

| 参数 | 原值 | 修改后 |
|------|------|--------|
| `--dataset_base_path` | 原始数据集路径 | `"./models/train/{ModelName}-LoRA-splited-cache"` |
| `--dataset_metadata_path` | 原始 metadata.csv 路径 | **删除此参数**（`UnifiedDataset` 检测不到 `metadata_path` 时自动进入 `load_from_cache` 模式，从 `.pth` 缓存读取） |
| `--model_id_with_origin_paths` | 包含 DiT + TextEncoder + VAE | **保持不变** |
| `--offload_models` | （无） | 追加，填入 **TextEncoder + VAE 条目** |
| `--output_path` | 原始路径 | `"./models/train/{ModelName}-LoRA-splited"` |
| `--task` | （无） | 追加 `--task "sft:train"` |
| `--dataset_repeat` | `50` | 保持 `50`（从缓存读取很快） |

### 第四步：替换原脚本内容

将 `lora/{ModelName}.sh` 的内容替换为两阶段结构：

```bash
#!/bin/bash

# ==================== 第一阶段：前处理计算 ====================
accelerate launch examples/{series}/model_training/train.py \
    --dataset_base_path "./data/diffsynth_example_dataset/{series}/{ModelName}" \
    --dataset_metadata_path "./data/diffsynth_example_dataset/{series}/{ModelName}/metadata.csv" \
    --max_pixels 1048576 --dataset_repeat 1 \
    --model_id_with_origin_paths "{text_encoder条目},{vae条目},{transformer条目}" \
    --offload_models "{transformer条目}" \
    --output_path "./models/train/{ModelName}-LoRA-splited-cache" \
    {其他通用参数} \
    --task "sft:data_process"

# ==================== 第二阶段：DiT 训练 ====================
accelerate launch examples/{series}/model_training/train.py \
    --dataset_base_path "./models/train/{ModelName}-LoRA-splited-cache" \
    --max_pixels 1048576 --dataset_repeat 50 \
    --model_id_with_origin_paths "{text_encoder条目},{vae条目},{transformer条目}" \
    --offload_models "{text_encoder条目},{vae条目}" \
    --output_path "./models/train/{ModelName}-LoRA-splited" \
    {其他通用参数} \
    --task "sft:train"
```

## 示例

以 Qwen-Image LoRA 训练为例（参考 `examples/qwen_image/model_training/special/split_training/Qwen-Image-LoRA.sh`）：

> 下例逐字复制上游脚本，因此保留了它那份显式 `--lora_target_modules` 列表，**不要照抄这串层名**。新建脚本时两阶段都写 `--lora_target_modules ""`（自动探测，见 [training-scripts.md](training-scripts.md) §3）。

```bash
#!/bin/bash

# ==================== 第一阶段：前处理计算 ====================
accelerate launch examples/qwen_image/model_training/train.py \
    --dataset_base_path "./data/diffsynth_example_dataset/qwen_image/Qwen-Image" \
    --dataset_metadata_path "./data/diffsynth_example_dataset/qwen_image/Qwen-Image/metadata.csv" \
    --max_pixels 1048576 --dataset_repeat 1 \
    --model_id_with_origin_paths "Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors,Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors" \
    --offload_models "Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors" \
    --learning_rate 1e-4 --num_epochs 5 \
    --remove_prefix_in_ckpt "pipe.dit." --output_path "./models/train/Qwen-Image-LoRA-splited-cache" \
    --lora_base_model "dit" --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" --lora_rank 32 \
    --use_gradient_checkpointing --dataset_num_workers 8 --find_unused_parameters \
    --task "sft:data_process"

# ==================== 第二阶段：DiT 训练 ====================
accelerate launch examples/qwen_image/model_training/train.py \
    --dataset_base_path "./models/train/Qwen-Image-LoRA-splited-cache" \
    --max_pixels 1048576 --dataset_repeat 50 \
    --model_id_with_origin_paths "Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors,Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors" \
    --offload_models "Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
    --learning_rate 1e-4 --num_epochs 5 \
    --remove_prefix_in_ckpt "pipe.dit." --output_path "./models/train/Qwen-Image-LoRA-splited" \
    --lora_base_model "dit" --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" --lora_rank 32 \
    --use_gradient_checkpointing --dataset_num_workers 8 --find_unused_parameters \
    --task "sft:train"
```

## 关键检查点

- [ ] `--model_id_with_origin_paths`：两阶段都保留完整模型列表，不删条目
- [ ] `--offload_models`：第一阶段填 DiT，第二阶段填 TextEncoder + VAE，格式与 `--model_id_with_origin_paths` 一致
- [ ] `--dataset_base_path`：第二阶段指向第一阶段的缓存路径
- [ ] `--dataset_metadata_path`：第二阶段删除
- [ ] `--dataset_repeat`：**第一阶段必须为 1**（不可使用其他值），第二阶段恢复原值（50 或 100）
- [ ] 第一阶段 `--task "sft:data_process"`，第二阶段 `--task "sft:train"`
- [ ] 第一阶段的 `--output_path` 和第二阶段的 `--dataset_base_path` 路径完全一致
- [ ] LoRA 相关参数（`lora_base_model`、`lora_target_modules`、`lora_rank`）两阶段都保留，且 `--lora_target_modules ""` 两阶段都显式写出（漏写会退回 parser 默认值）
- [ ] 通用参数（`learning_rate`、`num_epochs`、`use_gradient_checkpointing` 等）两阶段都保留

## 验证

双阶段训练完成后，使用原验证脚本加载第二阶段输出的 checkpoint（`./models/train/{ModelName}-LoRA-splited/` 下的对应文件）即可。
