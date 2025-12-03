# 两阶段拆分训练

本文档介绍拆分训练，能够自动将训练过程拆分为两阶段进行，减少显存占用，同时加快训练速度。

（拆分训练是实验性特性，尚未进行大规模验证，如果在使用中出现问题，请在 GitHub 上给我们提 issue。）

## 拆分训练

在大部分模型的训练过程中，大量计算发生在“前处理”中，即“与去噪模型无关的计算”，包括 VAE 编码、文本编码等。当对应的模型参数固定时，这部分计算的结果是重复的，在多个 epoch 中每个数据样本的计算结果完全相同，因此我们提供了“拆分训练”功能，该功能可以自动分析并拆分训练过程。

对于普通文生图模型的标准监督训练，拆分过程是非常简单的，只需要把所有 [`Pipeline Units`](/docs/zh/Developer_Guide/Building_a_Pipeline.md#units) 的计算拆分到第一阶段，将计算结果存储到硬盘中，然后在第二阶段从硬盘中读取这些结果并进行后续计算即可。但如果前处理过程中需要梯度回传，情况就变得极其复杂，为此，我们引入了一个计算图拆分算法用于分析如何拆分计算。

## 计算图拆分算法

> （我们会在后续的文档更新中补充计算图拆分算法的详细细节）

## 使用拆分训练

拆分训练已支持[标准监督训练](/docs/zh/Training/Supervised_Fine_Tuning.md)和[直接蒸馏训练](/docs/zh/Training/Direct_Distill.md)，在训练命令中通过 `--task` 参数控制，以 Qwen-Image 模型的 LoRA 训练为例，拆分前的训练命令为：

```shell
accelerate launch examples/qwen_image/model_training/train.py \
  --dataset_base_path data/example_image_dataset \
  --dataset_metadata_path data/example_image_dataset/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters
```

拆分后，在第一阶段中，做如下修改：

* 将 `--dataset_repeat` 改为 1，避免重复计算
* 将 `--output_path` 改为第一阶段计算结果保存的路径
* 添加额外参数 `--task "sft:data_process"`
* 删除 `--model_id_with_origin_paths` 中的 DiT 模型

```shell
accelerate launch examples/qwen_image/model_training/train.py \
  --dataset_base_path data/example_image_dataset \
  --dataset_metadata_path data/example_image_dataset/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image-LoRA-splited-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --task "sft:data_process"
```

在第二阶段，做如下修改：

* 将 `--dataset_base_path` 改为第一阶段的 `--output_path`
* 删除 `--dataset_metadata_path`
* 添加额外参数 `--task "sft:train"`
* 删除 `--model_id_with_origin_paths` 中的 Text Encoder 和 VAE 模型

```shell
accelerate launch examples/qwen_image/model_training/train.py \
  --dataset_base_path "./models/train/Qwen-Image-LoRA-splited-cache" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image-LoRA-splited" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --task "sft:train"
```

我们提供了样例训练脚本和验证脚本，位于 `examples/qwen_image/model_training/special/split_training`。

## 训练框架设计思路

训练框架通过 `DiffusionTrainingModule` 的 `split_pipeline_units` 方法拆分 `Pipeline` 中的计算单元。
