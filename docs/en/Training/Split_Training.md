# Two-Stage Split Training

This document introduces split training, which can automatically divide the training process into two stages, reducing VRAM usage while accelerating training speed.

(Split training is an experimental feature that has not yet undergone large-scale validation. If you encounter any issues while using it, please submit an issue on GitHub.)

## Split Training

In the training process of most models, a large amount of computation occurs in "preprocessing," i.e., "computations unrelated to the denoising model," including VAE encoding, text encoding, etc. When the corresponding model parameters are fixed, the results of these computations are repetitive. For each data sample, the computational results are identical across multiple epochs. Therefore, we provide a "split training" feature that can automatically analyze and split the training process.

For standard supervised training of ordinary text-to-image models, the splitting process is straightforward. It only requires splitting the computation of all [`Pipeline Units`](/docs/en/Developer_Guide/Building_a_Pipeline.md#units) into the first stage, storing the computational results to disk, and then reading these results from disk in the second stage for subsequent computations. However, if gradient backpropagation is required during preprocessing, the situation becomes extremely complex. To address this, we introduced a computational graph splitting algorithm to analyze how to split the computation.

## Computational Graph Splitting Algorithm

> (We will supplement the detailed specifics of the computational graph splitting algorithm in future document updates)

## Using Split Training

Split training already supports [Standard Supervised Training](/docs/en/Training/Supervised_Fine_Tuning.md) and [Direct Distillation Training](/docs/en/Training/Direct_Distill.md). The `--task` parameter in the training command controls this. Taking LoRA training of the Qwen-Image model as an example, the pre-split training command is:

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

After splitting, in the first stage, make the following modifications:

* Change `--dataset_repeat` to 1 to avoid redundant computation
* Change `--output_path` to the path where the first-stage computation results are saved
* Add the additional parameter `--task "sft:data_process"`
* Remove the DiT model from `--model_id_with_origin_paths`

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

In the second stage, make the following modifications:

* Change `--dataset_base_path` to the `--output_path` of the first stage
* Remove `--dataset_metadata_path`
* Add the additional parameter `--task "sft:train"`
* Remove the Text Encoder and VAE models from `--model_id_with_origin_paths`

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

We provide sample training scripts and validation scripts located at `examples/qwen_image/model_training/special/split_training`.

## Training Framework Design Concept

The training framework splits the computational units in the `Pipeline` through the `split_pipeline_units` method of `DiffusionTrainingModule`.