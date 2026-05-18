# 启用 DeepSpeed

训练框架基于 `accelerate` 与 `deepspeed` 构建，因此原生地支持启用 DeepSpeed 的训练特性。

## 配置训练参数

DeepSpeed 参数可通过 `accelerate config` 在终端交互式地配置。

* DeepSpeed ZeRO Stage 1：对优化器状态进行分片，在与 DDP（分布式数据并行）保持速度一致的同时，提供内存优化。
* DeepSpeed ZeRO Stage 2：对优化器状态和梯度进行分片，在与 DDP 保持速度一致的同时，提供更显著的内存优化。
* DeepSpeed ZeRO Stage 2 Offload：将优化器状态和梯度卸载到 CPU。会增加分布式通信量以及 GPU-CPU 设备间的数据传输开销，但能带来大幅内存节省。
* DeepSpeed ZeRO Stage 3：对优化器状态、梯度、模型参数（可选包括激活值）进行分片。会增加分布式通信量，但能提供更强的内存优化效果。
* DeepSpeed ZeRO Stage 3 Offload：将优化器状态、梯度、模型参数（可选包括激活值）全部卸载到 CPU。会显著增加分布式通信量和 GPU-CPU 数据传输开销，但可实现更极致的内存节省。

## DeepSpeed ZeRO Stage 3

DeepSpeed ZeRO Stage 3 是多卡训练中显存占用较小的训练模式，但需要修改部分配置文件。我们为部分模型提供了样例，主要通过 `--config_file` 指定 `deepspeed` 配置。

需要注意的是，`deepspeed_zero3_offload` 模式与 `pytorch` 原生的梯度检查点机制不兼容，我们为此对 `deepspeed` 的`checkpointing` 接口做了适配。用户需要在 `deepspeed` 配置中填写 `activation_checkpointing` 字段以启用梯度检查点。

以下为 Qwen-Image 模型的低显存模型训练脚本，脚本中同时开启了两阶段拆分训练：

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
  --output_path "./models/train/Qwen-Image_lora-splited-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --task "sft:data_process" \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters

accelerate launch --config_file examples/qwen_image/model_training/special/low_vram_training/deepspeed_zero3_cpuoffload.yaml examples/qwen_image/model_training/train.py \
  --dataset_base_path "./models/train/Qwen-Image_lora-splited-cache" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --task "sft:train" \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --initialize_model_on_cpu
```

其中，`accelerate` 和 `deepspeed` 的配置文件如下：

```yaml
compute_environment: LOCAL_MACHINE
debug: true
deepspeed_config:
  deepspeed_config_file: examples/qwen_image/model_training/special/low_vram_training/ds_z3_cpuoffload.json
  zero3_init_flag: true
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": false,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 5e7,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_max_live_parameters": 1e8,
        "stage3_max_reuse_distance": 1e8,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "activation_checkpointing": {
        "partition_activations": false,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": false
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```
