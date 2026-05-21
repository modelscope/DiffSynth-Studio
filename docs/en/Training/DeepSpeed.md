# Enabling DeepSpeed

The training framework is built on `accelerate` and `deepspeed`, thus natively supporting DeepSpeed training features.

## Configuring Training Parameters

DeepSpeed parameters can be configured interactively in the terminal via `accelerate config`.

* DeepSpeed ZeRO Stage 1: Shards optimizer states, providing memory optimization while maintaining speed consistent with DDP (Distributed Data Parallel).
* DeepSpeed ZeRO Stage 2: Shards optimizer states and gradients, providing more significant memory optimization while maintaining speed consistent with DDP.
* DeepSpeed ZeRO Stage 2 Offload: Offloads optimizer states and gradients to CPU. Increases distributed communication and GPU-CPU data transfer overhead, but provides substantial memory savings.
* DeepSpeed ZeRO Stage 3: Shards optimizer states, gradients, and model parameters (optionally including activations). Increases distributed communication but provides stronger memory optimization.
* DeepSpeed ZeRO Stage 3 Offload: Offloads optimizer states, gradients, and model parameters (optionally including activations) entirely to CPU. Significantly increases distributed communication and GPU-CPU data transfer overhead, but achieves more extreme memory savings.

## DeepSpeed ZeRO Stage 3

DeepSpeed ZeRO Stage 3 is a training mode with lower VRAM usage in multi-GPU training, but requires modifying some configuration files. We provide examples for some models, primarily by specifying the `deepspeed` configuration via `--config_file`.

Please note that the `deepspeed_zero3_offload` mode is incompatible with PyTorch's native gradient checkpointing mechanism. To address this, we have adapted the `checkpointing` interface of `deepspeed`. Users need to fill the `activation_checkpointing` field in the `deepspeed` configuration to enable gradient checkpointing.

Below is the script for low VRAM model training for the Qwen-Image model, with two-stage split training also enabled:

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

The configurations for `accelerate` and `deepspeed` are as follows:

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