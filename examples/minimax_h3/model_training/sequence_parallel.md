# Experimental MiniMax-H3 training sequence parallelism

`train.py --sequence_parallel_size N` shards the main H3 DiT token sequence
across all `N` training ranks. The default, `1`, retains the existing path.
This targets the DiT activation cost of long Ref2VA sequences, including
multiple reference images. It does not change the model's attention context
or remove reference tokens.

## Computation and gradient convention

Each rank processes the same sample, timestep and noise. Main DiT blocks and
the output head operate on local token slices. Ulysses all-to-all exchanges
local tokens/all heads for all tokens/local heads, runs global attention,
then reverses that exchange. Original packed segment boundaries, position
IDs, modality/time indices and output masks are retained. Additional padding
needed to divide the sequence across ranks does not attend to real tokens.

Input embedding and the text token refiner remain replicated. The small
video/audio output predictions are gathered, so each rank computes the same
original flow-matching loss. The gather backward multiplies local gradients
by `N`, compensating for DDP/ZeRO's later parameter-gradient average. The
split backward gathers and divides by `N` before the replicated embedding.
Do not additionally divide the loss by the SP size or use a gradient-summing
DDP communication hook. Ordinary gradient accumulation still applies.

The training launcher keeps a complete epoch on each rank instead of
sharding the dataset as in data parallelism. Rank zero's CPU batch and a new
Torch RNG seed are broadcast before preprocessing and loss computation.
The learning-rate scheduler advances once per optimizer update. With the
launcher's batch size of one, the unique-sample batch size is the gradient
accumulation count, independent of `N`.

## Scope

- One SP group containing the entire distributed world; no SP+DP hybrid.
- `N` must equal the launched process count and divide the attention head
  count. Standard H3 has 56 heads, allowing 2, 4 or 8 ranks, for example.
- Standard H3 attention, `sft` and `sft:train`; full tuning and LoRA use the
  same path. Do not enable SP for `sft:data_process`.
- ControlNet, VDN/custom attention, FSDP and `--enable_model_cpu_offload`
  are not supported by this implementation. Do not combine the flag with
  Accelerate's separate `parallelism_config`.
- There is no sharding of VAE, Qwen/text encoding or the text token refiner.
  Those stages can still run out of memory. Cached conditioning is preferable
  to repeatedly running and broadcasting raw-media preprocessing. All ranks
  currently load a batch before the leader broadcast, so input I/O is replicated.
- The model weights are not sharded by this feature. DDP still requires a full
  copy on each device; ZeRO-3 is a separate mechanism for sharding parameters.
- CPU/Gloo and tiny-model CUDA/NCCL, BF16 and ZeRO-3 numerical/training-entry
  checks passed locally, including full tuning and LoRA. Quantization and
  real-checkpoint memory/performance remain unvalidated. In particular, ZeRO's configured global
  batch size still counts ranks, not unique samples in this SP group.

## Small regression, without weights or GPUs

Install the normal repository dependencies plus its audio dependencies needed
to import the H3 pipeline. From the repository root:

```bash
CUDA_VISIBLE_DEVICES="" DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch \
python -B -m torch.distributed.run --standalone --nproc_per_node=2 \
  examples/dev_tools/test_minimax_h3_sequence_parallel.py \
  --device cpu --report /path/to/output/h3_sp_cpu2.json
```

Repeat with `--nproc_per_node=4`. This uses the actual H3 model with tiny
dimensions, actual collectives/DDP, PEFT LoRA, the Ref2VA packing function,
flow-matching/CFG loss and the Accelerate training launcher. It compares
outputs, losses, all parameter gradients and optimizer updates to an
unsharded reference. It also checks non-divisible sequence lengths,
conditioning masks, checkpointing, reference-image counts, reference audio,
zero audio-loss weight, input/RNG synchronization, gradient accumulation,
the incomplete final accumulation group and scheduler step counts.
Unsupported configurations are checked before training, including a mismatched
process count, custom attention, preprocessing mode and a micro-batch size other
than one.

For a later CUDA communication check, expose two free GPUs and use
`--device cuda` with the same script. It selects NCCL and still uses tiny
FP32 weights. Passing it is not a real-model BF16 or memory benchmark.

Additional GPU checks (normal training dependencies, including DeepSpeed,
are required for the ZeRO-3 cases):

```bash
python -B -m torch.distributed.run --standalone --nproc_per_node=2 \
  examples/dev_tools/validate_minimax_h3_sp_cuda.py \
  --engine zero3 --precision bfloat16 --report /path/to/output/zero3_bf16.json
```

Use `--engine ddp` or `--precision float32` for the other numerical cases.
This checks gradients and optimizer updates against an unsharded reference,
then exercises the training launcher. Tolerances are embedded in the report.
The script also supports `--mode benchmark --precision bfloat16 --references 8`;
run once without `--sp` and once with it, in separate processes on the same
GPUs. Add `--checkpoint` to both arms to measure that configuration. It uses a
reduced-size, random H3 model with synthetic conditioning, two warmup steps
and three measured steps. Numerical checks cap PyTorch allocation at 4 GiB
per rank; benchmarks cap it at 12 GiB. Non-PyTorch allocations are outside
those caps. Sharing GPUs can affect timings.

## Saving and restarting from adapter weights

SP adds no checkpoint tensors or prefixes. The existing `ModelLogger` gathers
and exports the trainable weights. With ZeRO-3, enable
`zero3_save_16bit_model: true` in the Accelerate configuration (already set in
`full/accelerate_config_zero3.yaml`), or
`stage3_gather_16bit_weights_on_model_save: true` in a DeepSpeed JSON config.
All ranks must participate in saving, even though only rank zero writes a file.

For LoRA trained with `--remove_prefix_in_ckpt 'pipe.dit.'`, restart with the
same base weights, adapter targets/rank and `--lora_checkpoint /path/to/epoch-N.safetensors`.
This restores **adapter weights only**; it starts a fresh optimizer/scheduler
and does not restore the data position or RNG state. SP may be enabled or
disabled for loading. This feature does not add full-job checkpoint/resume.

The real logger and adapter-loading method can be checked with tiny weights:

```bash
CUDA_VISIBLE_DEVICES="" DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch \
python -B -m torch.distributed.run --standalone --nproc_per_node=2 \
  examples/dev_tools/test_minimax_h3_sp_checkpoint.py \
  --device cpu --output /path/to/output/h3-sp-checkpoint-cpu
```

For CUDA, expose two GPUs, replace `--device cpu` with
`--device cuda --engine zero3 --precision bfloat16` and use a separate output
directory. `--engine ddp` also works. The check saves two epochs, verifies
every reloaded weight exactly with SP off/on, then compares loss, gradients
and one fresh-optimizer update. It does not download pretrained weights.

## Real-checkpoint trial, when hardware is available

First produce Ref2VA conditioning with the existing
`lora/MiniMax-H3-Ref2VA.sh` data-processing stage, with SP disabled. Keep
weights, conditioning, runtime caches and outputs on the data disk.

For the training stage, add the SP flag and use exactly that many processes.
For example, the following is an **unvalidated ZeRO-3 trial template** using
an existing, separately validated ZeRO-3 configuration and local weights:

```bash
accelerate launch --config_file /path/to/zero3.yaml --num_processes 8 \
  examples/minimax_h3/model_training/train.py \
  --sequence_parallel_size 8 \
  --dataset_base_path /path/to/ref2va-conditioning-cache \
  --model_paths '["/path/to/Ref2VA/transformer/model*.safetensors"]' \
  --processor_path /path/to/Ref2VA/processor \
  --num_frames 124 --dataset_repeat 1 --num_epochs 1 \
  --learning_rate 1e-4 --gradient_accumulation_steps 1 \
  --lora_base_model dit \
  --lora_target_modules 'attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2' \
  --lora_rank 32 --use_gradient_checkpointing --find_unused_parameters \
  --remove_prefix_in_ckpt 'pipe.dit.' \
  --output_path /path/to/output/h3-sp-trial --task sft:train
```

This feature branch does not include PR #1797's preset-LoRA fusion fix.
If using a preset adapter with ZeRO-3, use a checkpoint merged offline or
validate that fix separately; changing SP does not repair adapter fusion.

Before treating this as production support, compare SP off/on with identical
weights, cached input, reference counts, precision, checkpointing and unique
sample exposure. Check loss/gradients and one optimizer update first, then
measure peak allocated/reserved memory on every rank and optimizer-step time
after warmup. Include both a short reference input and the long input that
previously OOMed. A local BF16 synthetic test on two RTX 3090s (6 blocks,
hidden size 512, LoRA rank 8, 8768 packed tokens) reduced maximum per-rank
PyTorch peak allocation from 1589.9 MiB to 870.4 MiB; with checkpointing,
433.4 MiB to 269.3 MiB. This is not a pretrained-H3 memory result or evidence
that the original OOM is resolved. No production throughput claim is made.
