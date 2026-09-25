# Qwen-Image-2.1 LoRA training on 16 GB GPUs (V100 / T4)

Target hardware: Baidu AI Studio V100-16GB, Google Colab T4 (16 GB).
Both lack BF16 tensor cores, so this recipe runs everything in FP16 and keeps the
peak GPU footprint under ~14 GB by combining three existing framework features:

| technique | what it saves | framework feature |
|---|---|---|
| split training (`sft:data_process` -> `sft:train`) | text encoder + VAE are computed once and cached to disk; stage 2 never loads them | `--task` |
| INT4 NF4 weights (bitsandbytes) | DiT 13.3 GB (BF16) -> ~3.6 GB | `--quant_options ...:bitsandbytes_nf4` |
| FP16 instead of BF16 | V100/T4 have no BF16 tensor cores | `--force_fp16` (added in this branch) |
| 8-bit AdamW | optimizer state 4x smaller | `--customized_optimizer bitsandbytes.optim.AdamW8bit` |
| gradient checkpointing + accumulation | activations | `--use_gradient_checkpointing`, `--gradient_accumulation_steps` |

The LoRA is trained on top of the frozen INT4 DiT and saved as a standalone
adapter (`epoch-*.safetensors`); it is never merged into the base weights, so it
can be hot-loaded at inference time with `pipe.load_lora(pipe.dit, ...)`.

## Quick start

```bash
# Baidu AI Studio (V100)
bash examples/qwen_image_21/model_training/special/low_vram_training/setup_baidu_studio.sh
# Google Colab (T4)
bash examples/qwen_image_21/model_training/special/low_vram_training/setup_colab_t4.sh

# then, on either machine:
bash examples/qwen_image_21/model_training/special/low_vram_training/Qwen-Image-2.1-16GB.sh
```

Environment knobs: `DATA_DIR`, `CACHE_DIR`, `OUT_DIR`, `MAX_PIXELS`
(default 262144 = 512x512; use 1048576 for 1024x1024 on V100), `LORA_RANK`
(default 32), `EPOCHS`, `GRAD_ACCUM`, `SKIP_STAGE1=1` to reuse a cache.

## Notes

* `DIFFSYNTH_DISABLE_FLEX_ATTN=1` is exported by the script: the flex-attention
  route is `torch.compile`d with Triton, which is unreliable on sm_70 (V100).
  With the flag the DiT falls back to the materialized-mask SDPA route.
* Stage 1 quantizes the text encoder to NF4 as well (it only runs forward
  passes there), which keeps the 8 GB Qwen3-VL encoder inside 16 GB together
  with the VAE. Stage 2 does not load the text encoder at all.
* Disk budget: ~31 GB of model weights + the stage-1 cache (a few hundred MB
  for the example dataset). Baidu AI Studio: keep everything under
  `/home/aistudio/work` so it survives restarts.
* Expected peak VRAM (512x512, rank 32): ~10-12 GB. At 1024x1024 the T4 may
  still OOM; lower `MAX_PIXELS` or `LORA_RANK` first.