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
## Dataset format

DiffSynth-Studio does not read one-.txt-per-image folders directly. A dataset is a
folder of images plus a metadata file (csv / json / jsonl) with an `image` column
and a `prompt` column:

```
my_data/
+-- metadata.csv
+-- 0001.jpg
+-- 0002.png
```

```
image,prompt
0001.jpg,"a cute dog, sitting"
0002.png,a cat on a wall
```

If your data is in the common "same-name .txt next to each image" layout, convert it
first (also handles missing captions and sub-folders):

```
python examples/qwen_image_21/model_training/special/low_vram_training/prepare_dataset_from_txt.py --image_dir my_data
```

Then point the training script at it:

```
DATA_DIR=my_data DS_SUBDIR=. bash examples/qwen_image_21/model_training/special/low_vram_training/Qwen-Image-2.1-16GB.sh
```

(or edit `DS=` in the script: `DS="/qwen_image_21/Qwen-Image-2.1"` is only
the example-dataset default.)
## T2I speed optimizations in this branch

Training is text-to-image only, so two per-step costs are constant across steps and
are now computed once per (resolution, prefix length) and cached on the DiT:

1. **RoPE frequencies** (QwenImage21Rope.forward) rebuild index lists with Python
   loops every step.
2. **Attention routing**: without flex-attention (V100/T4) the processor splits the
   text prefix into per-image segments and issues one SDPA call per segment plus one
   for the target tokens. For a pure text prefix the segment masks are exactly
   "causal inside the prefix, full attention from target to everything", i.e. one
   dense block-causal mask, so a **single SDPA call** is mathematically identical
   (verified to 1.5e-8 against both the splited route and the flex BlockMask route).

The dense route activates automatically when the prefix contains no image tokens
(T2I). Set `DIFFSYNTH_QWEN21_T2I_FAST_ATTN=0` to fall back to the splited route.
Measured CPU-side per-step overhead removed: ~1.7 ms at 512x512, ~3.1 ms at
1024x1024; on the GPU side the win is the collapsed kernel-launch chain (N+1 SDPA
calls -> 1), which matters most on V100/T4 at batch size 1.
### Attention kernel choice on V100 / T4

flash-attn 2 does **not** support these cards: official releases only ship sm_80/90
kernels (sm_70/75 were removed after v2.4.x, and v2.5+ dropped sm_75 entirely), and
current main targets sm_80+. Do not waste time compiling it from source on V100/T4.

Instead, install xformers to get a memory-efficient fused kernel that does support
sm_70/75:

```
uv pip install --python  xformers --index-url https://mirrors.aliyun.com/pytorch-wheels/cu121/
```

Caveat: the Qwen-Image-2.1 DiT passes an explicit attention mask (block-causal), and
`attention_forward` only uses the xformers/FA routes when no mask is given, so with
the mask the fused dense route still ends at SDPA. xformers mainly helps if you later
switch to mask-free attention or other pipelines. The single-call dense mask route
added above remains the effective optimization for this model.