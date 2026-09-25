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

Environment knobs: `DATA_DIR`, `DS_SUBDIR`, `CACHE_DIR`, `OUT_DIR`, `MAX_PIXELS`
(default 262144 = 512x512; use 1048576 for 1024x1024 on V100), `LORA_RANK`
(default 32), `EPOCHS`, `GRAD_ACCUM`, `GC_BLOCKS`, `NUM_WORKERS`, `SKIP_STAGE1=1`
to reuse a cache, `FORCE_STAGE1=1` to rebuild it, `DRY_RUN=1` to print the two
`accelerate launch` commands instead of running them.

## Model weights: what downloads what

**The setup scripts download no weights** - they only build the venv, install
torch + deps and fetch this repo. The weights come from **modelscope.cn**
(reachable from CN managed notebooks, no GitHub involved) the first time a stage
needs them, and are cached in `${DIFFSYNTH_MODEL_BASE_PATH:-./models}/Qwen/Qwen-Image-2.1`:

| component | size | fetched by | used by |
|---|---|---|---|
| `text_encoder/` (Qwen3-VL, 4 shards) | 16.3 GB | stage 1 | stage 1 only |
| `vae/` | 1.3 GB | stage 1 | stage 1 only |
| `processor/` (tokenizer) | ~15 MB | stage 1 | stage 1 only |
| `transformer/` (DiT, 2 shards) | 13.3 GB | stage 2 | stage 2 only |

Each stage passes only the models it runs in `--model_id_with_origin_paths`, so
stage 1 never pulls the DiT and stage 2 never pulls the text encoder. Downloads
are resumable and skip files that already exist, so a killed run just continues.

Pull them up front, with a progress bar and a disk check, instead of letting the
first training run stall on a silent download:

```bash
PRELOAD_MODELS=1 bash examples/qwen_image_21/model_training/special/low_vram_training/setup_baidu_studio.sh
# or per stage
PRELOAD_MODELS=1 PRELOAD_STAGE=stage1 bash .../setup_baidu_studio.sh   # 17.6 GB
PRELOAD_MODELS=1 PRELOAD_STAGE=stage2 bash .../setup_baidu_studio.sh   # 13.3 GB
```

Once stage 1 has written its cache, the text encoder is never read again:

```bash
rm -rf /home/aistudio/work/models/Qwen/Qwen-Image-2.1/text_encoder    # 16 GB back
```

(`FREE_TEXT_ENCODER=1` does this automatically after stage 1; `KEEP_TEXT_ENCODER=1`
silences the reminder.)

## Notes

* `DIFFSYNTH_DISABLE_FLEX_ATTN=1` is exported by the script: the flex-attention
  route is `torch.compile`d with Triton, which is unreliable on sm_70 (V100).
  With the flag the DiT falls back to the materialized-mask SDPA route.
* Stage 1 quantizes the text encoder to NF4 as well (it only runs forward
  passes there), which keeps the 16 GB Qwen3-VL encoder inside 16 GB together
  with the VAE. Stage 2 does not load the text encoder at all.
* **Host RAM matters as much as VRAM here.** Online NF4 materializes the FP16
  checkpoint in host memory before packing it (16.3 GB for the text encoder,
  13.3 GB for the DiT). Two escape hatches when the runtime has little RAM
  (a free Colab shows ~13 GB):
  * `TE_DISK_OFFLOAD=1` - stage 1 streams the FP16 text encoder straight from the
    safetensors file instead of quantizing it (`--offload_models`), so it costs
    almost nothing in RAM or VRAM and only pays extra disk reads.
  * `DIFFSYNTH_QUANT_STREAM=1` - keep NF4 but pack it **one transformer block at a
    time, straight from disk**, instead of holding the whole fp checkpoint in RAM.
    The host peak drops from ~16 GB to ~1.5 GB and the resulting weights are
    bit-identical to the default path (verified on a small model: same packed
    tensors, same forward output). Requires a model that declares
    `_no_split_modules` (the DiT and the Qwen3-VL text encoder both do).
  `Qwen-Image-2.1-16GB.sh` turns both on automatically when host RAM < 24 GB.
* **`NO_NF4=1`** is the fallback when bitsandbytes itself cannot run on the GPU:
  stage 1 streams the text encoder from disk and stage 2 trains with
  `--enable_model_cpu_offload` (FP16 weights on the CPU, one layer at a time).
  It is markedly slower, but it needs no INT4 support at all.
* **`transformers>=4.57.0` is required**: the text encoder is `Qwen3VLForConditionalGeneration`,
  which does not exist in older releases. Both setup scripts pin it and verify the
  import after installing, so a stale `transformers` fails in seconds instead of
  after a 17 GB download.
* **Baidu AI Studio (read-only conda env):** the notebook image mounts its conda
  `site-packages` read-only, so `uv`/`pip` cannot patch it (`Permission denied
  (os error 13)` while removing `tokenizers-*.dist-info/INSTALLER`). `setup_baidu_studio.sh`
  therefore builds a dedicated venv at `/home/aistudio/work/venv-diffsynth` and
  installs torch + all training deps there. The notebook kernel itself is left
  untouched (Paddle keeps working). `Qwen-Image-2.1-16GB.sh` activates that venv
  automatically when it exists, so you do not need to restart the kernel; for
  interactive use run `source /home/aistudio/work/venv-diffsynth/bin/activate`.
  A half-finished install left behind in the system env is harmless. The setup
  also writes `/home/aistudio/work/diffsynth_env.sh`, which pins the venv, the
  uv cache and `HF_HOME` / `MODELSCOPE_CACHE` under `/home/aistudio/work` so
  downloaded weights survive notebook restarts; the training script sources it
  automatically.
* Disk budget: ~31 GB of model weights (17.6 GB for stage 1 + 13.3 GB for stage 2)
  + the stage-1 cache (a few hundred MB for the example dataset, but it grows with
  resolution and dataset size - each cached item holds the prompt embeds *and* the
  VAE latents). Baidu AI Studio: keep everything under `/home/aistudio/work` so it
  survives restarts; `setup_baidu_studio.sh` prints free space up front and warns
  below 60 GB.
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

Then point the training script at it - `DS_SUBDIR` is the path under `DATA_DIR`
that holds `metadata.csv`:

```
DATA_DIR=my_data DS_SUBDIR=. bash examples/qwen_image_21/model_training/special/low_vram_training/Qwen-Image-2.1-16GB.sh
```

The default `DS_SUBDIR=qwen_image_21/Qwen-Image-2.1` matches the bundled example
dataset; use `DS_SUBDIR=.` for a folder you prepared yourself.

VAE outputs are cached by stage 1 too, so stage 2 reads neither the images nor the
VAE: each `*.pth` in `CACHE_DIR` holds the prompt embeds, the latents and the
scheduler inputs for one sample. Changing `MAX_PIXELS` or the dataset invalidates
the cache - rebuild it with `FORCE_STAGE1=1`.
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
uv pip install --python "$(command -v python)" xformers --index-url https://mirrors.aliyun.com/pytorch-wheels/cu121/
```

Caveat: the Qwen-Image-2.1 DiT passes an explicit attention mask (block-causal), and
`attention_forward` only uses the xformers/FA routes when no mask is given, so with
the mask the fused dense route still ends at SDPA. xformers mainly helps if you later
switch to mask-free attention or other pipelines. The single-call dense mask route
added above remains the effective optimization for this model.
### More speed knobs (T2I)

* **Selective gradient checkpointing**: `--gradient_checkpointing_blocks N`
  (env `GC_BLOCKS` in the shell script) checkpoints only the first N of the 32
  transformer blocks. The first blocks hold the largest activations, so N=16 keeps
  most of the VRAM saving while the last 16 blocks run without the recompute
  overhead (typically ~10-15%% faster backward). Verified numerically identical to
  full checkpointing (grad diff 0.0). Start from N=16 and lower it until VRAM fits.
* **Fixed resolution**: pass `--height/--width` (or keep one `MAX_PIXELS`) so the
  per-shape cache never misses.
* **DataLoader workers**: `--dataset_num_workers 4` keeps the GPU fed when reading
  the stage-1 cache from disk.
* **torch.compile**: `pipe.compile_pipeline()` exists upstream (regional compile of
  `QwenImage21TransformerBlock`). On V100/T4 it can fuse the modulation/norm chains,
  but expect minutes of compile time and test loss curves before trusting it.
### 1024x1024 on 16 GB

Yes, with the right knobs. At 1024x1024 the joint sequence is ~4400 tokens, so the
dense S x S attention mask would cost ~73 MB per layer; the T2I fast path therefore
decomposes the block-causal mask into **two mask-free SDPA calls** (causal on the
text prefix, unmasked from target tokens to everything), verified identical to the
splited route (loss and gradients match to 0.0). No mask tensor is materialized at
all.

Recommended stage-2 settings for 1024x1024:

```
MAX_PIXELS=1048576 GC_BLOCKS=-1 GRAD_ACCUM=8 \
  bash examples/qwen_image_21/model_training/special/low_vram_training/Qwen-Image-2.1-16GB.sh
```

Approximate peak VRAM (batch 1, full gradient checkpointing): INT4 DiT 3.6 GB +
optimizer/LoRA ~0.5 GB + activations ~7-9 GB = ~12-13 GB. If it OOMs on T4, add
`--use_gradient_checkpointing_offload` (stores activations on CPU; slower but
bounded) or drop to `GC_BLOCKS=-1` + `MAX_PIXELS=786432` (896x896).