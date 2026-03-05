# Prior-Based Diffusion Step Skip

**~70% fewer inference steps, same quality, zero retraining.**

When you have a **fixed identity or scene** and only **one aspect varies** (e.g. motion, lip-sync, lighting), early diffusion steps are largely redundant. This module lets you:

1. **Generate a prior** — Run full inference once, save latents at each step
2. **Infer from prior** — Load a saved latent (e.g. step 6) and run only the remaining 3–4 steps

---

## Concept: What Is a Prior?

In diffusion models, generation is a **multi-step denoising process**. Each step refines the latent representation:

```
Step 0 (noisy) → Step 1 → Step 2 → … → Step T (clean)
```

The key insight: **early steps fix identity and structure** (who/what is in the scene), while **later steps refine the varying aspect** (motion, expression, lighting). When identity is fixed and only one aspect changes, the early trajectory is nearly identical across runs. We can reuse it.

A **prior** is a saved latent at some intermediate step. Instead of starting from pure noise every time, we inject a prior and run only the remaining steps. The prior encodes “what we already know” — the identity — so we only compute “what changes” — the motion or other variable.

### Pseudocode: Standard Diffusion (No Prior)

```
function GENERATE(prompt, image, num_steps):
    latents ← sample_noise(shape)           # Start from random noise
    timesteps ← scheduler.get_timesteps(num_steps)

    for step in 0 .. num_steps - 1:
        t ← timesteps[step]
        noise_pred ← model(latents, t, prompt, image)
        latents ← scheduler.step(latents, noise_pred, t)
        # ... (optionally save latents here for prior generation)

    return decode(latents)
```

### Pseudocode: Prior-Based Step Skip

```
function GENERATE_PRIOR(prompt, image, num_steps, output_dir):
    # One-time: run full inference and save latents at each step
    latents ← sample_noise(shape)
    timesteps ← scheduler.get_timesteps(num_steps)

    for step in 0 .. num_steps - 1:
        t ← timesteps[step]
        noise_pred ← model(latents, t, prompt, image)
        latents ← scheduler.step(latents, noise_pred, t)
        save(latents, output_dir / f"step_{step}.pt")   # ← Prior checkpoint

    save_metadata(timesteps, scheduler_params, output_dir)
    return decode(latents)


function INFER_FROM_PRIOR(prompt, image, prior_dir, start_step):
    # Accelerated: load prior, run only remaining steps
    prior_latents ← load(prior_dir / f"step_{start_step}.pt")
    metadata ← load_metadata(prior_dir)
    timesteps ← metadata.timesteps
    num_steps ← len(timesteps)

    latents ← prior_latents
    # Skip steps 0 .. start_step; begin at start_step + 1
    for step in (start_step + 1) .. num_steps - 1:
        t ← timesteps[step]
        noise_pred ← model(latents, t, prompt, image)   # New prompt can differ!
        latents ← scheduler.step(latents, noise_pred, t)

    return decode(latents)
```

### Why It Works

| Phase              | Steps (e.g. 10-step run) | What happens                          |
|--------------------|---------------------------|---------------------------------------|
| Identity formation | 0–5                       | Geometry, lighting, scene layout      |
| **Inflection**     | **6**                     | Identity fixed; motion not committed  |
| Refinement         | 7–9                       | Temporal details, sharpness          |

By injecting the prior at step 6, we skip redundant identity formation. The remaining steps refine the **varying aspect** (e.g. motion) driven by the **new prompt**. Same identity, different motion — with ~70% fewer steps.

### Constraints

- **Same scheduler**: Prior and inference must use identical `num_inference_steps`, `denoising_strength`, `sigma_shift`.
- **Same conditioning (identity)**: Same input image (I2V) or seed-dependent structure.
- **Varying aspect**: Prompt (or other conditioning) can change for the refinement phase.

---

## Quick Start

Scripts work from **repo root** or from this directory. Run from repo root for consistent paths.

### Step 1: Generate the prior

**From repo root:**

```bash
# Download example image and run full inference
python examples/wanvideo/prior_based_step_skip/generate_prior.py \
    --download_example \
    --output_dir ./prior_output \
    --num_inference_steps 10
```

**Or with your own image:**

```bash
python examples/wanvideo/prior_based_step_skip/generate_prior.py \
    --image path/to/image.jpg \
    --output_dir ./prior_output \
    --num_inference_steps 10
```

**From this directory:**

```bash
cd examples/wanvideo/prior_based_step_skip

# With --download_example (downloads to repo root data/)
python generate_prior.py --download_example --output_dir ./prior_output --num_inference_steps 10

# Or with your own image
python generate_prior.py --image path/to/image.jpg --output_dir ./prior_output --num_inference_steps 10
```

Output: `./prior_output/run_<id>/` with `step_0000.pt` … `step_0009.pt`, `run_metadata.json`, and `output_full.mp4`.

### Step 2: Run accelerated inference

```bash
# From repo root (replace run_<id> with actual run ID from step 1)
python examples/wanvideo/prior_based_step_skip/infer_from_prior.py \
    --prior_dir ./prior_output/run_<id> \
    --start_step 6 \
    --image data/examples/wan/input_image.jpg \
    --prompt "Different motion: the boat turns sharply to the left."
```

Or from this directory:

```bash
python infer_from_prior.py \
    --prior_dir ./prior_output/run_<id> \
    --start_step 6 \
    --image data/examples/wan/input_image.jpg \
    --prompt "Different motion: the boat turns sharply to the left."
```

This runs only 3 steps (7, 8, 9) instead of 10 — ~70% fewer steps.

## How It Works

| Steps   | Content                                      |
|---------|-----------------------------------------------|
| 1–5     | Identity formation (geometry, lighting)      |
| **6**   | **Inflection point** — identity formed, motion not yet committed |
| 7–10    | Temporal refinement (details, sharpness)      |

By injecting the latent at step 6, we skip redundant identity formation. The remaining steps refine the motion (or other varying aspect) driven by the new prompt.

## Scripts

| Script              | Purpose                                                |
|---------------------|--------------------------------------------------------|
| `generate_prior.py` | Full inference with latent saving at each step         |
| `infer_from_prior.py` | Accelerated inference from a saved prior             |
| `prior_utils.py`    | Latent save/load, metadata, scheduler validation       |

## Options

### generate_prior.py

- `--image` — Input image (required unless `--download_example`)
- `--download_example` — Download example image from ModelScope (saves to `data/examples/wan/`)
- `--output_dir` — Where to save latents (default: `./prior_output`)
- `--num_inference_steps` — Total steps (default: 10)
- `--start_step` — Not used here; for reference when calling infer_from_prior
- `--save_decoded_videos` — Decode and save video at each step (for finding formation point)

### infer_from_prior.py

- `--prior_dir` — Path to prior run (e.g. `./prior_output/run_123`)
- `--start_step` — Step to resume from (default: 6)
- `--image` — Same image used for prior generation
- `--prompt` — New prompt for the varying aspect

## Scheduler Identity

The scheduler used during prior generation **must match** inference. The scripts save and validate:

- `num_inference_steps`
- `denoising_strength`
- `sigma_shift`
- `scheduler_timesteps` and `scheduler_sigmas`

Do not change these between prior generation and inference.

## Requirements

- DiffSynth-Studio installed (`pip install -e .` from repo root)
- GPU with ≥8GB VRAM (low-VRAM config uses disk offload)
- Wan2.1-I2V-14B-480P model (downloaded automatically from ModelScope)

## See Also

- [Scheduler README](SCHEDULER_README.md) — What the scheduler is, its role in the prior, and how to use/modify it
- [Wan model documentation](../../../docs/en/Model_Details/Wan.md)
- [Model inference examples](../model_inference_low_vram/)
