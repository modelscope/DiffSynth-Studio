# The Scheduler in Prior-Based Step Skip

This document explains **what the scheduler is**, **why it matters for the prior**, and **how to use and modify it** correctly.

---

## What Is the Scheduler?

In diffusion models, generation is an iterative denoising process. The **scheduler** defines:

1. **The trajectory** — which points in “noise space” we visit (timesteps and sigmas)
2. **The step rule** — how to update the latent given the model’s prediction

The model predicts a direction (velocity); the scheduler decides how far to move along that direction at each step.

### Timesteps and Sigmas

- **Timestep** `t`: A scalar (often 0–1000) that tells the model “how noisy” the current latent is. The model is conditioned on `t`.
- **Sigma** `σ`: A noise level used in the flow-matching update. For flow matching, `σ ≈ t / 1000` (normalized timestep).

The scheduler produces two arrays of length `num_inference_steps`:

```
timesteps = [t₀, t₁, t₂, …, t_{T-1}]   # High → low (noisy → clean)
sigmas    = [σ₀, σ₁, σ₂, …, σ_{T-1}]   # High → low
```

These are **deterministic** given the scheduler parameters. Different parameters → different trajectory → different results.

### The Step Formula (Flow Matching)

At each step `i`, the scheduler computes the next latent:

```
latent_{i+1} = latent_i + model_output × (σ_{i+1} − σ_i)
```

So the model’s output is scaled by the **sigma difference** between the current and next step. The scheduler’s `sigmas` array is what makes this math correct.

---

## Pseudocode: What the Scheduler Does

```
# ═══════════════════════════════════════════════════════════════════════════
# SCHEDULER: Definition and Role
# ═══════════════════════════════════════════════════════════════════════════

# 1. SET_TIMESTEPS: Build the denoising trajectory from parameters
#    Inputs: num_inference_steps, denoising_strength, sigma_shift (or shift)
#    Outputs: timesteps[], sigmas[] — the exact sequence for this run
#
function SCHEDULER_SET_TIMESTEPS(num_steps, denoising_strength, sigma_shift):
    sigma_start ← sigma_min + (sigma_max - sigma_min) × denoising_strength
    sigmas ← linspace(sigma_start, sigma_min, num_steps)
    sigmas ← sigma_shift × sigmas / (1 + (sigma_shift - 1) × sigmas)   # Rescale
    timesteps ← sigmas × 1000   # Map to 0–1000 range for model conditioning
    return (sigmas, timesteps)


# 2. STEP: Update latent using model output and sigma difference
#    The model predicts a "velocity"; we move the sample by (σ_next − σ_curr)
#
function SCHEDULER_STEP(model_output, timestep, sample, sigmas, timesteps):
    step_id ← index of timestep in timesteps
    σ      ← sigmas[step_id]
    σ_next ← sigmas[step_id + 1]   # or 0 if last step
    sample_next ← sample + model_output × (σ_next − σ)
    return sample_next


# 3. Standard diffusion loop (no prior)
#
function DENOISE_STANDARD(prompt, image, num_steps):
    (sigmas, timesteps) ← SCHEDULER_SET_TIMESTEPS(num_steps, 1.0, 5.0)
    latents ← sample_noise()

    for i in 0 .. num_steps - 1:
        t ← timesteps[i]
        noise_pred ← model(latents, t, prompt, image)
        latents ← SCHEDULER_STEP(noise_pred, t, latents, sigmas, timesteps)

    return decode(latents)
```

---

## Why the Scheduler Matters for the Prior

The prior latent was produced at a **specific point** on a **specific trajectory**. That trajectory is fully defined by `(timesteps, sigmas)`.

If inference uses a **different** trajectory (e.g. different `num_inference_steps`, `denoising_strength`, or `sigma_shift`):

- The sigma differences `(σ_{i+1} − σ_i)` change
- The step formula produces wrong updates
- The denoising path no longer matches what the model expects

So: **prior and inference must use the same scheduler trajectory**. We achieve this by saving and restoring `timesteps` and `sigmas` from the prior run.

---

## What We Did With the Scheduler (Prior-Based Step Skip)

### 1. Save the trajectory when generating the prior

When we run full inference to build the prior, we save not only the latents but also the scheduler’s `timesteps` and `sigmas`:

```python
# prior_utils.py — save_run_metadata()
timesteps = pipe.scheduler.timesteps.cpu().tolist()
sigmas    = pipe.scheduler.sigmas.cpu().tolist()

metadata = {
    "scheduler_timesteps": timesteps,
    "scheduler_sigmas": sigmas,
    "num_inference_steps": len(timesteps),
    "denoising_strength": denoising_strength,
    "sigma_shift": sigma_shift,
    # ...
}
```

### 2. Override the scheduler when inferring from the prior

When resuming from a prior, we **replace** the scheduler’s arrays with the saved ones instead of recomputing them:

```python
# wan_video.py — pipeline __call__
self.scheduler.set_timesteps(num_inference_steps, denoising_strength=..., shift=sigma_shift)

# Prior-based step skip: use the exact trajectory from the prior run
if prior_latents is not None and prior_timesteps is not None and start_from_step is not None:
    self.scheduler.timesteps = prior_timesteps.to(...)
    if prior_sigmas is not None:
        self.scheduler.sigmas = prior_sigmas.to(...)
```

### 3. Skip early steps in the loop

We still iterate over the full `timesteps` array, but we skip the steps we’ve already “done” via the prior:

```python
start_idx = start_from_step + 1
for progress_id, timestep in enumerate(timesteps):
    if progress_id < start_idx:
        continue   # Skip — we loaded the latent after step start_from_step
    # ... run model, scheduler.step(), etc.
```

---

## Pseudocode: Prior Flow With Scheduler Handling

```
# ═══════════════════════════════════════════════════════════════════════════
# PRIOR GENERATION: Save latents AND scheduler state
# ═══════════════════════════════════════════════════════════════════════════

function GENERATE_PRIOR(prompt, image, num_steps, output_dir):
    # Build trajectory (deterministic from params)
    (sigmas, timesteps) ← SCHEDULER_SET_TIMESTEPS(num_steps, 1.0, 5.0)
    latents ← sample_noise()

    for i in 0 .. num_steps - 1:
        t ← timesteps[i]
        noise_pred ← model(latents, t, prompt, image)
        latents ← SCHEDULER_STEP(noise_pred, t, latents, sigmas, timesteps)
        save(latents, output_dir / f"step_{i}.pt")

    # CRITICAL: Save the trajectory so inference can reuse it exactly
    save_metadata({
        "scheduler_timesteps": timesteps,
        "scheduler_sigmas": sigmas,
        "num_inference_steps": num_steps,
        "denoising_strength": 1.0,
        "sigma_shift": 5.0,
    }, output_dir)
    return decode(latents)


# ═══════════════════════════════════════════════════════════════════════════
# PRIOR INFERENCE: Load prior, override scheduler, run remaining steps
# ═══════════════════════════════════════════════════════════════════════════

function INFER_FROM_PRIOR(prompt, image, prior_dir, start_step):
    # Load prior latent (output of step start_step)
    prior_latents ← load(prior_dir / f"step_{start_step}.pt")
    metadata ← load_metadata(prior_dir)

    # Use the EXACT trajectory from the prior run — do NOT recompute
    timesteps ← metadata.scheduler_timesteps
    sigmas    ← metadata.scheduler_sigmas

    latents ← prior_latents
    start_idx ← start_step + 1

    for i in 0 .. len(timesteps) - 1:
        if i < start_idx:
            continue   # Skip steps 0..start_step (already in prior)
        t ← timesteps[i]
        noise_pred ← model(latents, t, prompt, image)
        latents ← SCHEDULER_STEP(noise_pred, t, latents, sigmas, timesteps)

    return decode(latents)
```

---

## How to Use and Modify the Scheduler

### Using the prior correctly

| Requirement | Reason |
|-------------|--------|
| Same `num_inference_steps` | Same trajectory length |
| Same `denoising_strength` | Same starting sigma |
| Same `sigma_shift` | Same sigma rescaling |
| Use saved `timesteps` and `sigmas` | Exact trajectory match; avoids float drift |

The scripts validate these via `validate_scheduler_match()` before inference.

### Modifying scheduler parameters

- **When generating the prior**: Choose `num_inference_steps`, `denoising_strength`, `sigma_shift` as needed. These are saved in metadata.
- **When inferring from the prior**: Pass the **same** values. The pipeline loads `prior_timesteps` and `prior_sigmas` from metadata and overrides the scheduler; the parameters are mainly for validation.

### Example: Changing the number of steps

If you want 20 steps instead of 10:

1. Generate a new prior with `--num_inference_steps 20`.
2. Use that prior with `infer_from_prior.py`; it will read `num_inference_steps: 20` from metadata.
3. You can use e.g. `--start_step 12` to skip the first 13 steps and run 7 steps.

You cannot mix a prior generated with 10 steps with inference configured for 20 steps — the trajectories differ.

### Code: Saving and loading scheduler state

**Saving** (in `prior_utils.save_run_metadata`):

```python
timesteps = pipe.scheduler.timesteps.cpu().tolist()
sigmas = pipe.scheduler.sigmas.cpu().tolist()
metadata = {
    "scheduler_timesteps": timesteps,
    "scheduler_sigmas": sigmas,
    "num_inference_steps": len(timesteps),
    "denoising_strength": denoising_strength,
    "sigma_shift": sigma_shift,
    # ...
}
```

**Loading** (in `infer_from_prior.py`):

```python
meta = load_prior_metadata(prior_dir)
prior_timesteps = torch.tensor(meta["scheduler_timesteps"], dtype=torch.float32)
prior_sigmas = torch.tensor(meta["scheduler_sigmas"], dtype=torch.float32)

# Passed to pipeline; pipeline overrides scheduler.timesteps and scheduler.sigmas
video = pipe(
    ...,
    prior_latents=prior_latents,
    prior_timesteps=prior_timesteps,
    prior_sigmas=prior_sigmas,
    start_from_step=args.start_step,
)
```

**Override in pipeline** (in `wan_video.py`):

```python
self.scheduler.set_timesteps(num_inference_steps, denoising_strength=..., shift=sigma_shift)

if prior_latents is not None and prior_timesteps is not None and start_from_step is not None:
    self.scheduler.timesteps = prior_timesteps.to(self.scheduler.timesteps.device)
    if prior_sigmas is not None:
        self.scheduler.sigmas = prior_sigmas.to(self.scheduler.sigmas.device)
```

### Example: Wan scheduler formula

The Wan scheduler (flow matching) uses:

```python
# diffsynth/diffusion/flow_match.py — set_timesteps_wan
sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
timesteps = sigmas * num_train_timesteps
```

- `denoising_strength`: 0–1; 1 = full denoising from max noise
- `shift` (sigma_shift): Rescales sigmas; default 5 for Wan

---

## Summary

| Concept | Role |
|---------|------|
| **Scheduler** | Defines the denoising trajectory (timesteps, sigmas) and the step update rule |
| **Timesteps** | Conditioning for the model; index into the trajectory |
| **Sigmas** | Used in the step formula; must match between prior and inference |
| **Prior + scheduler** | Prior latent lies on a specific trajectory; inference must use that same trajectory |
| **Override, don’t recompute** | Load saved `timesteps` and `sigmas` to guarantee consistency |
