# Audit: DPS Physics Loss and Frame Consistency

Date: 2026-03-23
Repo: `/home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD`
Audited against:
- `~/.cursor/plans/spad_neurips_full_plan_1cbbff23.plan.md`
- `/home/jw/.gemini/antigravity/brain/acbd9b9a-0815-4326-bd30-723934f46de3/dps_consistency_review.md.resolved`
- `/home/jw/.gemini/antigravity/brain/78a66b5e-e1c9-4dbe-adb2-cac119789fe7/spad_audit_report.md.resolved`

## 1. Implementation map

### Physics / DPS

- `diffsynth/diffusion/spad_forward.py`
  - `SPADForwardModel.intensity_to_detection_prob`
  - `SPADForwardModel.log_likelihood`
  - `SPADForwardModel.negative_log_likelihood`
  - `SPADForwardModel.measurement_loss`
  - `SPADMeasurementConsistency.forward`
  - Role: Bernoulli/Binomial forward model and NLL-like loss wrapper.

- `diffsynth/diffusion/flow_dps.py`
  - `compute_dps_correction`
  - `get_guidance_weight`
  - `flux_dps_inference`
  - Sampling hook: computes a correction from decoded `x0_hat = latents - sigma * noise_pred`, then adds that correction to `noise_pred` before `pipe.step(...)`.
  - Operating space: latent update, with loss evaluated after VAE decode in pixel space.

- `diffsynth/diffusion/latent_dps.py`
  - `compute_latent_dps_correction`
  - `get_guidance_weight`
  - Role: lightweight latent-space heuristic replacing pixel-space physics guidance.

- `validate_dps.py`
  - `make_dps_step_fn`
  - Active entry point for DPS experiments.
  - Sampling hook: monkeypatches `pipe.step`; on each step it computes `correction = compute_latent_dps_correction(...)`, sets `noise_pred = noise_pred + correction`, then calls the original scheduler step.
  - Operating space: latent space only.

### Frame consistency

- `paired_spad_dataset.py`
  - `PairedSPADDataset.__getitem__`
  - Role: loads exactly 2 SPAD frames from the same scene, sampled from different frame folders.

- `train_consistency.py`
  - `FluxConsistencyTrainingModule.forward`
  - Role: runs normal pipeline units for F1, then prepares F2 conditioning and passes both into the custom loss.

- `diffsynth/diffusion/consistency_loss.py`
  - `FlowMatchSFTWithConsistencyLoss`
  - Training hook: uses one noisy latent and one timestep shared by F1/F2, computes standard flow-matching loss on F1, then adds `MSE(noise_pred_f1, noise_pred_f2)`.
  - Operating space: DiT velocity prediction / latent-space prediction consistency, not measurement-space likelihood.

- `diffsynth/pipelines/flux_image.py`
  - `FluxImageUnit_ControlNet.process`
  - `model_fn_flux_image`
  - Needed for interpretation: `controlnet_conditionings` are VAE-encoded conditioning latents, not precomputed ControlNet residual stacks.

## 2. Core verdict

### A. SPAD forward model / likelihood

- `FAIL`: `H(x)` is not implemented as `softplus(alpha * L(x) + beta)`.
  - Current code uses `p = 1 - exp(-alpha * intensity)` directly in `spad_forward.py:40-47`.
  - There is no `beta` offset and no guaranteed strictly positive exposure variable `H`.

- `FAIL`: stable `log1mexp` is not implemented.
  - Current code uses `torch.log(p)` and `torch.log(1.0 - p)` in `spad_forward.py:72-79`.
  - Spec-required stable form `log(-expm1(-H))` is absent.
  - Current `p.clamp(eps, 1-eps)` avoids `log(0)` but biases the likelihood and creates dead gradients when the unclamped `p` falls below `eps`.

- `FAIL`: no explicit sRGB-to-linear conversion before likelihood evaluation.
  - Both `spad_forward.py:137` and `flow_dps.py:108` use `decoded.mean(dim=1)` on decoded RGB directly.
  - This applies the SPAD measurement model to gamma-compressed values rather than linear intensity.

- `FAIL`: no explicit `{0,255} -> {0,1}` measurement normalization inside the physics likelihood path.
  - `flow_dps.py` assumes `spad_measurement` is already in `[0,1]`.
  - `flux_dps_inference` has no loader or normalization helper for that tensor.
  - This is currently masked by the fact that `flow_dps.py` is not wired into the active validation script.

- `PARTIAL`: multi-frame Binomial form exists structurally via `num_frames`, but only if the caller passes normalized counts.
  - The implementation computes `y = measurement * num_frames`.
  - That matches the intended Binomial form only when `measurement` is a fraction, not raw integer counts.

### B. DPS / PaDIS-style sampling integration

- `FAIL`: the active inference path is not Bernoulli physics guidance.
  - `validate_dps.py` uses `latent_dps.py`, not `flow_dps.py`.
  - The implemented loss is `||x0_hat - z_spad||^2` in latent space, not the SPAD Bernoulli NLL.

- `FAIL`: both DPS implementations appear to use the correction with the wrong sign.
  - In both `flow_dps.py:132` and `latent_dps.py:69`, the code builds `correction = -guidance_scale * grad` and then adds it to `noise_pred`.
  - With this scheduler, `x_{next} = x_t + v * (sigma_next - sigma)` and `sigma_next - sigma < 0`.
  - Therefore adding `-eta * grad` to `v` moves the next sample in the `+grad` direction, not the `-grad` direction required to decrease NLL / L2 loss.
  - Minimal fix: flip the sign of the injected correction, not the scheduler.

- `FAIL`: pixel-space FlowDPS defaults to `NLL + L2`, not pure likelihood guidance.
  - `FlowDPSConfig` defaults `use_l2_loss=True` and `use_nll_loss=True`.
  - `SPADForwardModel.measurement_loss` also hardcodes `nll + l2`.
  - The spec asked for a small correction that increases `log p(y|x)`; the L2 term is an extra heuristic and should be opt-in, not default.

- `FAIL`: no PaDIS-style preconditioning / normalization.
  - `flow_dps.py` only clamps gradient values elementwise.
  - There is no `mean(|grad|)` normalization and no `sqrt(nll / K)` normalization.

- `FAIL`: no ramp-up schedule.
  - Both `flow_dps.py` and `latent_dps.py` implement only `constant`, `linear_decay`, and `cosine`.
  - All current schedules are flat or decreasing, opposite the requested “small early, larger later” guidance schedule.

- `FAIL`: weak-signal caution is violated in the active latent DPS path.
  - The active method is a direct latent pull toward `z_spad`.
  - That is closer to a heuristic projection than a weak Bernoulli consistency correction.

### C. Frame consistency

- `PASS`: it uses exactly 2 frames per training sample.
  - `paired_spad_dataset.py:122-124` samples `f1` and `f2`.

- `PASS`: it samples across multiple possible pairs over training.
  - Available frame folders are collected per scene.
  - Each `__getitem__` samples a fresh pair from the available set.
  - It does not use all pairs simultaneously; it uses one random pair per sample draw.

- `PASS`: frames are from the same scene.
  - The scene id is extracted from the metadata row and reused to construct every candidate frame path.

- `PASS`: resize/crop is spatially aligned.
  - `ImageCropAndResize` is deterministic center-crop, not random crop.
  - GT, F1, and F2 remain aligned after preprocessing.

- `PASS`: the loss shares the same noisy latent and timestep across F1/F2.
  - `consistency_loss.py:38-49` samples one timestep and one `noisy_latents`, then reuses both for F1 and F2.

- `PASS`: the F2 conditioning path is actually consistent with the pipeline.
  - Earlier audits flagged `train_consistency.py:68-73` as “bypassing ControlNet.”
  - That is not correct for this codebase.
  - `flux_image.py:473-489` shows that the standard pipeline unit for ControlNet produces VAE-encoded conditioning latents.
  - `model_fn_flux_image` still runs the ControlNet module using those latents.
  - So setting `controlnet_conditionings_f2 = [f2_latent]` matches the expected interface.

- `PARTIAL`: the consistency loss is image/latent prediction consistency only.
  - It is `MSE(noise_pred_f1, noise_pred_f2)`.
  - It is not measurement-space consistency and does not aggregate more than 2 frames in one loss.

- `WARN`: the consistency term is not scaled by the scheduler training weight.
  - `loss_sft` is multiplied by `pipe.scheduler.training_weight(timestep)`.
  - `loss_consistency` is added without the same weighting.
  - This makes the effective `consistency_weight` timestep-dependent.

## 3. What the earlier audits got right and wrong

### Confirmed

- Missing stable `log1mexp`
- Missing `softplus(alpha * L + beta)`
- Missing linear-intensity conversion
- Active `validate_dps.py` path is not Bernoulli physics guidance
- No PaDIS normalization
- No ramp-up schedule
- Current frame-consistency training uses exactly 2 frames per sample draw

### Corrected / new findings

- The earlier “F2 bypasses ControlNet” finding is incorrect in this codebase.
  - The pipeline expects ControlNet conditionings to be VAE latents.
  - F2 is prepared at exactly that interface.

- Additional issue not called out in the earlier audits:
  - Both DPS implementations inject the correction with the wrong sign relative to this scheduler.

- Additional issue not called out in the earlier audits:
  - Pixel-space `flow_dps.py` defaults to `NLL + L2` rather than pure log-likelihood guidance.

- Additional issue not called out in the earlier audits:
  - The consistency loss balance changes across timesteps because only the supervised term is scheduler-weighted.

- Additional non-core issue:
  - Several docs still describe the DPS path as Bernoulli FlowDPS even though the active script is latent L2 guidance, and some examples use stale CLI flags (`--lora_path`, `--val_csv`, `--dps_eta`) that do not match `validate_dps.py`.

## 4. Minimal patch plan

### Patch 1: make `spad_forward.py` match the spec

File: `diffsynth/diffusion/spad_forward.py`

- Replace `intensity_to_detection_prob` with:
  - `rgb_to_linear_intensity(...)` helper
  - `intensity_to_exposure(...)` returning `H = softplus(alpha * L + beta)`
  - optional `beta` buffer / constructor argument

- Replace `log_likelihood` implementation:
  - compute `H = intensity_to_exposure(...)`
  - clamp `H` to `>= 1e-6`
  - use:
    - `log_p = torch.log(-torch.expm1(-H))`
    - `log_1mp = -H`
  - single-frame NLL:
    - `ll = y * log_p + (1-y) * log_1mp`
  - multi-frame:
    - `S = measurement * num_frames`
    - `ll = S * log_p + (num_frames - S) * log_1mp`

- Make `measurement_loss` return only NLL by default.
  - If the L2 heuristic is kept, make it opt-in and off by default.

### Patch 2: fix DPS correction sign

Files:
- `diffsynth/diffusion/flow_dps.py`
- `diffsynth/diffusion/latent_dps.py`

Current lines:
- `flow_dps.py:132`
- `latent_dps.py:69`

Change:
- Replace `correction = -guidance_scale * grad ...`
- With `correction = +guidance_scale * grad ...`

Reason:
- `pipe.step()` uses `latents_next = latents + noise_pred * (sigma_next - sigma)`.
- Since `(sigma_next - sigma) < 0`, adding `+grad` to `noise_pred` is what moves the sample in `-grad` direction.

### Patch 3: add proper preconditioning and schedule

Files:
- `diffsynth/diffusion/flow_dps.py`
- `diffsynth/diffusion/latent_dps.py`

Changes:
- After computing `grad`, normalize by either:
  - `grad.abs().mean() + eps`, or
  - `sqrt(nll / K_pixels) + eps`
- Keep optional global norm clipping.
- Add schedule option:
  - `ramp_up`
  - or `sigma_ramp`
- Make that the recommended default for actual physics guidance.

### Patch 4: keep the active script honest

File: `validate_dps.py`

Choose one of these minimal directions:

- Option A: keep it as latent heuristic, but rename the interface and docs so it no longer claims Bernoulli / FlowDPS physics guidance.
- Option B: if pixel-space FlowDPS is re-enabled later, add a separate entry point instead of overloading the current script name.

This is mostly a correctness-of-claims fix, not a code-path fix.

### Patch 5: make consistency weighting stable across timesteps

File: `diffsynth/diffusion/consistency_loss.py`

Current:
- `loss = loss_sft + consistency_weight * loss_consistency`

Minimal change:
- either multiply `loss_consistency` by the same scheduler weight,
- or explicitly document that `consistency_weight` is intended to vary with timestep.

## 5. Micro-test plan

### Unit test 1: stable `log1mexp`

Goal:
- ensure the new likelihood stays finite for tiny `H`

Assertions:
- for `H = [1e-6, 1e-8, 1e-10]`, `torch.log(-torch.expm1(-H_clamped))` is finite
- no `nan`, no `inf`

### Unit test 2: gradient sign for `y = 0`

Goal:
- verify `dNLL/dH > 0`

Setup:
- `H = torch.tensor([0.1], requires_grad=True)`
- `nll = H`

Assertion:
- `H.grad.item() > 0`

### Unit test 3: gradient sign for `y = 1`

Goal:
- verify `dNLL/dH < 0`

Setup:
- `H = torch.tensor([0.1], requires_grad=True)`
- `nll = -torch.log(-torch.expm1(-H))`

Assertion:
- `H.grad.item() < 0`

### End-to-end test: guidance step should lower the target loss

Two tiny scalar tests are enough:

- For latent heuristic:
  - build a scalar toy with the real scheduler sign convention
  - verify that adding the current code’s correction raises the loss
  - verify that flipping the sign lowers the loss

- For pixel-space NLL:
  - make a dummy `vae_decoder` that is identity on a 1-channel latent
  - compute one `compute_dps_correction` step on a tiny tensor
  - verify that taking one scheduler step with the corrected sign lowers NLL

## 6. Direct answer to the frame question

Current frame consistency training uses exactly 2 frames per sample draw.

It does not aggregate 3+ frames in a single loss, and it does not enumerate all frame pairs in one step.

However, it does sample from a larger pool of available frame folders, so over training it sees many different random pairs from the 7 single-frame folders.

## 7. Open questions

No blocking questions for the audit itself.

If you want code changes next, the clean order is:
1. fix the DPS correction sign
2. decide whether `validate_dps.py` should remain explicitly heuristic or be replaced by a true physics path
3. patch `spad_forward.py` to the stable `H`-based likelihood
4. only then revisit schedules / PaDIS normalization
