# Code Review: Physics DPS Loss & Per-Frame Consistency Loss

## Files Reviewed

| File | Lines | Purpose |
|------|-------|---------|
| [spad_forward.py](file:///home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD/diffsynth/diffusion/spad_forward.py) | 141 | SPAD Bernoulli forward model + NLL |
| [flow_dps.py](file:///home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD/diffsynth/diffusion/flow_dps.py) | 325 | Pixel-space FlowDPS (VAE decode each step) |
| [latent_dps.py](file:///home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD/diffsynth/diffusion/latent_dps.py) | 71 | Latent-space DPS (lightweight, actually used) |
| [validate_dps.py](file:///home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD/validate_dps.py) | 200 | Inference script (uses latent DPS) |
| [consistency_loss.py](file:///home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD/diffsynth/diffusion/consistency_loss.py) | 72 | Per-frame consistency loss |
| [train_consistency.py](file:///home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD/train_consistency.py) | 219 | Consistency training script |
| [paired_spad_dataset.py](file:///home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD/paired_spad_dataset.py) | 139 | Dataset: pairs 2 random frame folders per scene |

---

## A. SPAD Forward Model (`spad_forward.py`) — NLL Issues

### 🔴 Issue 1: Missing `log1mexp` — Numerically Unstable NLL

Your spec explicitly says:

> NLL(H;y) = Σ_i [ (1 - y_i) * H_i  - y_i * log(1 - exp(-H_i)) ]
>
> IMPORTANT: implement `log(1 - exp(-H))` stably: `log1mexp(H) = log(-expm1(-H))`

The current code at [line 75](file:///home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD/diffsynth/diffusion/spad_forward.py#L75):

```python
ll = measurement * torch.log(p) + (1.0 - measurement) * torch.log(1.0 - p)
```

This computes `log(p) = log(1 - exp(-H))` directly. When `H` is small (low intensity), `exp(-H) ≈ 1`, so `p ≈ 0` and `log(p) → -∞`. The `clamp(eps, 1-eps)` on `p` at line 47 masks the issue but introduces **bias** — for small `H`, `p` gets artificially floored to `1e-6`, distorting the gradient.

**The spec says to use `torch.expm1`:**

```python
# Stable: log(1 - exp(-H)) = log(-expm1(-H))
log_p = torch.log(-torch.expm1(-H))  # numerically stable for small H
```

### 🔴 Issue 2: Missing `softplus` Mapping for H

Your spec says:

> H = softplus(alpha * L(x) + beta)

The current code uses `H = alpha * intensity` (a simple linear scaling, line 46). This means:
- **H can be exactly 0** when `intensity = 0`, causing `p = 0, log(p) = -∞`
- No `beta` offset is implemented
- No `softplus` to guarantee `H > 0`

**Suggested fix:**

```python
def intensity_to_H(self, intensity):
    """Map intensity to expected photon count via softplus."""
    intensity = intensity.clamp(min=0.0)
    H = F.softplus(self.alpha * intensity + self.beta)
    return H
```

### 🟡 Issue 3: NLL Formula Doesn't Match Spec's Canonical Form

The spec writes the NLL as:

```
NLL = Σ_i [ (1 - y_i) * H_i  - y_i * log(1 - exp(-H_i)) ]
```

But the current code computes it in terms of `p` (detection probability):

```python
ll = y * log(p) + (1-y) * log(1-p)
```

These are mathematically equivalent (`log(1-p) = log(exp(-H)) = -H` and `log(p) = log(1-exp(-H))`), but the **H-based formulation is more numerically stable** because `log(1-p) = -H` is trivially stable, and only the `y=1` term needs the `log1mexp` trick. The p-based formulation has instability on **both** terms.

### 🟡 Issue 4: No Inverse Gamma (sRGB → Linear)

Your spec says:

> Convert generated RGB to linear intensity (undo gamma if needed) or use luminance.

The code in `SPADMeasurementConsistency.forward()` (line 137) just takes `decoded_image.mean(dim=1)` — averaging sRGB channels without applying inverse gamma. This is a physics modeling error. The Bernoulli model should apply to linear intensity, not gamma-compressed values.

**Fix (in `flow_dps.py` line 108 and `spad_forward.py` line 137):**

```python
def srgb_to_linear(x):
    """Inverse sRGB gamma (simplified)."""
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

intensity = srgb_to_linear(decoded_01).mean(dim=1, keepdim=True)
```

---

## B. FlowDPS Guidance (`flow_dps.py` + `latent_dps.py`)

### 🟢 `flow_dps.py` — Pixel-Space DPS (Correct but Unused)

The pixel-space DPS in `flow_dps.py` is **conceptually correct**:
- ✅ Estimates `x_0_hat = x_t - sigma * v_theta` (flow-Tweedie)
- ✅ Decodes through VAE, computes Bernoulli NLL, backprops gradient to latents
- ✅ Applies gradient-based correction to velocity: `noise_pred += -eta * grad`
- ✅ Has schedule, clamping, step gating

But it's **not the version actually used at inference** — `validate_dps.py` uses `latent_dps.py` instead (to avoid OOM from VAE decoding at every step).

### 🔴 `latent_dps.py` — NOT Physics-Based at All

The latent-space DPS is a **pure L2 latent-consistency** loss:

```python
x0_hat = latents - sigma * noise_pred
diff = x0_hat - spad_latent  # z_spad = VAE_encode(SPAD_image)
correction = -eta * normalize(diff)
```

This is **not** the Bernoulli NLL from the spec. It's just pulling the predicted clean latent toward the VAE-encoded SPAD measurement. This is:
- ❌ Not the physics forward model (no Bernoulli likelihood)
- ❌ Comparing latent representations, not physical measurements
- ⚠️ The VAE-encoded SPAD binary image has no particular physical meaning in latent space

**This is flagged as "risky"** per your spec: it's essentially a hard projection in latent space rather than a principled likelihood gradient. It may help visually but it's not physics-consistent DPS.

### 🟡 Guidance Schedule

Your spec says:

> Keep eta small early; guidance can grow later when structure emerges.
> Example: eta(t) = eta0 * (sigma_t² / sigma_max²) or monotonic ramp-up.

The current schedule options are `constant`, `linear_decay`, `cosine` — all of which either stay flat or **decrease** over time. None implement the recommended **ramp-up** (grow as denoising proceeds). The `linear_decay` default actually does the opposite of the spec recommendation.

### 🟡 PaDIS-style Normalization Missing

The spec says:

> step = eta(t) / (mean(|grad|) + eps)  OR  eta(t) / (sqrt(nll/K_pixels) + eps)
> and gradient clipping (global norm clip) as needed

In `flow_dps.py`, only **element-wise clamping** is done (line 130), not the PaDIS-style normalization by mean gradient magnitude. In `latent_dps.py`, the gradient IS normalized by its L2 norm (line 67–68), which is closer to PaDIS but uses L2 norm instead of mean absolute value.

---

## C. Per-Frame Consistency Loss (`consistency_loss.py`)

### 🟢 Core Design — Correct

The consistency loss framework is sound:
- ✅ Same noisy latent `z_t` and timestep `t` for both frames
- ✅ `L = L_flow_match(F1) + λ * ||v_F1 - v_F2||²`
- ✅ Stop-gradient on F2 (line 63) — smart memory optimization

### Your Question: Does It Use Only Two Frames?

**Yes, it uses exactly 2 frames per step** (F1, F2). However:

- The `PairedSPADDataset` randomly samples F1 and F2 from **up to 7 different frame folders** each time `__getitem__` is called. So across training, all pairwise combinations of frame folders are covered stochastically.
- Each epoch, every scene gets a **fresh random pair** — this is proper data augmentation.
- It does **not** compute consistency across >2 frames simultaneously (e.g., no triplet or N-way consistency in a single batch).

This is the right design choice — N-way consistency would require N forward passes per step and wouldn't give proportionally better gradients. The random pairing across epochs covers all combinations.

### 🟡 Issue: F2 Goes Through VAE Encoder, Not ControlNet

In [train_consistency.py line 72–73](file:///home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD/train_consistency.py#L72-L73):

```python
f2_latent = self.pipe.vae_encoder(f2_tensor, tiled=False)
inputs[0]["controlnet_conditionings_f2"] = [f2_latent]
```

Then in `consistency_loss.py` line 57:

```python
inputs_f2["controlnet_conditionings"] = conditionings_f2  # [f2_latent]
```

The F2 conditioning is the **VAE-encoded F2 image** passed directly as `controlnet_conditionings`. But `controlnet_conditionings` is supposed to be the **output of the ControlNet** (processed features), not a raw VAE latent. For F1, it goes through the normal pipeline units (which run the ControlNet), but F2 skips this.

This means F1 and F2 are processed via **different pathways** — F1 through ControlNet, F2 through VAE encoder only. The consistency loss would be enforcing agreement between two very different conditioning signals, which is **not the intended behavior**.

> [!CAUTION]
> This looks like a bug. F2 should also be processed through the ControlNet to produce proper `controlnet_conditionings_f2`, not just VAE-encoded. The pipeline's `unit_runner` for ControlNet should be called on F2 as well.

**However**, running the ControlNet twice per step would double the most expensive part of the forward pass. This is likely why it was implemented this way — as a memory/compute shortcut. If the ControlNet is the bottleneck, you could:
1. Run ControlNet for F2 in a `torch.no_grad()` block (already done for the DiT pass)
2. Cache F2 ControlNet outputs across the epoch (but they depend on the noisy latent state)

---

## D. Summary Table

| Component | Issue | Severity | Status |
|-----------|-------|----------|--------|
| NLL: No `log1mexp` | Numerical instability for small H | 🔴 Bug | Not implemented |
| NLL: No `softplus(H)` | H=0 → log(0) → NaN/Inf | 🔴 Bug | Not implemented |
| NLL: sRGB not linearized | Physics model applied to wrong color space | 🟡 Modeling error | Not implemented |
| `latent_dps.py`: Not physics-based | L2 latent loss ≠ Bernoulli NLL | 🔴 Design gap | By design (OOM workaround) |
| Guidance schedule: No ramp-up | Spec says eta should grow, code only decays | 🟡 Config gap | Easy to add |
| PaDIS normalization | Missing adaptive step normalization | 🟡 Missing feature | Not implemented |
| Consistency: F2 via VAE not ControlNet | F2 conditioning uses wrong pathway | 🔴 Likely bug | See discussion above |
| Consistency: Only 2 frames | Pairs, not N-way | 🟢 Correct | By design, proper |

---

## E. Recommended Fixes (Priority Order)

### 1. Fix NLL Stability (Critical — affects `flow_dps.py` correctness)

Rewrite `spad_forward.py` `log_likelihood` to use H-based formulation with `log1mexp`:

```python
def log_likelihood(self, intensity, measurement):
    H = F.softplus(self.alpha * intensity + self.beta).clamp(min=1e-6)
    # log(1 - exp(-H)) = log(-expm1(-H)), stable for small H
    log_p = torch.log(-torch.expm1(-H))
    log_1mp = -H  # log(exp(-H)) = -H, trivially stable
    ll = measurement * log_p + (1.0 - measurement) * log_1mp
    return ll.flatten(1).sum(1)
```

### 2. Add sRGB → Linear conversion

```python
def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
```

### 3. Add ramp-up guidance schedule

```python
elif schedule == "sigma_ramp":
    # eta grows as sigma decreases (structure emerges)
    return relative_pos  # 0→1 over the active range
```

### 4. Fix F2 ControlNet pathway in consistency training

Process F2 through the ControlNet (under `torch.no_grad()`) instead of just VAE encoding it.

### 5. Consider making pixel-space DPS usable

The pixel-space `flow_dps.py` is correct in principle but OOMs. Options:
- Apply guidance only every N steps (e.g., every 5th step)
- Use half-resolution VAE decode for gradient computation
- Use the latent DPS as approximation but document it clearly as NOT physics-based
