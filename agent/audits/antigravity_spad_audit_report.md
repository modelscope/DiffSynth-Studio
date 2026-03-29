# SPAD Physics DPS & Consistency Loss Audit Report

## 1. Implementation Map

| Component | File Path | Function / Class | Description |
|-----------|-----------|------------------|-------------|
| **Forward Model & NLL** | `diffsynth/diffusion/spad_forward.py` | `SPADForwardModel` (`intensity_to_detection_prob`, `log_likelihood`) | 1-bit SPAD Bernoulli forward model and NLL computation. |
| **Pixel-Space FlowDPS** | `diffsynth/diffusion/flow_dps.py` | `compute_dps_correction`, `get_guidance_weight` | DPS gradient correction decoding via VAE to pixel space. |
| **Latent-Space DPS** | `diffsynth/diffusion/latent_dps.py` | `compute_latent_dps_correction` | Lightweight L2 latent consistency used as an approximation for physics DPS. |
| **DPS Inference** | `validate_dps.py` | `make_dps_step_fn` | Active inference script utilizing latent-space DPS. |
| **Consistency Loss** | `diffsynth/diffusion/consistency_loss.py` | `FlowMatchSFTWithConsistencyLoss` | Enforces structural consistency on predicted velocity for two different frames. |
| **Consistency Target** | `train_consistency.py` | `FluxConsistencyTrainingModule.forward` | Extracts F2 features to pass down as `controlnet_conditionings_f2`. |
| **Consistency Data Pairs** | `paired_spad_dataset.py` | `PairedSPADDataset.__getitem__` | Samples exactly TWO random binary frames per scene from different frame folders. |

---

## 2. Correctness Report

### Part A1: Forward model + likelihood

*   **softplus Mapping**: 🔴 **FAIL**. `spad_forward.py`, line 45 calculates `p = 1.0 - torch.exp(-self.alpha * intensity)`. There is no `softplus` and no `beta` offset. When `intensity = 0`, `H = 0` exactly.
*   **sRGB to Linear Conversion**: 🔴 **FAIL**. In `flow_dps.py` line 108 and `spad_forward.py` line 137, the VAE `decoded_image` (which is typically in sRGB / gamma space) is simply averaged `decoded_image.mean(dim=1)` without any inverse gamma linearisation.
*   **Numerical Stability (log1mexp)**: 🔴 **FAIL**. The code uses `torch.log(1.0 - p)` which is naive `log(exp(-H))` and `torch.log(p)` which evaluates to `log(1 - exp(-H))`. Without using the `-expm1` trick, the calculation will underflow for small `H` yielding `log(0) = -inf`. The explicit clamping on `p` in line 47 (`p.clamp(eps, 1.0 - eps)`) avoids explicit `NaNs` but heavily biases gradients when H is small.

### Part A2: PaDIS/DPS-style integration

*   **Pixel-space vs Latent-space**: 🟡 **WARNING**. `flow_dps.py` implements the mathematically correct pixel-space DPS, but the main pipeline script `validate_dps.py` is hardcoded to use `latent_dps.py` which computes a purely L2 latent consistency loss (`||x_0_hat - z_spad||^2`).
*   **Gradient Preconditioning (PaDIS-like)**: 🔴 **FAIL**. `flow_dps.py` scales the gradient by exactly `eta` (line 132), followed by a naive clamp (`clamp(-clip, clip)`). It does not normalize by the mean gradient magnitude (`mean(\dvert grad\dvert)`) nor by `sqrt(nll)`. Latent DPS normalizes by L2 norm instead.
*   **Guidance Schedule `eta(t)`**: 🔴 **FAIL**. The implemented schedules are `constant`, `linear_decay`, and `cosine`. All of them maintain or *decay* eta over time. The spec explicitly advises a ramp-up schedule ("keep eta small early; guidance can grow later").

### Part A3: Weak-signal reality check

*   **Hard Projection Risk**: 🔴 **FAIL (in Latent DPS)**. Because the latent DPS performs an L2 distance match directly onto a VAE-encoded SPAD image (`z_spad`), it is functioning as a hard projection in latent space instead of an authentic weak-signal likelihood correction. The VAE-encoded binary image is not a physically meaningful representation for L2 matching.

### Consistency Loss Audit

**Q: Does it use only two frames, or multiple pairs?**
It uses **exactly 2 frames per step**. (See `paired_spad_dataset.py` lines 122-127).
However, it randomly samples these two frames from a larger pool of up to 7 available folders for the same scene every time `__getitem__` is called. Therefore, across epochs, the model sees a vast number of diverse frame *pairs* for each scene.

**Loss Form:**
The loss enforces image/latent-space consistency (specifically, difference in predicted flow velocity `||eps_F1 - eps_F2||^2`) rather than measurement-space consistency.

**Bug Noted (from previous audit):**
*Note: A previous suspicion that F2 bypassed the ControlNet pipeline was incorrect. The `unit_runner` merely performs VAE-encoding, and the actual ControlNet forward pass happens identically for both F1 and F2 inside `pipe.model_fn`.*

---

## 3. Minimal Patch Plan

### Patch 1: Numerical Stability & softplus in `spad_forward.py`
Replace `intensity_to_detection_prob` and `log_likelihood` in `SPADForwardModel`:

```python
    def intensity_to_H(self, intensity: torch.Tensor) -> torch.Tensor:
        """Map intensity to expected photon count via softplus."""
        intensity = intensity.clamp(min=0.0)
        # Using softplus to ensure H > 0. Alpha/Beta could easily be adapted later.
        beta = 1e-4  # minimal beta
        H = F.softplus(self.alpha * intensity + beta).clamp(min=1e-6)
        return H

    def log_likelihood(
        self, intensity: torch.Tensor, measurement: torch.Tensor
    ) -> torch.Tensor:
        H = self.intensity_to_H(intensity)
        
        if self.num_frames == 1:
            # log(1 - exp(-H)) = log(-expm1(-H))
            log_p = torch.log(-torch.expm1(-H))
            # log(exp(-H)) = -H
            log_1mp = -H 
            ll = measurement * log_p + (1.0 - measurement) * log_1mp
        else:
            y = measurement * self.num_frames
            n_minus_y = self.num_frames - y
            log_p = torch.log(-torch.expm1(-H))
            log_1mp = -H
            ll = y * log_p + n_minus_y * log_1mp

        return ll.flatten(1).sum(1)
```

### Patch 2: sRGB to Linear Transformation (`flow_dps.py` & `spad_forward.py`)
Add an inverse gamma utility and use it before flattening color channels to intensity:

```python
def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

# Usage in flow_dps.py and spad_forward.py (SPADMeasurementConsistency):
linear_decoded = srgb_to_linear(decoded_image)
intensity = linear_decoded.mean(dim=1, keepdim=True)
```

### Patch 3: PaDIS-style Gradient Preconditioning (`flow_dps.py`)
In `compute_dps_correction` line 125, update gradient processing:

```python
    # PaDIS preconditioning
    mean_grad_norm = grad.abs().mean() + 1e-8
    grad = grad / mean_grad_norm
    
    if gradient_clamp > 0:
        grad = grad.clamp(-gradient_clamp, gradient_clamp)
```

### Patch 4: Ramp-up schedule (`flow_dps.py`)
In `get_guidance_weight` line 152, add:

```python
    elif schedule == "ramp_up":
        return relative_pos  # Grows from 0 to 1 linearly over the active range
```

### Patch 5: Consistency Loss F2 Pathway (`train_consistency.py`)
Run F2 through the ControlNet instead of just the VAE. To save memory, wrap the ControlNet execution for F2 in `torch.no_grad()`.

---

## 4. Micro-Test Plan

You can place these in a temporary file (e.g. `test_dps.py`) and execute via pytest.

```python
import torch
import torch.nn.functional as F

# 1. Test log1mexp Stability
def test_log1mexp_stability():
    H = torch.tensor([1e-6, 1e-8], dtype=torch.float32)
    # Naive will fail or return log(0)
    naive_p = 1.0 - torch.exp(-H)
    naive_log_p = torch.log(naive_p)
    # Expm1 will succeed
    stable_log_p = torch.log(-torch.expm1(-H))
    
    assert not torch.isnan(stable_log_p).any()
    assert not torch.isinf(stable_log_p).any()

# 2. Test Gradient Sign
def test_gradient_sign():
    H = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
    y1 = torch.tensor([1.0]) # Measurement = 1
    y0 = torch.tensor([0.0]) # Measurement = 0
    
    # NLL for y=1
    nll1 = -torch.log(-torch.expm1(-H))
    nll1.backward()
    assert H.grad.item() < 0 # dNLL/dH < 0 when y=1 (increasing H lowers NLL)
    
    H.grad.zero_()
    
    # NLL for y=0
    nll0 = H
    nll0.backward()
    assert H.grad.item() > 0 # dNLL/dH > 0 when y=0 (decreasing H lowers NLL)

# 3. Test sRGB to Linear Conversion
def test_srgb_to_linear():
    def srgb_to_linear(x):
        return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
        
    srgb = torch.tensor([0.0, 0.5, 1.0])
    lin = srgb_to_linear(srgb)
    assert lin[0] == 0.0
    assert lin[2] == 1.0
    assert lin[1] < 0.5 # Gamma curves down

# 4. End-to-End Fake Guidance
def test_e2e_guidance_decreases_nll():
    # Setup dummy latents and SPAD measurement
    latents = torch.randn(1, 4, 64, 64, requires_grad=True)
    spad_meas = torch.randint(0, 2, (1, 1, 64, 64)).float()
    
    # Forward pass mock
    intensity = F.softplus(latents[:, 0:1, :, :]).clamp(min=1e-6)
    log_p = torch.log(-torch.expm1(-intensity))
    log_1mp = -intensity
    
    nll = -(spad_meas * log_p + (1.0 - spad_meas) * log_1mp).mean()
    
    grad = torch.autograd.grad(nll, latents)[0]
    
    # Apply step
    latents_new = latents - 0.1 * grad
    
    # Compute new NLL
    intensity_new = F.softplus(latents_new[:, 0:1, :, :]).clamp(min=1e-6)
    nll_new = -(spad_meas * torch.log(-torch.expm1(-intensity_new)) + (1.0 - spad_meas) * (-intensity_new)).mean()
    
    assert nll_new.item() < nll.item()

---

## 5. Opus 4.6 Fixes Verification (Updated)

I have reviewed the fixes applied by Opus 4.6. Here is the status:

*   ✅ **NLL Stability & Softplus (`spad_forward.py`)**: Fixed. Uses `softplus(alpha * I + beta)` and stable `_log1mexp`.
*   ✅ **sRGB to Linear Conversion (`spad_forward.py`, `flow_dps.py`)**: Fixed. The `srgb_to_linear` utility is implemented and applied before the forward model.
*   ✅ **PaDIS Preconditioning (`flow_dps.py`, `latent_dps.py`)**: Fixed. Gradient is now normalized by `grad.abs().mean() + 1e-8`.
*   ✅ **DPS Guidance Schedule (`flow_dps.py`)**: Fixed. The `ramp_up` and `sigma_ramp` schedules were properly added to `get_guidance_weight`.
*   ✅ **Consistency Loss F2 Pathway (`train_consistency.py`)**: Verified correct. The manual substitution of the VAE-encoded latent into `controlnet_conditionings_f2` perfectly mirrors the behavior of `FluxImageUnit_ControlNet`, ensuring F2 correctly passes through the ControlNet forward pass inside `model_fn`. No fix was needed here.
```
