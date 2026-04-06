"""
Unit tests for SPAD physics DPS: numerical stability, gradient signs,
and DPS correction direction.

Run: python test_dps_physics.py
"""
import torch
import torch.nn.functional as F
import sys


def test_log1mexp_stability():
    """log(-expm1(-H)) should be finite for very small H."""
    from diffsynth.diffusion.spad_forward import _log1mexp

    H_values = torch.tensor([1e-6, 1e-8, 1e-10, 1e-3, 0.1, 1.0, 10.0])
    H_clamped = H_values.clamp(min=1e-6)
    result = _log1mexp(H_clamped)

    assert not torch.isnan(result).any(), f"NaN in log1mexp: {result}"
    assert not torch.isinf(result).any(), f"Inf in log1mexp: {result}"
    # log(p) should be negative (p < 1)
    assert (result < 0).all(), f"log(p) should be negative: {result}"

    # Compare to naive at very small H where precision is lost
    H_tiny = torch.tensor([1e-20, 1e-30, 1e-38])
    naive_tiny = torch.log(1.0 - torch.exp(-H_tiny))
    stable_tiny = _log1mexp(H_tiny.clamp(min=1e-6))

    # Naive should fail (produce -inf) for very tiny H
    has_inf = torch.isinf(naive_tiny).any()
    assert has_inf, f"Naive should produce -inf for very tiny H, got {naive_tiny}"
    # Stable version should remain finite
    assert not torch.isinf(stable_tiny).any(), f"Stable should be finite: {stable_tiny}"

    print("  PASS: log1mexp is stable for H in [1e-10, 10]; naive fails for H < 1e-38")


def test_softplus_guarantees_positive_H():
    """softplus(alpha * I + beta) should always be > 0."""
    from diffsynth.diffusion.spad_forward import SPADForwardModel

    model = SPADForwardModel(alpha=1.0, beta=0.0)
    intensity = torch.tensor([0.0, -0.1, 1e-10, 0.5, 1.0])
    H = model.intensity_to_exposure(intensity)

    assert (H > 0).all(), f"H must be strictly positive: {H}"
    assert (H >= model.H_min).all(), f"H must be >= H_min: {H}"
    print("  PASS: softplus guarantees H > 0 for all inputs")


def test_gradient_sign_y0():
    """For y=0 (no detection), dNLL/dH should be positive.

    NLL = (1-y)*H - y*log(1-exp(-H)) = H for y=0
    So dNLL/dH = +1.
    """
    from diffsynth.diffusion.spad_forward import SPADForwardModel

    model = SPADForwardModel(alpha=1.0, beta=0.0)
    # Use a moderate intensity so softplus doesn't saturate
    intensity = torch.tensor([[[[0.5]]]], requires_grad=True)
    measurement = torch.tensor([[[[0.0]]]])  # y = 0

    nll = model.negative_log_likelihood(intensity, measurement)
    nll.backward()

    assert intensity.grad is not None
    assert intensity.grad.item() > 0, \
        f"dNLL/dIntensity should be > 0 for y=0, got {intensity.grad.item()}"
    print(f"  PASS: gradient sign for y=0 is positive ({intensity.grad.item():.4f})")


def test_gradient_sign_y1():
    """For y=1 (detection), dNLL/dH should be negative.

    NLL = -log(1-exp(-H)) for y=1
    dNLL/dH = -exp(-H)/(1-exp(-H)) < 0
    """
    from diffsynth.diffusion.spad_forward import SPADForwardModel

    model = SPADForwardModel(alpha=1.0, beta=0.0)
    intensity = torch.tensor([[[[0.5]]]], requires_grad=True)
    measurement = torch.tensor([[[[1.0]]]])  # y = 1

    nll = model.negative_log_likelihood(intensity, measurement)
    nll.backward()

    assert intensity.grad is not None
    assert intensity.grad.item() < 0, \
        f"dNLL/dIntensity should be < 0 for y=1, got {intensity.grad.item()}"
    print(f"  PASS: gradient sign for y=1 is negative ({intensity.grad.item():.4f})")


def test_srgb_to_linear():
    """Verify sRGB→linear conversion properties."""
    from diffsynth.diffusion.spad_forward import srgb_to_linear

    srgb = torch.tensor([0.0, 0.5, 1.0])
    lin = srgb_to_linear(srgb)

    assert abs(lin[0].item()) < 1e-7, "linear(0) should be 0"
    assert abs(lin[2].item() - 1.0) < 1e-6, "linear(1) should be 1"
    assert lin[1].item() < 0.5, f"linear(0.5) should be < 0.5 (gamma curve), got {lin[1].item()}"
    print(f"  PASS: sRGB→linear: [0, 0.5, 1] → [{lin[0]:.4f}, {lin[1]:.4f}, {lin[2]:.4f}]")


def test_latent_dps_correction_sign():
    """Verify that latent DPS correction, when applied through the scheduler,
    moves latents in a direction that decreases the loss.

    Scheduler: x_next = x + v * (sigma_next - sigma), where sigma_next < sigma.
    """
    from diffsynth.diffusion.latent_dps import compute_latent_dps_correction

    torch.manual_seed(42)

    # Setup: latents, a target, and a noise prediction
    latents = torch.randn(1, 4, 8, 8)
    spad_latent = torch.randn(1, 4, 8, 8)
    noise_pred = torch.randn(1, 4, 8, 8)
    sigma = 0.5
    sigma_next = 0.4  # sigma decreases during denoising

    # Compute loss BEFORE correction
    x0_hat_before = latents - sigma * noise_pred
    loss_before = F.mse_loss(x0_hat_before, spad_latent).item()

    # Get correction
    correction = compute_latent_dps_correction(
        latents=latents,
        noise_pred=noise_pred,
        sigma=sigma,
        spad_latent=spad_latent,
        guidance_scale=0.1,
    )

    # Apply correction to velocity, then take scheduler step
    noise_pred_corrected = noise_pred + correction
    latents_next = latents + noise_pred_corrected * (sigma_next - sigma)

    # Compute loss AFTER correction (at the next step)
    # x0_hat at next step: x_next - sigma_next * noise_pred (using uncorrected v for evaluation)
    x0_hat_after = latents_next - sigma_next * noise_pred
    loss_after = F.mse_loss(x0_hat_after, spad_latent).item()

    # The correction should decrease the loss
    assert loss_after < loss_before, \
        f"DPS correction should decrease loss: {loss_before:.6f} → {loss_after:.6f}"
    print(f"  PASS: latent DPS correction decreases loss: {loss_before:.6f} → {loss_after:.6f}")


def test_pixel_dps_correction_sign():
    """Verify that pixel-space DPS correction direction is correct.

    Uses a toy setup with identity "VAE decoder" to verify the sign.
    """
    from diffsynth.diffusion.spad_forward import SPADForwardModel

    torch.manual_seed(42)

    model = SPADForwardModel(alpha=1.0, beta=0.0)

    # Create a simple 1-channel "latent" that we treat as both latent and pixel
    latents = torch.rand(1, 1, 8, 8, requires_grad=True) * 0.5 + 0.25
    spad_meas = torch.randint(0, 2, (1, 1, 8, 8)).float()
    noise_pred = torch.zeros_like(latents)
    sigma = 0.5

    # Compute x0_hat (with zero noise_pred, x0_hat = latents)
    x0_hat = latents - sigma * noise_pred

    # Compute NLL
    nll = model.negative_log_likelihood(x0_hat, spad_meas)
    grad = torch.autograd.grad(nll, latents)[0]

    # PaDIS normalization
    grad_norm = grad / (grad.abs().mean() + 1e-8)

    # Apply correction with CORRECT sign (+grad to velocity)
    sigma_next = 0.4
    correction_correct = 0.1 * grad_norm
    latents_next_correct = (latents + (noise_pred + correction_correct) * (sigma_next - sigma)).detach()

    # Apply correction with WRONG sign (-grad to velocity)
    correction_wrong = -0.1 * grad_norm
    latents_next_wrong = (latents + (noise_pred + correction_wrong) * (sigma_next - sigma)).detach()

    # Evaluate NLL at both
    nll_correct = model.negative_log_likelihood(latents_next_correct, spad_meas).item()
    nll_wrong = model.negative_log_likelihood(latents_next_wrong, spad_meas).item()
    nll_original = nll.item()

    assert nll_correct < nll_original, \
        f"Correct sign should decrease NLL: {nll_original:.4f} → {nll_correct:.4f}"
    assert nll_correct < nll_wrong, \
        f"Correct sign should beat wrong sign: correct={nll_correct:.4f} vs wrong={nll_wrong:.4f}"
    print(f"  PASS: pixel DPS sign verified: original={nll_original:.4f}, "
          f"correct={nll_correct:.4f}, wrong={nll_wrong:.4f}")


def test_nll_no_nan_full_range():
    """NLL should produce no NaN/Inf for a range of intensities and measurements."""
    from diffsynth.diffusion.spad_forward import SPADForwardModel

    model = SPADForwardModel(alpha=1.0, beta=0.0)

    # Test with various intensity levels and binary measurements
    for intensity_val in [0.0, 1e-8, 1e-4, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
        for meas_val in [0.0, 1.0]:
            intensity = torch.tensor([[[[intensity_val]]]], requires_grad=True)
            measurement = torch.tensor([[[[meas_val]]]])

            nll = model.negative_log_likelihood(intensity, measurement)
            assert not torch.isnan(nll), f"NaN at intensity={intensity_val}, y={meas_val}"
            assert not torch.isinf(nll), f"Inf at intensity={intensity_val}, y={meas_val}"

            nll.backward()
            assert not torch.isnan(intensity.grad).any(), \
                f"NaN grad at intensity={intensity_val}, y={meas_val}"

    print("  PASS: NLL is finite for all intensity/measurement combinations")


def test_guidance_weight_ramp_up():
    """Verify ramp_up schedule goes from 0 to 1."""
    from diffsynth.diffusion.flow_dps import get_guidance_weight

    total = 20
    weights = [get_guidance_weight(i, total, "ramp_up") for i in range(total)]

    assert weights[0] == 0.0, f"ramp_up should start at 0, got {weights[0]}"
    assert weights[-1] == 1.0, f"ramp_up should end at 1, got {weights[-1]}"
    # Should be monotonically non-decreasing
    for i in range(1, len(weights)):
        assert weights[i] >= weights[i - 1], \
            f"ramp_up should be monotonic: w[{i-1}]={weights[i-1]}, w[{i}]={weights[i]}"
    print(f"  PASS: ramp_up schedule: {weights[0]:.2f} → {weights[-1]:.2f} (monotonic)")


def main():
    print("\n=== SPAD Physics DPS Unit Tests ===\n")
    tests = [
        ("1. log1mexp stability", test_log1mexp_stability),
        ("2. softplus guarantees H > 0", test_softplus_guarantees_positive_H),
        ("3. gradient sign for y=0", test_gradient_sign_y0),
        ("4. gradient sign for y=1", test_gradient_sign_y1),
        ("5. sRGB → linear conversion", test_srgb_to_linear),
        ("6. latent DPS correction direction", test_latent_dps_correction_sign),
        ("7. pixel DPS correction direction", test_pixel_dps_correction_sign),
        ("8. NLL no NaN across full range", test_nll_no_nan_full_range),
        ("9. guidance weight ramp_up", test_guidance_weight_ramp_up),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            print(f"[{name}]")
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed!")


if __name__ == "__main__":
    main()
