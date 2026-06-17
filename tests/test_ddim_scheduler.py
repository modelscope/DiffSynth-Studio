import torch

from diffsynth.diffusion.ddim_scheduler import DDIMScheduler


def test_bf16_timestep_does_not_overflow_alphas_cumprod():
    # Regression for #1499: in bfloat16 training a timestep of 999 rounds up
    # (999 is not exactly representable in bf16), which used to index past
    # alphas_cumprod (length num_train_timesteps) and raise IndexError.
    scheduler = DDIMScheduler()
    timestep = torch.tensor([999.0], dtype=torch.bfloat16)

    # The bf16 round-trip pushes the raw timestep past the valid index range ...
    assert int(timestep.flatten().tolist()[0]) >= scheduler.num_train_timesteps

    sample = torch.zeros(1, 4, 8, 8)
    noise = torch.randn(1, 4, 8, 8)
    # ... yet add_noise must neither raise IndexError nor produce NaNs.
    assert torch.isfinite(scheduler.add_noise(sample, noise, timestep)).all()

    v_scheduler = DDIMScheduler(prediction_type="v_prediction")
    assert torch.isfinite(v_scheduler.training_target(sample, noise, timestep)).all()


if __name__ == "__main__":
    test_bf16_timestep_does_not_overflow_alphas_cumprod()
    print("ok")
