"""
Per-Frame Consistency Loss (IC-Light inspired)

Given two different binary SPAD frames F1 and F2 of the same scene,
and the same noisy latent z_t, the predicted velocities should be identical
because the underlying clean image is the same.

Loss: L_consistency = ||v_theta(z_t, t, F1) - v_theta(z_t, t, F2)||^2

This is added to the standard flow-matching loss during training.

Reference: IC-Light (ICLR 2024) - light transport consistency via noise prediction alignment
"""

import torch
import torch.nn.functional as F
from .base_pipeline import BasePipeline


def FlowMatchSFTWithConsistencyLoss(
    pipe: BasePipeline,
    consistency_weight: float = 0.1,
    **inputs,
):
    """Flow-matching loss + per-frame consistency loss.

    Expects inputs to contain:
      - input_latents: VAE-encoded ground truth
      - controlnet_conditionings: list with conditioning from F1
      - controlnet_conditionings_f2: list with conditioning from F2
      - All other standard inputs (prompt_emb, etc.)

    Returns: L = L_flow_match + lambda * ||v_F1 - v_F2||^2
    """
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"])
    noisy_latents = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    inputs["latents"] = noisy_latents

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    noise_pred_f1 = pipe.model_fn(**models, **inputs, timestep=timestep)

    loss_sft = F.mse_loss(noise_pred_f1.float(), training_target.float())
    loss_sft = loss_sft * pipe.scheduler.training_weight(timestep)

    conditionings_f2 = inputs.get("controlnet_conditionings_f2")
    if conditionings_f2 is not None:
        inputs_f2 = dict(inputs)
        inputs_f2["controlnet_conditionings"] = conditionings_f2
        inputs_f2["latents"] = noisy_latents

        # Stop-gradient on F2: no activations stored, F2 prediction is a fixed
        # target. This halves memory and avoids a degenerate collapse solution.
        # Over training, F1/F2 are randomly assigned so both directions are covered.
        with torch.no_grad():
            noise_pred_f2 = pipe.model_fn(**models, **inputs_f2, timestep=timestep)

        loss_consistency = F.mse_loss(noise_pred_f1.float(), noise_pred_f2.float())
        # Apply same scheduler training weight to consistency term so the
        # effective balance between SFT and consistency is timestep-invariant.
        loss_consistency = loss_consistency * pipe.scheduler.training_weight(timestep)
        loss = loss_sft + consistency_weight * loss_consistency
    else:
        loss = loss_sft

    return loss
