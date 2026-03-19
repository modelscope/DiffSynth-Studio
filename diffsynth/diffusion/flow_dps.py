"""
Phase 3b: FlowDPS -- Diffusion Posterior Sampling for Rectified Flow Models

Implements inference-time physics-consistent guidance for the FLUX pipeline.
At each denoising step, we:
  1. Predict the clean image x_0 from the current state (via the velocity prediction)
  2. Decode x_0 through the VAE to pixel space
  3. Compute the measurement loss: -log p(y | D(x_0))
  4. Backpropagate to get the gradient w.r.t. latents
  5. Correct the velocity prediction (or latents) using this gradient

This is a zero-shot technique -- no retraining required.

References:
  - Chung et al., "Diffusion Posterior Sampling for General Noisy Inverse Problems" (ICLR 2023)
  - Song et al., "Pseudoinverse-Guided Diffusion Models for Inverse Problems" (ICLR 2023)
  - Adapted for rectified flow (FLUX) rather than score-based diffusion
"""

import torch
import torch.nn.functional as F
from PIL import Image
from typing import Union, Callable
from tqdm import tqdm

from .spad_forward import SPADForwardModel, SPADMeasurementConsistency


class FlowDPSConfig:
    """Configuration for FlowDPS guidance."""

    def __init__(
        self,
        spad_measurement: torch.Tensor = None,
        alpha: float = 1.0,
        num_frames: int = 1,
        guidance_scale: float = 0.1,
        guidance_schedule: str = "constant",
        start_step: int = 0,
        stop_step: int = -1,
        use_l2_loss: bool = True,
        use_nll_loss: bool = True,
        gradient_clamp: float = 1.0,
    ):
        """
        Args:
            spad_measurement: SPAD observation tensor [1, C, H, W] in [0, 1].
            alpha: SPAD forward model sensitivity.
            num_frames: Number of accumulated binary frames.
            guidance_scale: Base step size for gradient correction (eta).
            guidance_schedule: "constant", "linear_decay", "cosine".
            start_step: First step to apply guidance (0-indexed).
            stop_step: Last step to apply guidance (-1 = all steps).
            use_l2_loss: Include L2 measurement loss.
            use_nll_loss: Include Bernoulli NLL loss.
            gradient_clamp: Max gradient magnitude (for stability).
        """
        self.spad_measurement = spad_measurement
        self.alpha = alpha
        self.num_frames = num_frames
        self.guidance_scale = guidance_scale
        self.guidance_schedule = guidance_schedule
        self.start_step = start_step
        self.stop_step = stop_step
        self.use_l2_loss = use_l2_loss
        self.use_nll_loss = use_nll_loss
        self.gradient_clamp = gradient_clamp


def compute_dps_correction(
    latents: torch.Tensor,
    noise_pred: torch.Tensor,
    sigma: float,
    vae_decoder: Callable,
    spad_measurement: torch.Tensor,
    spad_model: SPADForwardModel,
    guidance_scale: float,
    gradient_clamp: float = 1.0,
    use_l2: bool = True,
    use_nll: bool = True,
    device: str = "cuda",
    tiled: bool = False,
    tile_size: int = 128,
    tile_stride: int = 64,
) -> torch.Tensor:
    """Compute the DPS gradient correction for one denoising step.

    The predicted clean sample is:
      x_0_hat = x_t - sigma * v_theta(x_t, t)
    where v_theta is the velocity (noise_pred) and sigma is the noise level.

    We decode x_0_hat through the VAE, compute the measurement loss,
    and return the gradient w.r.t. the latents.

    Args:
        latents: Current noisy latents [B, C, H, W].
        noise_pred: Predicted velocity [B, C, H, W].
        sigma: Current noise level.
        vae_decoder: VAE decoder callable.
        spad_measurement: SPAD observation [B, C, H, W] in [0, 1].
        spad_model: Differentiable SPAD forward model.
        guidance_scale: Step size for gradient correction.
        gradient_clamp: Max gradient magnitude.
        use_l2: Include L2 measurement loss.
        use_nll: Include Bernoulli NLL loss.

    Returns:
        Gradient correction to add to noise_pred.
    """
    latents_detached = latents.detach().requires_grad_(True)

    x0_hat = latents_detached - sigma * noise_pred.detach()

    decoded = vae_decoder(x0_hat, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
    decoded_01 = (decoded + 1.0) / 2.0

    intensity = decoded_01.mean(dim=1, keepdim=True)
    spad_meas = spad_measurement
    if spad_meas.shape[1] == 3:
        spad_meas = spad_meas.mean(dim=1, keepdim=True)

    loss = torch.tensor(0.0, device=device, dtype=latents.dtype)

    if use_nll:
        nll = spad_model.negative_log_likelihood(intensity, spad_meas)
        loss = loss + nll

    if use_l2:
        predicted_measurement = spad_model(intensity) / max(spad_model.num_frames, 1)
        l2 = F.mse_loss(predicted_measurement, spad_meas)
        loss = loss + l2

    grad = torch.autograd.grad(loss, latents_detached, create_graph=False)[0]

    if gradient_clamp > 0:
        grad = grad.clamp(-gradient_clamp, gradient_clamp)

    correction = -guidance_scale * grad
    return correction


def get_guidance_weight(
    progress_id: int,
    total_steps: int,
    schedule: str = "constant",
    start_step: int = 0,
    stop_step: int = -1,
) -> float:
    """Compute the guidance weight for the current step."""
    if stop_step < 0:
        stop_step = total_steps

    if progress_id < start_step or progress_id >= stop_step:
        return 0.0

    active_range = stop_step - start_step
    relative_pos = (progress_id - start_step) / max(active_range - 1, 1)

    if schedule == "constant":
        return 1.0
    elif schedule == "linear_decay":
        return 1.0 - relative_pos
    elif schedule == "cosine":
        import math
        return 0.5 * (1.0 + math.cos(math.pi * relative_pos))
    else:
        return 1.0


def flux_dps_inference(
    pipe,
    dps_config: FlowDPSConfig,
    prompt: str = "",
    negative_prompt: str = "",
    cfg_scale: float = 1.0,
    embedded_guidance: float = 3.5,
    height: int = 512,
    width: int = 512,
    seed: int = None,
    num_inference_steps: int = 28,
    denoising_strength: float = 1.0,
    controlnet_inputs=None,
    tiled: bool = False,
    tile_size: int = 128,
    tile_stride: int = 64,
    progress_bar_cmd=tqdm,
    **kwargs,
) -> Image.Image:
    """Run FLUX inference with FlowDPS measurement guidance.

    This wraps the standard FluxImagePipeline.__call__ but injects
    a gradient-based correction at each denoising step.
    """
    spad_model = SPADForwardModel(
        alpha=dps_config.alpha,
        num_frames=dps_config.num_frames,
    ).to(pipe.device)

    pipe.scheduler.set_timesteps(
        num_inference_steps,
        denoising_strength=denoising_strength,
        shift=kwargs.get("sigma_shift"),
    )

    inputs_posi = {"prompt": prompt}
    inputs_nega = {"negative_prompt": negative_prompt}
    inputs_shared = {
        "cfg_scale": cfg_scale,
        "embedded_guidance": embedded_guidance,
        "t5_sequence_length": kwargs.get("t5_sequence_length", 512),
        "input_image": kwargs.get("input_image"),
        "denoising_strength": denoising_strength,
        "height": height,
        "width": width,
        "seed": seed,
        "rand_device": kwargs.get("rand_device", "cpu"),
        "sigma_shift": kwargs.get("sigma_shift"),
        "num_inference_steps": num_inference_steps,
        "multidiffusion_prompts": (),
        "multidiffusion_masks": (),
        "multidiffusion_scales": (),
        "kontext_images": None,
        "controlnet_inputs": controlnet_inputs,
        "ipadapter_images": None,
        "ipadapter_scale": 1.0,
        "eligen_entity_prompts": None,
        "eligen_entity_masks": None,
        "eligen_enable_on_negative": False,
        "eligen_enable_inpaint": False,
        "infinityou_id_image": None,
        "infinityou_guidance": 1.0,
        "flex_inpaint_image": None,
        "flex_inpaint_mask": None,
        "flex_control_image": None,
        "flex_control_strength": 0.5,
        "flex_control_stop": 0.5,
        "value_controller_inputs": None,
        "step1x_reference_image": None,
        "nexus_gen_reference_image": None,
        "lora_encoder_inputs": None,
        "lora_encoder_scale": 1.0,
        "tea_cache_l1_thresh": None,
        "tiled": tiled,
        "tile_size": tile_size,
        "tile_stride": tile_stride,
        "progress_bar_cmd": progress_bar_cmd,
    }

    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
            unit, pipe, inputs_shared, inputs_posi, inputs_nega
        )

    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    total_steps = len(pipe.scheduler.timesteps)

    spad_meas = dps_config.spad_measurement
    if spad_meas is not None:
        spad_meas = spad_meas.to(device=pipe.device, dtype=pipe.torch_dtype)

    for progress_id, timestep in enumerate(progress_bar_cmd(pipe.scheduler.timesteps)):
        timestep_tensor = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

        noise_pred = pipe.cfg_guided_model_fn(
            pipe.model_fn,
            cfg_scale,
            inputs_shared,
            inputs_posi,
            inputs_nega,
            **models,
            timestep=timestep_tensor,
            progress_id=progress_id,
        )

        weight = get_guidance_weight(
            progress_id, total_steps,
            schedule=dps_config.guidance_schedule,
            start_step=dps_config.start_step,
            stop_step=dps_config.stop_step,
        )

        if weight > 0 and spad_meas is not None:
            sigma = pipe.scheduler.sigmas[progress_id].item()
            if sigma > 0.01:
                pipe.load_models_to_device(["vae_decoder"])
                correction = compute_dps_correction(
                    latents=inputs_shared["latents"],
                    noise_pred=noise_pred,
                    sigma=sigma,
                    vae_decoder=pipe.vae_decoder,
                    spad_measurement=spad_meas,
                    spad_model=spad_model,
                    guidance_scale=dps_config.guidance_scale * weight,
                    gradient_clamp=dps_config.gradient_clamp,
                    use_l2=dps_config.use_l2_loss,
                    use_nll=dps_config.use_nll_loss,
                    device=pipe.device,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )
                noise_pred = noise_pred + correction
                pipe.load_models_to_device(pipe.in_iteration_models)

        inputs_shared["latents"] = pipe.step(
            pipe.scheduler,
            progress_id=progress_id,
            noise_pred=noise_pred,
            **inputs_shared,
        )

    pipe.load_models_to_device(["vae_decoder"])
    image = pipe.vae_decoder(
        inputs_shared["latents"],
        device=pipe.device,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    image = pipe.vae_output_to_image(image)
    pipe.load_models_to_device([])

    return image
