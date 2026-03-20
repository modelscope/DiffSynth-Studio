"""
Utilities for prior-based diffusion step skip.

Saves latent tensors at each denoising step and metadata required for resuming inference.
"""

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch


def build_step_callback(
    output_dir: str,
    run_id: Optional[str] = None,
    save_decoded_videos: bool = False,
) -> Tuple[Callable, str]:
    """
    Build a step_callback for WanVideoPipeline that saves latents at each step.

    Args:
        output_dir: Directory to save latents (e.g. ./prior_output)
        run_id: Optional run identifier; defaults to timestamp-based
        save_decoded_videos: If True, decode latents to video at each step (for inspection).
            Requires pipe to be passed via closure; we return a factory.

    Returns:
        (callback_factory, run_id): A function that takes (pipe) and returns the actual
        step_callback. The caller must pass the pipe so we can decode if requested.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if run_id is None:
        run_id = f"run_{int(time.time())}"

    run_dir = output_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    def make_callback(pipe) -> Callable:
        def step_callback(step_index: int, latents: torch.Tensor, timestep: torch.Tensor) -> None:
            # Save latent
            latent_path = run_dir / f"step_{step_index:04d}.pt"
            torch.save(latents.cpu(), latent_path)

            # Optionally decode and save video for inspection
            if save_decoded_videos and pipe is not None:
                pipe.load_models_to_device(["vae"])
                video = pipe.vae.decode(
                    latents,
                    device=pipe.device,
                    tiled=getattr(pipe, "_prior_tiled", True),
                    tile_size=getattr(pipe, "_prior_tile_size", (30, 52)),
                    tile_stride=getattr(pipe, "_prior_tile_stride", (15, 26)),
                )
                video_frames = pipe.vae_output_to_video(video)
                video_path = run_dir / f"step_{step_index:04d}.mp4"
                _save_video_frames(video_frames, str(video_path), fps=16)
                pipe.load_models_to_device([])

        return step_callback

    return make_callback, run_id


def save_run_metadata(
    output_dir: str,
    run_id: str,
    pipe,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    denoising_strength: Optional[float] = None,
    sigma_shift: Optional[float] = None,
    **extra: Any,
) -> None:
    """Save run_metadata.json for prior/inference compatibility checks."""
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    timesteps = (
        pipe.scheduler.timesteps.cpu().tolist()
        if hasattr(pipe.scheduler.timesteps, "cpu")
        else list(pipe.scheduler.timesteps)
    )
    sigmas = (
        pipe.scheduler.sigmas.cpu().tolist()
        if hasattr(pipe.scheduler.sigmas, "cpu")
        else list(pipe.scheduler.sigmas)
    )

    metadata = {
        "run_id": run_id,
        "num_inference_steps": len(timesteps),
        "scheduler_timesteps": timesteps,
        "scheduler_sigmas": sigmas,
        "denoising_strength": denoising_strength if denoising_strength is not None else 1.0,
        "sigma_shift": sigma_shift if sigma_shift is not None else 5.0,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        **extra,
    }

    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_prior_metadata(prior_dir: str) -> dict:
    """Load run_metadata.json from a prior run directory."""
    path = Path(prior_dir) / "run_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_scheduler_match(current_config: dict, prior_metadata: dict) -> None:
    """
    Validate that current inference config matches the prior's scheduler.
    Raises ValueError if mismatch.
    """
    for key in ("num_inference_steps", "denoising_strength", "sigma_shift"):
        curr = current_config.get(key)
        prior = prior_metadata.get(key)
        if curr is not None and prior is not None and curr != prior:
            raise ValueError(
                f"Scheduler mismatch: {key} is {curr} but prior used {prior}. "
                "Prior and inference must use identical scheduler parameters."
            )


def _save_video_frames(frames: list, path: str, fps: int = 16) -> None:
    """Save list of PIL Images to MP4."""
    import imageio
    import numpy as np

    writer = imageio.get_writer(path, fps=fps, quality=8)
    for frame in frames:
        arr = np.array(frame) if hasattr(frame, "size") else frame
        writer.append_data(arr)
    writer.close()
