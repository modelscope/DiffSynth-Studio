#!/usr/bin/env python3
"""
Run accelerated inference from a saved prior.

Loads a latent from a prior run and performs only the remaining denoising steps.
~70% fewer steps, same quality, zero retraining.

Example:
    python infer_from_prior.py \\
        --prior_dir ./prior_output/run_1234567890 \\
        --start_step 6 \\
        --image path/to/image.jpg \\
        --prompt "Different motion description"
"""

import argparse
import os
import sys

# Ensure prior_utils is importable when run from repo root or from this directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "..", ".."))

import torch
from PIL import Image

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video

from prior_utils import load_prior_metadata, validate_scheduler_match

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args():
    p = argparse.ArgumentParser(description="Infer from prior (accelerated)")
    p.add_argument("--prior_dir", type=str, required=True, help="Path to prior run directory")
    p.add_argument("--start_step", type=int, default=6, help="Step to resume from (0-indexed)")
    p.add_argument("--image", type=str, required=True, help="Input image (must match prior)")
    p.add_argument("--prompt", type=str, default=None, help="New prompt for refinement")
    p.add_argument("--output", type=str, default=None, help="Output video path")
    p.add_argument("--model", type=str, default="I2V-480P", choices=["I2V-480P"])
    return p.parse_args()


def main():
    args = parse_args()

    # Load prior metadata and validate
    meta = load_prior_metadata(args.prior_dir)
    validate_scheduler_match(
        {
            "num_inference_steps": meta["num_inference_steps"],
            "denoising_strength": meta["denoising_strength"],
            "sigma_shift": meta["sigma_shift"],
        },
        meta,
    )

    # Load prior latent
    latent_path = os.path.join(args.prior_dir, f"step_{args.start_step:04d}.pt")
    if not os.path.exists(latent_path):
        raise FileNotFoundError(f"Prior latent not found: {latent_path}")
    prior_latents = torch.load(latent_path, map_location="cpu", weights_only=True)
    prior_timesteps = torch.tensor(meta["scheduler_timesteps"], dtype=torch.float32)
    # Sigmas required for scheduler.step(); fallback: timesteps/1000 for Wan
    if "scheduler_sigmas" in meta:
        prior_sigmas = torch.tensor(meta["scheduler_sigmas"], dtype=torch.float32)
    else:
        prior_sigmas = prior_timesteps / 1000.0

    height = meta["height"]
    width = meta["width"]
    num_frames = meta["num_frames"]

    prompt = args.prompt or (
        "A small boat bravely sails through the waves. The blue sea is turbulent, "
        "white foam splashing against the hull. Sunlight reflects on the water."
    )

    image = Image.open(args.image).convert("RGB").resize((width, height))

    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                model_id="Wan-AI/Wan2.1-I2V-14B-480P",
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                **vram_config,
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.1-I2V-14B-480P",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                **vram_config,
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.1-I2V-14B-480P",
                origin_file_pattern="Wan2.1_VAE.pth",
                **vram_config,
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.1-I2V-14B-480P",
                origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                **vram_config,
            ),
        ],
        tokenizer_config=ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/",
        ),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024**3) - 2,
    )

    remaining_steps = meta["num_inference_steps"] - args.start_step - 1
    print(f"Running {remaining_steps} steps (from step {args.start_step + 1} to {meta['num_inference_steps'] - 1})")

    video = pipe(
        prompt=prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        input_image=image,
        num_inference_steps=meta["num_inference_steps"],
        denoising_strength=meta["denoising_strength"],
        sigma_shift=meta["sigma_shift"],
        height=height,
        width=width,
        num_frames=num_frames,
        cfg_scale=5.0,
        tiled=True,
        prior_latents=prior_latents,
        prior_timesteps=prior_timesteps,
        prior_sigmas=prior_sigmas,
        start_from_step=args.start_step,
    )

    out_path = args.output or os.path.join(args.prior_dir, f"output_from_step_{args.start_step}.mp4")
    save_video(video, out_path, fps=16, quality=5)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
