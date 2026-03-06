#!/usr/bin/env python3
"""
Generate a prior for prior-based diffusion step skip.

Runs full inference and saves latent tensors at each denoising step.
Use infer_from_prior.py to run accelerated inference from the saved prior.

Example:
    # Image-to-video (Wan2.1-I2V-14B-480P)
    python generate_prior.py \\
        --image path/to/image.jpg \\
        --output_dir ./prior_output \\
        --num_inference_steps 10

    # With decoded videos at each step (for finding formation point)
    python generate_prior.py \\
        --image path/to/image.jpg \\
        --output_dir ./prior_output \\
        --save_decoded_videos
"""

import argparse
import os
import sys

# Ensure prior_utils is importable when run from repo root or from this directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import torch
from PIL import Image

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video

from prior_utils import build_step_callback, save_run_metadata

# Default negative prompt (Wan-style)
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args():
    p = argparse.ArgumentParser(description="Generate prior latents for step-skip inference")
    p.add_argument("--image", type=str, default=None, help="Path to input image (I2V); required unless --download_example")
    p.add_argument("--prompt", type=str, default=None, help="Text prompt (default: example prompt)")
    p.add_argument("--output_dir", type=str, default="./prior_output", help="Output directory")
    p.add_argument("--run_id", type=str, default=None, help="Run ID (default: timestamp)")
    p.add_argument("--num_inference_steps", type=int, default=10, help="Total denoising steps")
    p.add_argument("--denoising_strength", type=float, default=1.0)
    p.add_argument("--sigma_shift", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--save_decoded_videos", action="store_true", help="Decode and save video at each step")
    p.add_argument("--model", type=str, default="I2V-480P", choices=["I2V-480P", "T2V-1.3B"])
    p.add_argument("--download_example", action="store_true", help="Download example image from ModelScope")
    args = p.parse_args()
    if not args.image and not args.download_example:
        p.error("Either --image or --download_example is required")
    return args


def main():
    args = parse_args()

    if args.download_example:
        from modelscope import dataset_snapshot_download

        repo_root = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
        dataset_snapshot_download(
            dataset_id="DiffSynth-Studio/examples_in_diffsynth",
            local_dir=repo_root,
            allow_file_pattern="data/examples/wan/input_image.jpg",
        )
        args.image = os.path.join(repo_root, "data", "examples", "wan", "input_image.jpg")

    # Default prompt
    prompt = args.prompt or (
        "A small boat bravely sails through the waves. The blue sea is turbulent, "
        "white foam splashing against the hull. Sunlight reflects on the water. "
        "The camera pulls in to show the flag on the boat waving in the wind."
    )

    # Load image
    image = Image.open(args.image).convert("RGB").resize((args.width, args.height))

    # VRAM config for low-memory GPUs
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

    if args.model == "I2V-480P":
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
    else:
        # T2V-1.3B (no image)
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(
                    model_id="Wan-AI/Wan2.1-T2V-1.3B",
                    origin_file_pattern="diffusion_pytorch_model*.safetensors",
                    **vram_config,
                ),
                ModelConfig(
                    model_id="Wan-AI/Wan2.1-T2V-1.3B",
                    origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                    **vram_config,
                ),
                ModelConfig(
                    model_id="Wan-AI/Wan2.1-T2V-1.3B",
                    origin_file_pattern="Wan2.1_VAE.pth",
                    **vram_config,
                ),
            ],
            tokenizer_config=ModelConfig(
                model_id="Wan-AI/Wan2.1-T2V-1.3B",
                origin_file_pattern="google/umt5-xxl/",
            ),
            vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024**3) - 2,
        )

    # Build step callback
    make_callback, run_id = build_step_callback(
        output_dir=args.output_dir,
        run_id=args.run_id,
        save_decoded_videos=args.save_decoded_videos,
    )
    step_callback = make_callback(pipe)

    print(f"Generating prior: {args.num_inference_steps} steps -> {args.output_dir}/{run_id}")

    # Run inference with step callback
    pipe_kwargs = dict(
        prompt=prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        num_inference_steps=args.num_inference_steps,
        denoising_strength=args.denoising_strength,
        sigma_shift=args.sigma_shift,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        cfg_scale=args.cfg_scale,
        tiled=True,
        step_callback=step_callback,
    )
    if args.model == "I2V-480P":
        pipe_kwargs["input_image"] = image

    video = pipe(**pipe_kwargs)

    # Save metadata for infer_from_prior
    save_run_metadata(
        output_dir=args.output_dir,
        run_id=run_id,
        pipe=pipe,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        denoising_strength=args.denoising_strength,
        sigma_shift=args.sigma_shift,
    )

    # Save final video
    out_video_path = os.path.join(args.output_dir, run_id, "output_full.mp4")
    save_video(video, out_video_path, fps=16, quality=5)

    print(f"Done. Prior saved to {args.output_dir}/{run_id}")
    print(f"  Latents: step_0000.pt ... step_{args.num_inference_steps - 1:04d}.pt")
    print(f"  Metadata: run_metadata.json")
    print(f"  Full video: output_full.mp4")
    print(f"\nTo run accelerated inference from step 6:")
    print(f"  python examples/wanvideo/prior_based_step_skip/infer_from_prior.py \\")
    print(f"      --prior_dir {os.path.abspath(os.path.join(args.output_dir, run_id))} \\")
    print(f"      --start_step 6 --image {args.image}")


if __name__ == "__main__":
    main()
