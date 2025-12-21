#!/usr/bin/env python
"""
Multi-GPU Inference Example for DiffSynth-Studio

This script demonstrates how to use multiple GPUs for inference.

Usage:
    # Model Parallel (distribute models across GPUs):
    python multi_gpu_inference.py --mode model --prompt "a beautiful sunset"

    # Data Parallel (same model on all GPUs, batch processing):
    torchrun --nproc_per_node=2 multi_gpu_inference.py --mode data --batch_size 4

    # Tensor Parallel (split layers across GPUs):
    torchrun --nproc_per_node=2 multi_gpu_inference.py --mode tensor
"""

import argparse
import torch
import os
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig


def check_multi_gpu():
    """Check available GPUs and print info."""
    if not torch.cuda.is_available():
        print("CUDA is not available. Multi-GPU requires CUDA.")
        return 0

    num_gpus = torch.cuda.device_count()
    print(f"\n{'='*60}")
    print(f"Multi-GPU Configuration")
    print(f"{'='*60}")
    print(f"Available GPUs: {num_gpus}")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"  GPU {i}: {props.name}")
        print(f"         Memory: {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total")

    print(f"{'='*60}\n")
    return num_gpus


def run_model_parallel(args):
    """
    Model Parallel: Distribute different model components to different GPUs.

    Best for: Large models that don't fit on a single GPU.
    """
    print("Running Model Parallel inference...")

    # Load pipeline
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda:0",  # Initial device
        model_configs=[
            ModelConfig(
                download_source="huggingface",
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
            ),
            ModelConfig(
                download_source="huggingface",
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="text_encoder/model*.safetensors",
            ),
            ModelConfig(
                download_source="huggingface",
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
            ),
        ],
        tokenizer_config=ModelConfig(
            download_source="huggingface",
            model_id="Qwen/Qwen-Image",
            origin_file_pattern="tokenizer/",
        ),
        processor_config=ModelConfig(
            download_source="huggingface",
            model_id="Qwen/Qwen-Image",
            origin_file_pattern="processor/",
        ),
    )

    # Enable multi-GPU with model parallelism
    # This will distribute dit, text_encoder, vae across available GPUs
    pipe.enable_multi_gpu(mode="model")

    # Print model distribution
    pipe.print_model_distribution()

    # Generate image
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
    )

    # Save output
    output_path = Path(args.output_dir) / "model_parallel_output.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_path))
    print(f"Saved to: {output_path}")


def run_data_parallel(args):
    """
    Data Parallel: Same model on all GPUs, process different data in parallel.

    Best for: Batch inference to maximize throughput.
    Must be launched with torchrun.
    """
    from diffsynth.distributed import (
        init_distributed,
        get_rank,
        get_world_size,
        is_main_process,
        DataParallelPipeline,
        scatter_batch,
        gather_outputs,
    )

    # Initialize distributed
    init_distributed()
    rank = get_rank()
    world_size = get_world_size()
    local_device = f"cuda:{rank}"

    print(f"[Rank {rank}/{world_size}] Running Data Parallel inference on {local_device}...")

    # Load pipeline on local device
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=local_device,
        model_configs=[
            ModelConfig(
                download_source="huggingface",
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
            ),
            ModelConfig(
                download_source="huggingface",
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="text_encoder/model*.safetensors",
            ),
            ModelConfig(
                download_source="huggingface",
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
            ),
        ],
        tokenizer_config=ModelConfig(
            download_source="huggingface",
            model_id="Qwen/Qwen-Image",
            origin_file_pattern="tokenizer/",
        ),
        processor_config=ModelConfig(
            download_source="huggingface",
            model_id="Qwen/Qwen-Image",
            origin_file_pattern="processor/",
        ),
    )

    # Create batch of prompts
    prompts = [f"{args.prompt} - variation {i+1}" for i in range(args.batch_size)]

    # Scatter prompts across ranks
    local_prompts = scatter_batch(prompts)
    print(f"[Rank {rank}] Processing {len(local_prompts)} prompts: {local_prompts}")

    # Generate images locally
    local_images = []
    for prompt in local_prompts:
        image = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
        )
        local_images.append(image)

    # Gather results on main process
    all_images = gather_outputs(local_images)

    # Save on main process
    if is_main_process():
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(all_images):
            output_path = output_dir / f"data_parallel_output_{i}.png"
            img.save(str(output_path))
            print(f"Saved to: {output_path}")


def run_tensor_parallel(args):
    """
    Tensor Parallel: Split large layers across GPUs.

    Best for: Very large models where even single components don't fit on one GPU.
    Must be launched with torchrun.
    """
    from diffsynth.distributed import (
        init_distributed,
        get_rank,
        get_world_size,
        is_main_process,
        apply_tensor_parallelism,
    )

    # Initialize distributed
    init_distributed()
    rank = get_rank()
    world_size = get_world_size()
    local_device = f"cuda:{rank}"

    print(f"[Rank {rank}/{world_size}] Running Tensor Parallel inference...")

    # Load pipeline
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=local_device,
        model_configs=[
            ModelConfig(
                download_source="huggingface",
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
            ),
            ModelConfig(
                download_source="huggingface",
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="text_encoder/model*.safetensors",
            ),
            ModelConfig(
                download_source="huggingface",
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
            ),
        ],
        tokenizer_config=ModelConfig(
            download_source="huggingface",
            model_id="Qwen/Qwen-Image",
            origin_file_pattern="tokenizer/",
        ),
        processor_config=ModelConfig(
            download_source="huggingface",
            model_id="Qwen/Qwen-Image",
            origin_file_pattern="processor/",
        ),
    )

    # Apply tensor parallelism to large linear layers in the transformer
    if hasattr(pipe, 'dit') and pipe.dit is not None:
        apply_tensor_parallelism(
            pipe.dit,
            tp_layers=["linear", "proj", "mlp"],  # Layer name patterns to parallelize
            min_features=4096,  # Only parallelize layers with >= 4096 features
        )
        print(f"[Rank {rank}] Applied tensor parallelism to DiT")

    # Generate image
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
    )

    # Save on main process
    if is_main_process():
        output_path = Path(args.output_dir) / "tensor_parallel_output.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(str(output_path))
        print(f"Saved to: {output_path}")


def run_custom_device_map(args):
    """
    Custom Device Map: Manually assign models to specific GPUs.

    Useful for fine-grained control over memory distribution.
    """
    print("Running with custom device map...")

    # Define custom device map
    device_map = {
        "dit": "cuda:0",           # DiT (largest) on GPU 0
        "text_encoder": "cuda:1",  # Text encoder on GPU 1
        "vae": "cuda:1",           # VAE on GPU 1 (smaller)
    }

    # Check if we have enough GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"Warning: Custom device map requires 2 GPUs, but only {num_gpus} available.")
        print("Falling back to single GPU...")
        device_map = {k: "cuda:0" for k in device_map}

    # Load pipeline
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda:0",  # Initial device
        model_configs=[
            ModelConfig(
                download_source="huggingface",
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
            ),
            ModelConfig(
                download_source="huggingface",
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="text_encoder/model*.safetensors",
            ),
            ModelConfig(
                download_source="huggingface",
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
            ),
        ],
        tokenizer_config=ModelConfig(
            download_source="huggingface",
            model_id="Qwen/Qwen-Image",
            origin_file_pattern="tokenizer/",
        ),
        processor_config=ModelConfig(
            download_source="huggingface",
            model_id="Qwen/Qwen-Image",
            origin_file_pattern="processor/",
        ),
    )

    # Apply custom device map
    pipe.enable_multi_gpu(mode="model", device_map=device_map)

    # Print distribution
    pipe.print_model_distribution()

    # Generate image
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
    )

    # Save output
    output_path = Path(args.output_dir) / "custom_device_map_output.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_path))
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU inference example")

    parser.add_argument(
        "--mode",
        type=str,
        default="model",
        choices=["model", "data", "tensor", "custom"],
        help="Parallelism mode: model, data, tensor, or custom",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a beautiful sunset over the ocean",
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="blurry, ugly, bad quality",
        help="Negative prompt",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for data parallel mode",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory",
    )

    args = parser.parse_args()

    # Check GPUs
    num_gpus = check_multi_gpu()

    if num_gpus == 0:
        print("No GPUs available. Exiting.")
        return

    # Run selected mode
    if args.mode == "model":
        run_model_parallel(args)
    elif args.mode == "data":
        run_data_parallel(args)
    elif args.mode == "tensor":
        run_tensor_parallel(args)
    elif args.mode == "custom":
        run_custom_device_map(args)


if __name__ == "__main__":
    main()
