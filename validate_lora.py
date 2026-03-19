"""
FLUX LoRA Validation - Run inference on validation set
"""
import argparse
import torch
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import csv

from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from diffsynth.utils.controlnet import ControlNetInput
from diffsynth.utils.lora.flux import FluxLoRALoader
from diffsynth.core import load_state_dict

def main():
    parser = argparse.ArgumentParser(description="FLUX LoRA validation on val set")
    parser.add_argument("--lora_checkpoint", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--lora_target", type=str, default="dit", choices=["dit", "controlnet"], help="Which module to load LoRA into")
    parser.add_argument("--metadata_csv", type=str, default="/home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv", help="Validation CSV")
    parser.add_argument("--output_dir", type=str, default="./validation_outputs", help="Output directory")
    parser.add_argument("--steps", type=int, default=28, help="Inference steps")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--embedded_guidance", type=float, default=3.5, help="Embedded guidance")
    parser.add_argument("--denoising_strength", type=float, default=1.0, help="Denoising strength")
    parser.add_argument("--height", type=int, default=512, help="Output height (match training resolution)")
    parser.add_argument("--width", type=int, default=512, help="Output width (match training resolution)")
    parser.add_argument("--controlnet_scale", type=float, default=1.0, help="ControlNet conditioning scale")
    parser.add_argument("--processor_id", type=str, default="gray", help="ControlNet processor id for SPAD inputs")
    parser.add_argument("--controlnet_fp8", action="store_true", help="Load ControlNet in FP8 (may degrade fused LoRA quality)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples (None=all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs instead of skipping")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "input").mkdir(exist_ok=True)
    (output_dir / "output").mkdir(exist_ok=True)
    (output_dir / "ground_truth").mkdir(exist_ok=True)
    
    print("="*60)
    print("FLUX LoRA Validation")
    print("="*60)
    print(f"Checkpoint: {args.lora_checkpoint}")
    print(f"Validation CSV: {args.metadata_csv}")
    print(f"Output: {output_dir}")
    print("="*60)
    print()
    
    # Load validation samples
    print(f"Loading validation metadata...")
    samples = []
    with open(args.metadata_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
    
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    print(f"Processing {len(samples)} validation samples")
    
    # Load pipeline
    print("Loading FLUX pipeline...")
    vram_config = {
        "offload_dtype": torch.float8_e4m3fn,
        "offload_device": "cpu",
        "onload_dtype": torch.float8_e4m3fn,
        "onload_device": "cpu",
        "preparing_dtype": torch.float8_e4m3fn,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    
    vram_limit = torch.cuda.mem_get_info()[1] / (1024 ** 3) - 0.5
    controlnet_vram_config = vram_config if args.controlnet_fp8 else {}
    model_configs = [
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors", **vram_config),
        ModelConfig(model_id="InstantX/FLUX.1-dev-Controlnet-Union-alpha", origin_file_pattern="diffusion_pytorch_model.safetensors", **controlnet_vram_config),
    ]
    
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        vram_limit=vram_limit,
    )
    
    print(f"Loading LoRA ({args.lora_target}): {args.lora_checkpoint}")
    target_module = pipe.dit if args.lora_target == "dit" else pipe.controlnet
    if target_module is None:
        raise RuntimeError(f"Selected lora_target={args.lora_target} but module is None (controlnet not loaded?)")
    if args.lora_target == "controlnet":
        # Debug: compute match counts before fuse
        state_dict = load_state_dict(args.lora_checkpoint, torch_dtype=pipe.torch_dtype, device=pipe.device)
        # Use FluxLoRALoader directly to fuse and see match count
        loader = FluxLoRALoader(torch_dtype=pipe.torch_dtype, device=pipe.device)
        # Compute lora layer names
        name_dict = loader.get_name_dict(state_dict)
        lora_layer_names = set([n for n in name_dict])
        module_names = set([n for n, _ in target_module.named_modules()])
        matched = lora_layer_names & module_names
        print(f"[LoRA Debug] controlnet modules: {len(module_names)}, lora layers: {len(lora_layer_names)}, intersect: {len(matched)}")
        loader.fuse_lora_to_base_model(target_module, state_dict, alpha=1.0)
    else:
        pipe.load_lora(target_module, args.lora_checkpoint, alpha=1.0)
    
    # Run inference
    print("Running inference on validation set...")
    csv_base = Path(args.metadata_csv).parent
    
    for idx, sample in enumerate(tqdm(samples, desc="Generating")):
        output_path = output_dir / "output" / f"output_{idx:04d}.png"
        if output_path.exists() and not args.overwrite:
            continue
        # Load SPAD input and ground truth
        controlnet_key = "controlnet_image" if "controlnet_image" in sample else "input_image"
        if controlnet_key not in sample:
            raise KeyError(f"Missing control image column in CSV. Expected 'controlnet_image' (or legacy 'input_image'), got keys: {list(sample.keys())}")
        
        input_path = csv_base / sample[controlnet_key]
        gt_path = csv_base / sample['image']
        
        control_img = Image.open(input_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        
        # Generate
        with torch.no_grad():
            result = pipe(
                prompt=sample.get('prompt', ''),
                controlnet_inputs=[ControlNetInput(
                    image=control_img,
                    processor_id=args.processor_id,
                    scale=args.controlnet_scale,
                )],
                denoising_strength=args.denoising_strength,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                cfg_scale=args.cfg_scale,
                embedded_guidance=args.embedded_guidance,
                seed=args.seed + idx,  # Different seed per sample
                rand_device="cuda",
            )
        
        # Save
        control_img.save(output_dir / "input" / f"input_{idx:04d}.png")
        result.save(output_path)
        gt_img.save(output_dir / "ground_truth" / f"gt_{idx:04d}.png")
    
    print()
    print("✅ Validation complete!")
    print(f"   Input: {output_dir / 'input'}")
    print(f"   Output: {output_dir / 'output'}")
    print(f"   Ground truth: {output_dir / 'ground_truth'}")


if __name__ == "__main__":
    main()
