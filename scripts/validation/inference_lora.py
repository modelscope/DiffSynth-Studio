"""
FLUX ControlNet LoRA Inference for SPAD→RGB
Based on official low-VRAM inference examples
"""
import torch, argparse
import numpy as np
from pathlib import Path
from PIL import Image
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from diffsynth.utils.controlnet import ControlNetInput


def load_spad_image(path) -> Image.Image:
    """Load a SPAD image, handling 16-bit grayscale correctly."""
    img = Image.open(path)
    if img.mode == "I;16":
        arr = np.array(img, dtype=np.float32) * (255.0 / 65535.0)
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    return img.convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="FLUX ControlNet LoRA Inference")
    parser.add_argument("--lora_checkpoint", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--control_image", type=str, required=True, help="Path to SPAD input image")
    parser.add_argument("--output", type=str, default="./output_rgb.png", help="Output path")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt (usually empty for SPAD)")
    parser.add_argument("--height", type=int, default=512, help="Output height")
    parser.add_argument("--width", type=int, default=512, help="Output width")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--embedded_guidance", type=float, default=3.5, help="Embedded guidance")
    parser.add_argument("--controlnet_scale", type=float, default=1.0, help="ControlNet conditioning scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--lora_target", type=str, default="dit", choices=["dit", "controlnet"], help="Which module to load LoRA into")
    parser.add_argument("--controlnet_fp8", action="store_true", help="Load ControlNet in FP8 (may degrade fused LoRA quality)")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"FLUX ControlNet LoRA Inference")
    print(f"{'='*60}")
    print(f"  LoRA: {args.lora_checkpoint}")
    print(f"  SPAD: {args.control_image}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}\n")
    
    # Low-VRAM config (FP8 + CPU offload)
    print("[1/4] Configuring low-VRAM mode (FP8 + CPU offload)...")
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
    
    # Model configs
    print("[2/4] Loading models...")
    controlnet_vram_config = vram_config if args.controlnet_fp8 else {}
    model_configs = [
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors", **vram_config),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors", **vram_config),
        ModelConfig(model_id="InstantX/FLUX.1-dev-Controlnet-Union-alpha", origin_file_pattern="diffusion_pytorch_model.safetensors", **controlnet_vram_config),
    ]
    
    # Load pipeline
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=model_configs,
        vram_limit=torch.cuda.mem_get_info()[1] / (1024 ** 3) - 0.5,
    )
    
    # Load LoRA
    print(f"[3/4] Loading LoRA from: {args.lora_checkpoint}")
    target_module = pipe.dit if args.lora_target == "dit" else pipe.controlnet
    if target_module is None:
        raise RuntimeError(f"Selected lora_target={args.lora_target} but module is None (controlnet not loaded?)")
    pipe.load_lora(target_module, args.lora_checkpoint, alpha=1.0)
    
    # Load SPAD image
    print(f"[4/4] Loading SPAD image: {args.control_image}")
    control_img = load_spad_image(args.control_image)
    
    # Create ControlNet input
    controlnet_inputs = [ControlNetInput(image=control_img, processor_id="gray", scale=args.controlnet_scale)]
    
    # Generate
    print(f"\nGenerating RGB output ({args.steps} steps)...")
    print(f"[Inference] Parameters: height={args.height}, width={args.width}, steps={args.steps}, cfg={args.cfg_scale}, guidance={args.embedded_guidance}, seed={args.seed}")
    output = pipe(
        prompt=args.prompt,
        input_image=None,  # ControlNet-only generation (no GT conditioning) - MUST match training
        controlnet_inputs=controlnet_inputs,  # SPAD conditioning
        denoising_strength=1.0,  # Full generation from noise - MUST match training
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg_scale,
        embedded_guidance=args.embedded_guidance,
        seed=args.seed,
        rand_device=args.device,
    )
    
    # Save
    output.save(args.output)
    print(f"\n✅ Saved to: {args.output}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
