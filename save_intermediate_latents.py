#!/usr/bin/env python3
"""
Phase 2e: Save and decode intermediate latents at selected denoising steps.

For selected validation images, saves the intermediate decoded images
at steps [1, 5, 10, 20, 30, 40, 50] (or nearest available) to visualize
when structure locks in during the denoising process.

Usage:
  python save_intermediate_latents.py \
    --lora_checkpoint /path/to/lora.safetensors \
    --metadata_csv /path/to/metadata_val.csv \
    --output-dir intermediate_latents/
"""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import csv

from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from diffsynth.utils.controlnet import ControlNetInput
from diffsynth.utils.lora.flux import FluxLoRALoader
from diffsynth.core import load_state_dict


def main():
    parser = argparse.ArgumentParser(description="Save intermediate denoising latents")
    parser.add_argument("--lora_checkpoint", type=str, required=True)
    parser.add_argument("--lora_target", type=str, default="controlnet")
    parser.add_argument("--metadata_csv", type=str, default="/home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv")
    parser.add_argument("--output-dir", type=str, default="./intermediate_latents")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--save-at", type=int, nargs="+", default=[1, 3, 5, 7, 10, 14, 20, 27],
                        help="Step indices at which to decode and save (0-indexed)")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=20)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading pipeline...")
    pipe = FluxImagePipeline.from_pretrained(
        [
            ModelConfig("black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors"),
            ModelConfig("black-forest-labs/FLUX.1-dev", "text_encoder/model.safetensors"),
            ModelConfig("black-forest-labs/FLUX.1-dev", "text_encoder_2/*.safetensors"),
            ModelConfig("black-forest-labs/FLUX.1-dev", "ae.safetensors"),
            ModelConfig("InstantX/FLUX.1-dev-Controlnet-Union-alpha", "diffusion_pytorch_model.safetensors"),
        ],
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

    if args.lora_target == "controlnet":
        lora_loader = FluxLoRALoader(torch_dtype=torch.bfloat16, device="cuda")
        lora_sd = load_state_dict(args.lora_checkpoint, torch_dtype=torch.bfloat16, device="cuda")
        lora_sd = lora_loader.convert_state_dict(lora_sd)
        lora_loader.fuse_lora(pipe.controlnet, lora_sd, alpha=1.0)
    else:
        pipe.load_lora(pipe.dit, args.lora_checkpoint, alpha=1.0)

    csv_path = Path(args.metadata_csv)
    with open(csv_path) as f:
        samples = list(csv.DictReader(f))
    if args.max_samples:
        samples = samples[:args.max_samples]

    save_steps = set(args.save_at)

    for idx, sample in enumerate(tqdm(samples, desc="Processing images")):
        sample_dir = out / f"sample_{idx:04d}"
        sample_dir.mkdir(exist_ok=True)

        control_key = "controlnet_image" if "controlnet_image" in sample else "input_image"
        control_path = csv_path.parent / sample[control_key]
        gt_path = csv_path.parent / sample["image"]

        control_img = Image.open(control_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        control_img.save(sample_dir / "input.png")
        gt_img.save(sample_dir / "ground_truth.png")

        controlnet_input = ControlNetInput(image=control_img, processor_id="gray", scale=1.0)

        pipe.scheduler.set_timesteps(args.steps, denoising_strength=1.0)
        inputs_posi = {"prompt": ""}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            "cfg_scale": 1.0, "embedded_guidance": 3.5, "t5_sequence_length": 512,
            "input_image": None, "denoising_strength": 1.0,
            "height": args.height, "width": args.width,
            "seed": args.seed + idx, "rand_device": "cpu",
            "sigma_shift": None, "num_inference_steps": args.steps,
            "multidiffusion_prompts": (), "multidiffusion_masks": (), "multidiffusion_scales": (),
            "kontext_images": None, "controlnet_inputs": [controlnet_input],
            "ipadapter_images": None, "ipadapter_scale": 1.0,
            "eligen_entity_prompts": None, "eligen_entity_masks": None,
            "eligen_enable_on_negative": False, "eligen_enable_inpaint": False,
            "infinityou_id_image": None, "infinityou_guidance": 1.0,
            "flex_inpaint_image": None, "flex_inpaint_mask": None,
            "flex_control_image": None, "flex_control_strength": 0.5, "flex_control_stop": 0.5,
            "value_controller_inputs": None,
            "step1x_reference_image": None, "nexus_gen_reference_image": None,
            "lora_encoder_inputs": None, "lora_encoder_scale": 1.0,
            "tea_cache_l1_thresh": None,
            "tiled": False, "tile_size": 128, "tile_stride": 64,
            "progress_bar_cmd": lambda x: x,
        }

        for unit in pipe.units:
            inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
                unit, pipe, inputs_shared, inputs_posi, inputs_nega
            )

        pipe.load_models_to_device(pipe.in_iteration_models)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

        with torch.no_grad():
            for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
                timestep_t = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
                noise_pred = pipe.cfg_guided_model_fn(
                    pipe.model_fn, 1.0,
                    inputs_shared, inputs_posi, inputs_nega,
                    **models, timestep=timestep_t, progress_id=progress_id,
                )
                inputs_shared["latents"] = pipe.step(
                    pipe.scheduler, progress_id=progress_id,
                    noise_pred=noise_pred, **inputs_shared,
                )

                if progress_id in save_steps:
                    pipe.load_models_to_device(["vae_decoder"])
                    decoded = pipe.vae_decoder(
                        inputs_shared["latents"], device=pipe.device
                    )
                    img = pipe.vae_output_to_image(decoded)
                    img.save(sample_dir / f"step_{progress_id:03d}.png")
                    pipe.load_models_to_device(pipe.in_iteration_models)

            # Final image
            pipe.load_models_to_device(["vae_decoder"])
            decoded = pipe.vae_decoder(inputs_shared["latents"], device=pipe.device)
            final_img = pipe.vae_output_to_image(decoded)
            final_img.save(sample_dir / "final.png")

    print(f"\nDone! Results saved to {out}")


if __name__ == "__main__":
    main()
